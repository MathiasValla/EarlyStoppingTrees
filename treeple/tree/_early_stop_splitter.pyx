# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# Early stopping splitters for CART (dense only).

import numpy as np
from libc.math cimport exp, log, sqrt, fabs
from libc.stdlib cimport qsort, malloc, free
from libc.string cimport memcpy

from .._lib.sklearn.tree._criterion cimport Criterion, RegressionCriterion
from .._lib.sklearn.tree._partitioner cimport DensePartitioner, shift_missing_values_to_left_if_required
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._tree cimport ParentInfo
from .._lib.sklearn.tree._utils cimport RAND_R_MAX, rand_int, rand_uniform
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int8_t, intp_t, uint32_t

cdef float64_t INFINITY = np.inf
cdef float32_t FEATURE_THRESHOLD = 1e-7
cdef float64_t ONE_OVER_E = 0.36787944117144233  # 1/e
cdef intp_t N_GAIN_SAMPLES = 256  # max gains to sample for prophet τ
cdef intp_t MAX_FEATURES_BACKUP = 4096  # max n_features for fallback / selection_list in S+par

# Normal quantile (ppf) approximation for parametric tau; Moro/Beasley-Springer style, nogil.
cdef inline float64_t _norm_ppf(float64_t p) noexcept nogil:
    cdef float64_t q, t, z
    cdef float64_t c0 = 2.515517, c1 = 0.802853, c2 = 0.010328
    cdef float64_t d1 = 1.432788, d2 = 0.189269, d3 = 0.001308
    if p <= 1e-10:
        return -INFINITY
    if p >= 1.0 - 1e-10:
        return INFINITY
    if p <= 0.5:
        q = p
    else:
        q = 1.0 - p
    t = sqrt(-2.0 * log(q))
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    if p < 0.5:
        z = -z
    return z


def _test_norm_ppf(float64_t p):
    """Private test hook for the C-level normal quantile approximation."""
    return _norm_ppf(p)

# Standard normal CDF Φ(z), A&S 26.2.17 style, nogil.
cdef inline float64_t _norm_cdf(float64_t z) noexcept nogil:
    cdef float64_t t, phi_z
    cdef float64_t b1 = 0.319381530, b2 = -0.356563782, b3 = 1.781477937
    cdef float64_t b4 = -1.821255978, b5 = 1.330274429, p = 0.2316419
    cdef float64_t inv_sqrt_2pi = 0.3989422804014327
    if z <= -8.0:
        return 0.0
    if z >= 8.0:
        return 1.0
    phi_z = inv_sqrt_2pi * exp(-0.5 * z * z)
    t = 1.0 / (1.0 + p * fabs(z))
    t = phi_z * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    if z >= 0.0:
        return 1.0 - t
    return t

# CDF of noncentral chi-square with df=1 and noncentrality λ: χ²₁(λ). For x>0, P(X<=x) = Φ(√x - √λ) - Φ(-√x - √λ).
cdef inline float64_t _ncx2_1_cdf(float64_t x, float64_t lam) noexcept nogil:
    cdef float64_t sx, sl
    if x <= 0.0:
        return 0.0
    sx = sqrt(x)
    sl = sqrt(lam)
    return _norm_cdf(sx - sl) - _norm_cdf(-sx - sl)

# α-quantile of χ²₁(λ) by bisection (x in (0, max_x] such that CDF(x)=α).
cdef inline float64_t _ncx2_1_ppf(float64_t alpha, float64_t lam) noexcept nogil:
    cdef float64_t x_lo = 1e-12, x_hi = 50.0, x_mid, cdf
    cdef int i
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return x_hi
    for i in range(80):
        x_mid = 0.5 * (x_lo + x_hi)
        cdf = _ncx2_1_cdf(x_mid, lam)
        if cdf < alpha:
            x_lo = x_mid
        else:
            x_hi = x_mid
        if x_hi - x_lo < 1e-10:
            break
    return 0.5 * (x_lo + x_hi)

# MSE regression: G = (σ²/n) * χ²₁(λ). Fit scale s=σ²/n and λ from mean m and variance v: m = s(1+λ), v = s²*2(1+2λ). Solve for λ then s.
# Return tau = s * ncx2_1_ppf(alpha, λ). Fallback to empirical if variance invalid.
# Single-pass mean and variance: var = E[X^2] - E[X]^2.
cdef inline float64_t _quantile_ncx2_mse(
    float64_t* gains,
    intp_t n,
    float64_t alpha,
    intp_t empirical_idx,
) noexcept nogil:
    cdef float64_t mean = 0.0, sum_sq = 0.0, var = 0.0, lam, scale, disc, q
    cdef intp_t i
    if n <= 0:
        return -INFINITY
    for i in range(n):
        mean += gains[i]
        sum_sq += gains[i] * gains[i]
    mean = mean / n
    var = (sum_sq / n) - (mean * mean)
    if var < 0.0:
        var = 0.0
    if mean <= 1e-12 or var <= 0.0:
        return gains[min(empirical_idx, n - 1)]
    disc = 16.0 * mean * mean * mean * mean - 8.0 * mean * mean * var
    if disc < 0.0:
        lam = 0.0
    else:
        lam = ((4.0 * mean * mean - 2.0 * var) + sqrt(disc)) / (2.0 * var)
        if lam < 0.0:
            lam = 0.0
    scale = mean / (1.0 + lam)
    q = _ncx2_1_ppf(alpha, lam)
    return scale * q

# Parametric alpha-quantile: criterion_kind 0=empirical, 1=regression (exact scaled ncx2 for MSE), 2=classification (Gaussian approx for Gamma).
# Single-pass mean and variance for criterion_kind==2.
cdef inline float64_t _quantile_parametric(
    float64_t* gains,
    intp_t n,
    float64_t alpha,
    int criterion_kind,
    intp_t empirical_idx,
) noexcept nogil:
    """Compute alpha-quantile: 0=empirical, 1=MSE scaled ncx2, 2=classification Gaussian."""
    cdef float64_t mean = 0.0, sum_sq = 0.0, var = 0.0, std, tau
    cdef intp_t i
    if n <= 0:
        return -INFINITY
    if criterion_kind == 0:
        return gains[min(empirical_idx, n - 1)]
    if criterion_kind == 1:
        return _quantile_ncx2_mse(gains, n, alpha, empirical_idx)
    for i in range(n):
        mean += gains[i]
        sum_sq += gains[i] * gains[i]
    mean = mean / n
    var = (sum_sq / n) - (mean * mean)
    if var < 0.0:
        var = 0.0
    std = sqrt(var)
    if std < 1e-12 or var <= 0.0:
        return gains[min(empirical_idx, n - 1)]
    tau = mean + _norm_ppf(alpha) * std
    return tau


cdef inline float64_t _mse_centered_proxy(Criterion criterion) noexcept nogil:
    """Return the MSE split gain on a translation-invariant proxy scale."""
    cdef float64_t centered_left
    cdef float64_t centered_right
    cdef float64_t gain = 0.0
    cdef intp_t k
    if (<RegressionCriterion>criterion).weighted_n_node_samples <= 0.0:
        return 0.0
    for k in range((<RegressionCriterion>criterion).n_outputs):
        centered_left = (
            (<RegressionCriterion>criterion).sum_left[k]
            - (<RegressionCriterion>criterion).weighted_n_left
            / (<RegressionCriterion>criterion).weighted_n_node_samples
            * (<RegressionCriterion>criterion).sum_total[k]
        )
        centered_right = (
            (<RegressionCriterion>criterion).sum_right[k]
            - (<RegressionCriterion>criterion).weighted_n_right
            / (<RegressionCriterion>criterion).weighted_n_node_samples
            * (<RegressionCriterion>criterion).sum_total[k]
        )
        gain += (
            centered_left * centered_left
            / (<RegressionCriterion>criterion).weighted_n_left
            + centered_right * centered_right
            / (<RegressionCriterion>criterion).weighted_n_right
        )
    return gain


cdef inline void _init_split_record(SplitRecord* s, intp_t start_pos) noexcept nogil:
    s.impurity_left = INFINITY
    s.impurity_right = INFINITY
    s.pos = start_pos
    s.feature = 0
    s.threshold = 0.0
    s.improvement = -INFINITY
    s.missing_go_to_left = 0
    s.n_missing = 0


cdef inline void _sync_constant_features(
    intp_t[::1] features,
    intp_t[::1] constant_features,
    intp_t n_known_constants,
    intp_t n_found_constants,
) noexcept nogil:
    """Mirror sklearn's constant-feature bookkeeping for sibling/child nodes."""
    cdef intp_t n_total_constants = n_known_constants + n_found_constants

    if n_found_constants > 0:
        memcpy(
            &constant_features[n_known_constants],
            &features[n_known_constants],
            sizeof(intp_t) * n_found_constants,
        )
    if n_total_constants > 0:
        memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_total_constants)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

# Effective secretary exploration fraction: 1/e, 1/sqrt(n), or user float.
cdef inline float64_t _effective_explore_frac(
    float64_t explore_frac,
    bint use_sqrt_n,
    intp_t n_node,
) noexcept nogil:
    if use_sqrt_n and n_node > 0:
        return 1.0 / sqrt(<float64_t>n_node)
    if explore_frac > 0.0 and explore_frac <= 1.0:
        return explore_frac
    return ONE_OVER_E


cdef inline intp_t _exploration_budget_from_frac(
    float64_t eff_frac,
    intp_t budget_total,
) noexcept nogil:
    cdef intp_t n_explore

    if budget_total <= 0:
        return 0
    n_explore = <intp_t>(eff_frac * budget_total + 0.5)
    if n_explore < 1:
        n_explore = 1
    if n_explore > budget_total:
        n_explore = budget_total
    return n_explore


cdef class BaseEarlyStopSplitter(Splitter):
    """Base for early-stopping axis-aligned splitters. Stores dense X in init."""

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
        *argv,
        **kwargs
    ):
        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None
        self.explore_frac = float(kwargs.get("explore_frac", -1.0))
        self.use_sqrt_n = bool(kwargs.get("use_sqrt_n", False))
        self.n_split_calls = 0
        self.n_threshold_candidates = 0
        self.n_gain_evaluations = 0
        self.n_parametric_gain_samples = 0
        self.n_parametric_quantile_fits = 0

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        self.X = X_arr
        self.partitioner = DensePartitioner(
            X_arr,
            self.samples,
            self.feature_values,
            missing_values_in_feature_mask,
        )
        self.explore_rand_r_state = self.rand_r_state
        self.n_split_calls = 0
        self.n_threshold_candidates = 0
        self.n_gain_evaluations = 0
        self.n_parametric_gain_samples = 0
        self.n_parametric_quantile_fits = 0
        return 0

    def get_stats(self):
        split_calls = int(self.n_split_calls)
        threshold_candidates = int(self.n_threshold_candidates)
        gain_evaluations = int(self.n_gain_evaluations)
        parametric_gain_samples = int(self.n_parametric_gain_samples)
        parametric_quantile_fits = int(self.n_parametric_quantile_fits)
        denom = split_calls if split_calls > 0 else 1
        return {
            "split_calls": split_calls,
            "threshold_candidates": threshold_candidates,
            "gain_evaluations": gain_evaluations,
            "threshold_candidates_per_split": threshold_candidates / denom,
            "gain_evaluations_per_split": gain_evaluations / denom,
            "parametric_gain_samples": parametric_gain_samples,
            "parametric_quantile_fits": parametric_quantile_fits,
        }

    def __reduce__(self):
        return (type(self), (
            self.criterion,
            self.max_features,
            self.min_samples_leaf,
            self.min_weight_leaf,
            self.random_state,
            np.asarray(self.monotonic_cst) if self.monotonic_cst is not None else None,
        ), self.__getstate__())


# ---------------------------------------------------------------------------
# Helpers: evaluate split at (feat, pos); copy feature values and sort
# ---------------------------------------------------------------------------

cdef inline float64_t _eval_split(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t pos,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    SplitRecord* out,
) noexcept nogil:
    """Evaluate split at (feat, pos). Fill out and return proxy_impurity_improvement or -INFINITY if invalid."""
    splitter.n_threshold_candidates += 1
    if (pos - start) < splitter.min_samples_leaf or (end - pos) < splitter.min_samples_leaf:
        return -INFINITY
    splitter.criterion.update(pos)
    if splitter.criterion.weighted_n_left < min_weight_leaf or splitter.criterion.weighted_n_right < min_weight_leaf:
        return -INFINITY
    cdef float64_t imp = splitter.criterion.proxy_impurity_improvement()
    splitter.n_gain_evaluations += 1
    if imp <= -INFINITY:
        return -INFINITY
    out.feature = feat
    out.pos = pos
    out.n_missing = splitter.partitioner.n_missing
    out.missing_go_to_left = 0
    out.threshold = splitter.feature_values[pos - 1] / 2.0 + splitter.feature_values[pos] / 2.0
    if out.threshold == splitter.feature_values[pos] or out.threshold >= INFINITY or out.threshold <= -INFINITY:
        out.threshold = splitter.feature_values[pos - 1]
    return imp


cdef inline void _fill_and_sort_feature(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
) noexcept nogil:
    splitter.partitioner.sort_samples_and_feature_values(feat)


cdef inline bint _sorted_feature_is_constant(
    BaseEarlyStopSplitter splitter,
    intp_t start,
    intp_t end,
) noexcept nogil:
    cdef intp_t end_non_missing = end - splitter.partitioner.n_missing
    if end_non_missing == start:
        return 1
    return splitter.feature_values[end_non_missing - 1] <= splitter.feature_values[start] + FEATURE_THRESHOLD


cdef inline float64_t _eval_split_at_position(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t pos,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    SplitRecord* out,
) noexcept nogil:
    splitter.criterion.reset()
    return _eval_split(splitter, feat, pos, start, end, min_weight_leaf, out)


cdef inline float64_t _draw_extratree_split_gain_from_minmax(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    uint32_t* random_state,
    float32_t min_feature_value,
    float32_t max_feature_value,
    SplitRecord* out,
) noexcept nogil:
    """Evaluate one ExtraTree-style continuous threshold draw without sorting."""
    cdef float64_t threshold
    cdef float64_t cur_imp
    cdef intp_t pos

    _init_split_record(out, end)
    threshold = rand_uniform(min_feature_value, max_feature_value, random_state)
    if threshold == max_feature_value:
        threshold = min_feature_value

    pos = splitter.partitioner.partition_samples(threshold)
    splitter.n_threshold_candidates += 1
    if (pos - start) < splitter.min_samples_leaf or (end - pos) < splitter.min_samples_leaf:
        return -INFINITY

    splitter.criterion.reset()
    splitter.criterion.update(pos)
    if splitter.criterion.weighted_n_left < min_weight_leaf or splitter.criterion.weighted_n_right < min_weight_leaf:
        return -INFINITY

    cur_imp = splitter.criterion.proxy_impurity_improvement()
    splitter.n_gain_evaluations += 1
    if cur_imp <= -INFINITY:
        return -INFINITY

    out.feature = feat
    out.pos = pos
    out.threshold = threshold
    out.n_missing = splitter.partitioner.n_missing
    out.missing_go_to_left = 0
    out.improvement = cur_imp
    return cur_imp


cdef inline bint _draw_extratree_threshold_from_sorted_feature(
    BaseEarlyStopSplitter splitter,
    intp_t start,
    intp_t end,
    uint32_t* random_state,
    float64_t* threshold_out,
    intp_t* pos_out,
) noexcept nogil:
    """Mirror ExtraTree threshold sampling on the already-sorted feature stream."""
    cdef intp_t end_non_missing = end - splitter.partitioner.n_missing
    cdef float64_t threshold
    cdef intp_t lo
    cdef intp_t hi
    cdef intp_t mid

    if end_non_missing <= start:
        return 0

    threshold = rand_uniform(
        splitter.feature_values[start],
        splitter.feature_values[end_non_missing - 1],
        random_state,
    )
    if threshold == splitter.feature_values[end_non_missing - 1]:
        threshold = splitter.feature_values[start]

    lo = start
    hi = end_non_missing
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if splitter.feature_values[mid] <= threshold:
            lo = mid + 1
        else:
            hi = mid
    threshold_out[0] = threshold
    pos_out[0] = lo
    return 1


cdef inline intp_t _draw_extratree_split_batch(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    uint32_t* random_state,
    intp_t n_draws,
    float64_t* gain_buf,
    intp_t gain_buf_len,
    SplitRecord* best_in_feature,
    int criterion_kind,
) noexcept nogil:
    """Sample continuous ExtraTree-style thresholds in one batch for a sorted feature.

    Thresholds are generated once, sorted once, then matched against the sorted
    feature values with a single monotone scan. This keeps the same threshold
    distribution as repeated ExtraTree draws while removing repeated
    threshold-to-position scans.
    """
    cdef intp_t end_non_missing = end - splitter.partitioner.n_missing
    cdef float64_t min_feature_value
    cdef float64_t max_feature_value
    cdef float64_t threshold_stack[256]
    cdef float64_t* thresholds = threshold_stack
    cdef float64_t* thresholds_malloc = NULL
    cdef float64_t cur_imp
    cdef float64_t best_imp = -INFINITY
    cdef float64_t threshold
    cdef SplitRecord cur
    cdef intp_t draw_idx
    cdef intp_t pos
    cdef intp_t prev_pos
    cdef intp_t n_valid = 0

    if n_draws <= 0 or end_non_missing <= start:
        if best_in_feature != NULL:
            _init_split_record(best_in_feature, end)
        return 0

    min_feature_value = splitter.feature_values[start]
    max_feature_value = splitter.feature_values[end_non_missing - 1]
    if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
        if best_in_feature != NULL:
            _init_split_record(best_in_feature, end)
        return 0

    if n_draws > 256:
        thresholds_malloc = <float64_t*>malloc(n_draws * sizeof(float64_t))
        if thresholds_malloc == NULL:
            if best_in_feature != NULL:
                _init_split_record(best_in_feature, end)
            return 0
        thresholds = thresholds_malloc

    if best_in_feature != NULL:
        _init_split_record(best_in_feature, end)

    for draw_idx in range(n_draws):
        threshold = rand_uniform(min_feature_value, max_feature_value, random_state)
        if threshold == max_feature_value:
            threshold = min_feature_value
        thresholds[draw_idx] = threshold

    qsort(thresholds, n_draws, sizeof(float64_t), _cmp_float64)

    pos = start
    prev_pos = start
    splitter.criterion.reset()
    for draw_idx in range(n_draws):
        threshold = thresholds[draw_idx]
        while pos < end_non_missing and splitter.feature_values[pos] <= threshold:
            pos += 1
        splitter.n_threshold_candidates += 1
        if (pos - start) < splitter.min_samples_leaf or (end - pos) < splitter.min_samples_leaf:
            continue
        if pos > prev_pos:
            splitter.criterion.update(pos)
            prev_pos = pos
        if splitter.criterion.weighted_n_left < min_weight_leaf or splitter.criterion.weighted_n_right < min_weight_leaf:
            continue
        if criterion_kind == 1:
            cur_imp = _mse_centered_proxy(splitter.criterion)
        else:
            cur_imp = splitter.criterion.proxy_impurity_improvement()
        splitter.n_gain_evaluations += 1
        if cur_imp > -INFINITY:
            cur.feature = feat
            cur.pos = pos
            cur.threshold = threshold
            cur.n_missing = splitter.partitioner.n_missing
            cur.missing_go_to_left = 0
            cur.improvement = cur_imp
            if gain_buf != NULL and n_valid < gain_buf_len:
                gain_buf[n_valid] = cur_imp
            n_valid += 1
            if best_in_feature != NULL and cur_imp > best_imp:
                best_imp = cur_imp
                best_in_feature[0] = cur
                best_in_feature[0].improvement = cur_imp

    if thresholds_malloc != NULL:
        free(thresholds_malloc)
    return n_valid


cdef inline float64_t _draw_extratree_split_gain(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    uint32_t* random_state,
    SplitRecord* out,
) noexcept nogil:
    cdef float64_t sampled_threshold
    cdef float64_t cur_imp
    cdef intp_t pos

    _init_split_record(out, end)
    if not _draw_extratree_threshold_from_sorted_feature(
        splitter,
        start,
        end,
        random_state,
        &sampled_threshold,
        &pos,
    ):
        return -INFINITY

    cur_imp = _eval_split_at_position(
        splitter,
        feat,
        pos,
        start,
        end,
        min_weight_leaf,
        out,
    )
    if cur_imp > -INFINITY:
        out.threshold = sampled_threshold
    return cur_imp


cdef inline void _finalize_selected_split(
    BaseEarlyStopSplitter splitter,
    float64_t impurity,
    intp_t start,
    intp_t end,
    SplitRecord* best_split,
) noexcept nogil:
    cdef intp_t[::1] samples = splitter.samples

    splitter.partitioner.partition_samples_final(
        best_split.pos,
        best_split.threshold,
        best_split.feature,
        best_split.n_missing,
    )
    splitter.criterion.init_missing(best_split.n_missing)
    splitter.criterion.missing_go_to_left = best_split.missing_go_to_left
    splitter.criterion.reset()
    splitter.criterion.update(best_split.pos)
    splitter.criterion.children_impurity(
        &best_split.impurity_left,
        &best_split.impurity_right,
    )
    best_split.improvement = splitter.criterion.impurity_improvement(
        impurity,
        best_split.impurity_left,
        best_split.impurity_right,
    )
    shift_missing_values_to_left_if_required(best_split, samples, end)


cdef inline float64_t _best_gain_one_feature(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    SplitRecord* out,
) noexcept nogil:
    """For one feature (already filled and sorted), find best split over all positions. Return best gain or -INFINITY."""
    cdef intp_t p = start
    cdef float64_t best_imp = -INFINITY
    cdef float64_t cur_imp
    cdef SplitRecord cur
    splitter.criterion.reset()
    while p < end:
        while p + 1 < end and splitter.feature_values[p + 1] <= splitter.feature_values[p] + FEATURE_THRESHOLD:
            p += 1
        p += 1
        if p >= end:
            break
        cur_imp = _eval_split(splitter, feat, p, start, end, min_weight_leaf, &cur)
        if cur_imp > best_imp:
            best_imp = cur_imp
            out[0] = cur
    return best_imp


cdef inline bint _scan_feature_against_threshold(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t tau,
    bint strict,
    bint take_best,
    intp_t skip_pos,
    SplitRecord* out,
    int criterion_kind,
) noexcept nogil:
    """Scan one sorted feature once, optionally skipping one position.

    The scan reuses the exhaustive splitter's incremental criterion updates.
    It can either return the first split clearing tau or the best split among
    those clearing tau.
    """
    cdef intp_t p = start
    cdef float64_t cur_imp
    cdef float64_t best_imp = -INFINITY
    cdef SplitRecord cur
    cdef bint found = 0

    splitter.criterion.reset()
    _init_split_record(out, end)
    while p < end:
        while p + 1 < end and splitter.feature_values[p + 1] <= splitter.feature_values[p] + FEATURE_THRESHOLD:
            p += 1
        p += 1
        if p >= end:
            break
        if p == skip_pos:
            splitter.criterion.update(p)
            continue
        cur_imp = _eval_split(splitter, feat, p, start, end, min_weight_leaf, &cur)
        if cur_imp > -INFINITY and criterion_kind == 1:
            cur_imp = _mse_centered_proxy(splitter.criterion)
        if strict:
            if cur_imp <= tau:
                continue
        else:
            if cur_imp < tau:
                continue
        if not take_best:
            out[0] = cur
            out[0].improvement = cur_imp
            return 1
        if (not found) or cur_imp > best_imp:
            best_imp = cur_imp
            out[0] = cur
            out[0].improvement = cur_imp
            found = 1
    return found


cdef inline float64_t _secretary_gain_one_feature(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    uint32_t* random_state,
    SplitRecord* out,
) noexcept nogil:
    """For one feature (already filled and sorted), run secretary on thresholds: explore first 1/e positions, then first > max. Return selected gain."""
    cdef intp_t n = end - start
    cdef intp_t explore_positions = max(1, <intp_t>(n * ONE_OVER_E))
    cdef intp_t p = start
    cdef float64_t max_explore = -INFINITY
    cdef float64_t cur_imp
    cdef SplitRecord cur
    cdef intp_t idx = 0
    splitter.criterion.reset()
    _init_split_record(out, end)
    while p < end:
        while p + 1 < end and splitter.feature_values[p + 1] <= splitter.feature_values[p] + FEATURE_THRESHOLD:
            p += 1
        p += 1
        if p >= end:
            break
        cur_imp = _eval_split(splitter, feat, p, start, end, min_weight_leaf, &cur)
        if cur_imp <= -INFINITY:
            continue
        if idx < explore_positions:
            if cur_imp > max_explore:
                max_explore = cur_imp
                out[0] = cur
        else:
            if cur_imp > max_explore:
                out[0] = cur
                return cur_imp
        idx += 1
    return max_explore


cdef inline float64_t _secretary_gain_one_feature_random_explore(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    uint32_t* random_state,
    SplitRecord* out,
) noexcept nogil:
    """For one feature (already filled and sorted): sample ~1/e continuous thresholds, then
    scan exact thresholds once and return the best gain above the sampled maximum."""
    cdef intp_t p = start
    cdef intp_t n_candidates = 0
    cdef intp_t n_draws
    cdef float64_t max_explore = -INFINITY
    cdef SplitRecord explore_best
    cdef SplitRecord select_best

    _init_split_record(out, end)
    while p < end:
        while p + 1 < end and splitter.feature_values[p + 1] <= splitter.feature_values[p] + FEATURE_THRESHOLD:
            p += 1
        p += 1
        if p >= end:
            break
        n_candidates += 1

    if n_candidates <= 0:
        return -INFINITY

    n_draws = max(1, <intp_t>(ONE_OVER_E * n_candidates + 0.5))
    _draw_extratree_split_batch(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        random_state,
        n_draws,
        NULL,
        0,
        out,
        0,
    )
    if out.pos < end:
        max_explore = out.improvement
        explore_best = out[0]
    if _best_split_strictly_above(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        max_explore,
        &select_best,
    ):
        out[0] = select_best
        return select_best.improvement
    if explore_best.pos < end:
        out[0] = explore_best
    return max_explore


cdef int _cmp_float64(const void* a, const void* b) noexcept nogil:
    cdef float64_t x = (<const float64_t*>a)[0]
    cdef float64_t y = (<const float64_t*>b)[0]
    if x < y:
        return -1
    if x > y:
        return 1
    return 0


cdef inline intp_t _collect_gains_one_feature(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t* buf,
    intp_t buf_len,
    SplitRecord* best_in_feature,
) noexcept nogil:
    """Append gains from all valid splits of one feature into buf (up to buf_len). If best_in_feature is not NULL, set to best split in this feature. Return count appended."""
    cdef intp_t p = start
    cdef float64_t cur_imp
    cdef float64_t best_imp = -INFINITY
    cdef SplitRecord cur
    cdef intp_t n = 0
    splitter.criterion.reset()
    if best_in_feature != NULL:
        _init_split_record(best_in_feature, end)
    while p < end and n < buf_len:
        while p + 1 < end and splitter.feature_values[p + 1] <= splitter.feature_values[p] + FEATURE_THRESHOLD:
            p += 1
        p += 1
        if p >= end:
            break
        cur_imp = _eval_split(splitter, feat, p, start, end, min_weight_leaf, &cur)
        if cur_imp > -INFINITY:
            buf[n] = cur_imp
            n += 1
            if best_in_feature != NULL and cur_imp > best_imp:
                best_imp = cur_imp
                best_in_feature[0] = cur
                best_in_feature[0].improvement = cur_imp
    return n


cdef inline intp_t _collect_gains_one_feature_random_sample(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t* buf,
    intp_t buf_len,
    SplitRecord* best_in_feature,
    uint32_t* random_state,
) noexcept nogil:
    """Sample ~1/e ExtraTree-style continuous thresholds and append their gains."""
    cdef intp_t p = start
    cdef intp_t n_candidates = 0
    cdef intp_t n_draws
    cdef float64_t cur_imp
    cdef intp_t n = 0

    while p < end:
        while p + 1 < end and splitter.feature_values[p + 1] <= splitter.feature_values[p] + FEATURE_THRESHOLD:
            p += 1
        p += 1
        if p >= end:
            break
        n_candidates += 1

    if n_candidates <= 0 or buf_len <= 0:
        return 0

    n_draws = min(buf_len, max(1, <intp_t>(ONE_OVER_E * n_candidates + 0.5)))
    n = _draw_extratree_split_batch(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        random_state,
        n_draws,
        buf,
        buf_len,
        best_in_feature,
        0,
    )
    return n


cdef inline intp_t _collect_gains_one_feature_fraction(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t* buf,
    intp_t buf_len,
    uint32_t* random_state,
    float64_t sample_prob,
    int criterion_kind,
) noexcept nogil:
    """Collect gains from ExtraTree-style continuous threshold draws. The number of
    draws is a fraction of the distinct candidate thresholds, capped by buf_len."""
    cdef intp_t p = start
    cdef intp_t n_candidates = 0
    cdef intp_t n_draws
    cdef intp_t n = 0

    while p < end:
        while p + 1 < end and splitter.feature_values[p + 1] <= splitter.feature_values[p] + FEATURE_THRESHOLD:
            p += 1
        p += 1
        if p >= end:
            break
        n_candidates += 1

    if n_candidates <= 0 or buf_len <= 0 or sample_prob <= 0.0:
        return 0

    if sample_prob >= 1.0:
        n_draws = min(buf_len, n_candidates)
    else:
        n_draws = min(buf_len, max(1, <intp_t>(sample_prob * n_candidates + 0.5)))

    n = _draw_extratree_split_batch(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        random_state,
        n_draws,
        buf,
        buf_len,
        NULL,
        criterion_kind,
    )
    return n


cdef inline bint _first_split_above_threshold(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t tau,
    SplitRecord* out,
    int criterion_kind=0,
) noexcept nogil:
    """For one feature (filled and sorted), return the first split with gain >= tau."""
    return _scan_feature_against_threshold(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        tau,
        0,
        0,
        -1,
        out,
        criterion_kind,
    )


cdef inline bint _first_split_strictly_above(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t tau,
    SplitRecord* out,
) noexcept nogil:
    """For one feature (filled and sorted), return the first split with gain > tau."""
    return _scan_feature_against_threshold(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        tau,
        1,
        0,
        -1,
        out,
        0,
    )


cdef inline bint _best_split_above_threshold(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t tau,
    SplitRecord* out,
) noexcept nogil:
    """For one feature (filled and sorted), return the best split with gain >= tau."""
    return _scan_feature_against_threshold(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        tau,
        0,
        1,
        -1,
        out,
        0,
    )


cdef inline bint _best_split_strictly_above(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t tau,
    SplitRecord* out,
    int criterion_kind=0,
) noexcept nogil:
    """For one feature (filled and sorted), return the best split with gain > tau."""
    return _scan_feature_against_threshold(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        tau,
        1,
        1,
        -1,
        out,
        criterion_kind,
    )


cdef inline bint _best_split_above_threshold_skip_pos(
    BaseEarlyStopSplitter splitter,
    intp_t feat,
    intp_t start,
    intp_t end,
    float64_t min_weight_leaf,
    float64_t tau,
    intp_t skip_pos,
    SplitRecord* out,
) noexcept nogil:
    """For one feature (filled and sorted), return the best split with gain >= tau,
    excluding one exact threshold position from the scan."""
    return _scan_feature_against_threshold(
        splitter,
        feat,
        start,
        end,
        min_weight_leaf,
        tau,
        0,
        1,
        skip_pos,
        out,
        0,
    )


# ---------------------------------------------------------------------------
# (S) SecretarySplitter: one random pass over features. Exploration features use
# ExtraTree-style continuous threshold draws; selection features use the best
# exact split within the feature and stop on the first feature beating the
# exploration maximum.
# ---------------------------------------------------------------------------

cdef class SecretarySplitter(BaseEarlyStopSplitter):
    """(S) Secretary in a single pass over features.

    An exploration prefix of the random feature order is used to set the
    stopping level, where each exploration feature is scored by the best gain
    among ExtraTree-style continuous threshold draws. The remaining visited
    features are scored by their exact best split, and the first feature
    beating the exploration maximum is accepted. If none does, the best
    exploration split is returned.
    """

    def __reduce__(self):
        return (type(self), (
            self.criterion,
            self.max_features,
            self.min_samples_leaf,
            self.min_weight_leaf,
            self.random_state,
            np.asarray(self.monotonic_cst) if self.monotonic_cst is not None else None,
        ), self.__getstate__())

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_features = self.n_features
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] constant_features = self.constant_features
        cdef float32_t[::1] feature_values = self.feature_values
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* random_state = &self.rand_r_state
        cdef float64_t impurity = parent.impurity
        cdef intp_t n_known_constants = parent.n_constant_features
        cdef intp_t f_i, f_j, p
        cdef intp_t n_visited, n_found_constants, n_drawn_constants, n_total_constants
        cdef intp_t n_candidates, n_draws, n_visit_budget, n_explore, n_nonconstant_seen
        cdef SplitRecord best_split, current_split
        cdef float64_t cur_imp
        cdef float64_t max_explore = -INFINITY
        cdef float64_t eff_frac
        cdef bint in_exploration

        self.partitioner.init_node_split(start, end)
        _init_split_record(&best_split, end)
        n_total_constants = n_known_constants
        f_i = n_features
        eff_frac = _effective_explore_frac(self.explore_frac, self.use_sqrt_n, end - start)
        n_visited = 0
        n_found_constants = 0
        n_drawn_constants = 0

        cdef intp_t n_avail = n_features - n_known_constants
        if n_avail <= 0:
            split.pos = end
            parent.n_constant_features = n_known_constants
            _sync_constant_features(features, constant_features, n_known_constants, 0)
            return 0
        n_visit_budget = self.max_features if self.max_features < n_avail else n_avail
        n_explore = _exploration_budget_from_frac(eff_frac, n_visit_budget)
        n_nonconstant_seen = 0
        if n_features > MAX_FEATURES_BACKUP:
            return _node_split_best_fallback(self, parent, split)

        # Single pass: the exploration prefix of the random feature order uses
        # ExtraTree-style continuous draws to update the stopping threshold; the
        # remaining features use one exact best-splitter-style scan and stop on
        # the first improvement.
        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            current_split.feature = features[f_j]
            _fill_and_sort_feature(self, current_split.feature, start, end)
            if _sorted_feature_is_constant(self, start, end):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            in_exploration = n_nonconstant_seen < n_explore
            n_nonconstant_seen += 1
            p = start
            n_candidates = 0
            while p < end:
                while p + 1 < end and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD:
                    p += 1
                p += 1
                if p >= end:
                    break
                n_candidates += 1
            if n_candidates <= 0:
                continue
            if in_exploration:
                n_draws = max(1, <intp_t>(eff_frac * n_candidates + 0.5))
                _draw_extratree_split_batch(
                    self,
                    current_split.feature,
                    start,
                    end,
                    min_weight_leaf,
                    random_state,
                    n_draws,
                    NULL,
                    0,
                    &current_split,
                    0,
                )
                cur_imp = current_split.improvement if current_split.pos < end else -INFINITY
                if cur_imp > max_explore:
                    max_explore = cur_imp
                    best_split = current_split
            else:
                if _best_split_strictly_above(
                    self,
                    current_split.feature,
                    start,
                    end,
                    min_weight_leaf,
                    max_explore,
                    &current_split,
                ):
                    best_split = current_split
                    break

        if best_split.pos >= end:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        _finalize_selected_split(self, impurity, start, end, &best_split)
        split[0] = best_split
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0


# ---------------------------------------------------------------------------
# (S+par) SecretaryParamSplitter: random 1/e covariates; per feature sample
# p_thr_par continuous thresholds, fit gain distribution, take first exact
# threshold above q_thr_par quantile as within-feature best; selection = first
# (feature, threshold) with gain > max_explore
# ---------------------------------------------------------------------------

cdef class SecretaryParamSplitter(BaseEarlyStopSplitter):
    """(S+par) Secretary on covariates; per feature sample p_thr_par continuous
    thresholds, fit distribution (MLE/moments), first exact threshold above the
    q_thr_par quantile = within-feature best; selection = first better than all
    exploration bests."""

    def __cinit__(self, Criterion criterion, intp_t max_features, intp_t min_samples_leaf,
                  float64_t min_weight_leaf, object random_state, const int8_t[:] monotonic_cst,
                  float64_t alpha=0.5, object criterion_kind=None,
                  float64_t p_thr_par=0.1, float64_t q_thr_par=0.9, *argv, **kwargs):
        self.alpha = alpha
        self.p_thr_par = p_thr_par
        self.q_thr_par = q_thr_par
        n_par = int(kwargs.get("n_gain_samples_par", 256))
        self.n_gain_samples_par = min(max(n_par, 1), 256)
        if criterion_kind == 'regression' and criterion.n_outputs == 1:
            self.criterion_kind = 1
        elif criterion_kind == 'classification':
            self.criterion_kind = 2
        else:
            self.criterion_kind = 0

    def __reduce__(self):
        return (type(self), (
            self.criterion,
            self.max_features,
            self.min_samples_leaf,
            self.min_weight_leaf,
            self.random_state,
            np.asarray(self.monotonic_cst) if self.monotonic_cst is not None else None,
            self.alpha,
            None,  # criterion_kind
            self.p_thr_par,
            self.q_thr_par,
            self.n_gain_samples_par,
        ), self.__getstate__())

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_features = self.n_features
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] constant_features = self.constant_features
        cdef float32_t[::1] feature_values = self.feature_values
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* random_state = &self.rand_r_state
        cdef float64_t impurity = parent.impurity
        cdef intp_t n_known_constants = parent.n_constant_features
        cdef intp_t f_i, f_j, p
        cdef intp_t n_visited = 0
        cdef intp_t n_found_constants = 0
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_total_constants = n_known_constants
        cdef SplitRecord best_split, feat_best
        cdef float64_t max_explore = -INFINITY
        cdef bint in_exploration
        cdef float64_t feat_buf[256]
        cdef intp_t n_feat_gains
        cdef float64_t tau_j
        cdef float64_t reward_gain
        cdef intp_t empirical_idx
        cdef float64_t eff_frac
        cdef intp_t n_visit_budget
        cdef intp_t n_explore
        cdef intp_t n_nonconstant_seen = 0

        self.partitioner.init_node_split(start, end)
        _init_split_record(&best_split, end)
        f_i = n_features
        eff_frac = _effective_explore_frac(self.explore_frac, self.use_sqrt_n, end - start)
        cdef intp_t n_avail = n_features - n_known_constants
        if n_avail <= 0:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        n_visit_budget = self.max_features if self.max_features < n_avail else n_avail
        n_explore = _exploration_budget_from_frac(eff_frac, n_visit_budget)
        if n_features > MAX_FEATURES_BACKUP:
            return _node_split_best_fallback(self, parent, split)

        # Single pass over the random feature order: the exploration prefix uses
        # sampled gains to fit a working threshold, and the remaining features
        # are tested immediately against the global exploration maximum.
        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            feat_best.feature = features[f_j]
            _fill_and_sort_feature(self, feat_best.feature, start, end)
            if _sorted_feature_is_constant(self, start, end):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            in_exploration = n_nonconstant_seen < n_explore
            n_nonconstant_seen += 1
            if in_exploration:
                n_feat_gains = _collect_gains_one_feature_fraction(
                    self, feat_best.feature, start, end, min_weight_leaf,
                    &feat_buf[0], self.n_gain_samples_par, random_state,
                    self.p_thr_par, self.criterion_kind)
                if n_feat_gains > 0:
                    self.n_parametric_gain_samples += n_feat_gains
                    self.n_parametric_quantile_fits += 1
                    empirical_idx = min(n_feat_gains - 1, <intp_t>(self.q_thr_par * n_feat_gains))
                    if self.criterion_kind == 0:
                        qsort(&feat_buf[0], n_feat_gains, sizeof(float64_t), _cmp_float64)
                        tau_j = feat_buf[empirical_idx]
                    else:
                        tau_j = _quantile_parametric(
                            &feat_buf[0], n_feat_gains, self.q_thr_par, self.criterion_kind,
                            empirical_idx)
                    if _first_split_above_threshold(
                            self, feat_best.feature, start, end, min_weight_leaf,
                            tau_j, &feat_best, self.criterion_kind):
                        if self.criterion_kind == 1:
                            reward_gain = _mse_centered_proxy(self.criterion)
                        else:
                            reward_gain = self.criterion.proxy_impurity_improvement()
                        if reward_gain > max_explore:
                            max_explore = reward_gain
                            best_split = feat_best
            else:
                if _best_split_strictly_above(
                        self, feat_best.feature, start, end, min_weight_leaf,
                        max_explore, &feat_best, self.criterion_kind):
                    best_split = feat_best
                    break

        if best_split.pos >= end:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        _finalize_selected_split(self, impurity, start, end, &best_split)
        split[0] = best_split
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0


# ---------------------------------------------------------------------------
# (S+all) CovariateSecretaryAllSplitter: random 1/e covariates, all thresholds; then first better or best in exploration
# ---------------------------------------------------------------------------

cdef class CovariateSecretaryAllSplitter(BaseEarlyStopSplitter):
    """(S+all) Secretary on covariates with an exploration prefix of the random
    feature order; reward = max gain over all thresholds."""

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_features = self.n_features
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] constant_features = self.constant_features
        cdef float32_t[::1] feature_values = self.feature_values
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* random_state = &self.rand_r_state
        cdef float64_t impurity = parent.impurity
        cdef intp_t n_known_constants = parent.n_constant_features
        cdef intp_t f_i, f_j, p, i
        cdef intp_t n_visited = 0
        cdef intp_t n_found_constants = 0
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_total_constants = n_known_constants
        cdef SplitRecord best_split, feat_best
        cdef float64_t reward
        cdef float64_t max_explore = -INFINITY
        cdef float64_t eff_frac
        cdef bint in_exploration
        cdef intp_t n_visit_budget
        cdef intp_t n_explore
        cdef intp_t n_nonconstant_seen = 0

        self.partitioner.init_node_split(start, end)
        _init_split_record(&best_split, end)
        f_i = n_features
        eff_frac = _effective_explore_frac(self.explore_frac, self.use_sqrt_n, end - start)
        cdef intp_t n_avail = n_features - n_known_constants
        if n_avail <= 0:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        n_visit_budget = self.max_features if self.max_features < n_avail else n_avail
        n_explore = _exploration_budget_from_frac(eff_frac, n_visit_budget)
        if n_features > MAX_FEATURES_BACKUP:
            return _node_split_best_fallback(self, parent, split)

        # Single pass: evaluate features in random order, explore on the prefix,
        # then stop on the first later feature whose exact within-feature best
        # exceeds the exploration maximum.
        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            feat_best.feature = features[f_j]
            _fill_and_sort_feature(self, feat_best.feature, start, end)
            if _sorted_feature_is_constant(self, start, end):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            in_exploration = n_nonconstant_seen < n_explore
            n_nonconstant_seen += 1
            if in_exploration:
                reward = _best_gain_one_feature(self, feat_best.feature, start, end, min_weight_leaf, &feat_best)
                if reward > -INFINITY and reward > max_explore:
                    max_explore = reward
                    best_split = feat_best
            else:
                if _best_split_strictly_above(
                    self, feat_best.feature, start, end, min_weight_leaf, max_explore, &feat_best
                ):
                    best_split = feat_best
                    break

        if best_split.pos >= end:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        _finalize_selected_split(self, impurity, start, end, &best_split)
        split[0] = best_split
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0


# ---------------------------------------------------------------------------
# (S^2) DoubleSecretarySplitter: random 1/e covariates; per covariate sample
# ~1/e continuous thresholds to set tau, then best exact threshold above tau;
# then first better feature or best in exploration
# ---------------------------------------------------------------------------

cdef class DoubleSecretarySplitter(BaseEarlyStopSplitter):
    """(S^2) Secretary on covariates; reward = best among ~1/e continuous
    threshold draws followed by the best exact threshold above that level.
    Use an exploration prefix of the random feature order; then first better or
    best in exploration."""

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_features = self.n_features
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] constant_features = self.constant_features
        cdef float32_t[::1] feature_values = self.feature_values
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* random_state = &self.rand_r_state
        cdef float64_t impurity = parent.impurity
        cdef intp_t n_known_constants = parent.n_constant_features
        cdef intp_t f_i, f_j, p, i
        cdef intp_t n_visited = 0
        cdef intp_t n_found_constants = 0
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_total_constants = n_known_constants
        cdef SplitRecord best_split, feat_best
        cdef float64_t reward
        cdef float64_t max_explore = -INFINITY
        cdef float64_t eff_frac
        cdef bint in_exploration
        cdef intp_t n_visit_budget
        cdef intp_t n_explore
        cdef intp_t n_nonconstant_seen = 0

        self.partitioner.init_node_split(start, end)
        _init_split_record(&best_split, end)
        f_i = n_features
        eff_frac = _effective_explore_frac(self.explore_frac, self.use_sqrt_n, end - start)
        cdef intp_t n_avail = n_features - n_known_constants
        if n_avail <= 0:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        n_visit_budget = self.max_features if self.max_features < n_avail else n_avail
        n_explore = _exploration_budget_from_frac(eff_frac, n_visit_budget)
        if n_features > MAX_FEATURES_BACKUP:
            return _node_split_best_fallback(self, parent, split)

        # Single pass: evaluate features in random order, explore on the prefix
        # with the inner threshold rule, then stop on the first later feature
        # whose realized gain exceeds the exploration maximum.
        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            feat_best.feature = features[f_j]
            _fill_and_sort_feature(self, feat_best.feature, start, end)
            if _sorted_feature_is_constant(self, start, end):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            in_exploration = n_nonconstant_seen < n_explore
            n_nonconstant_seen += 1
            if in_exploration:
                reward = _secretary_gain_one_feature_random_explore(
                    self, feat_best.feature, start, end, min_weight_leaf, random_state, &feat_best)
                if reward > -INFINITY and reward > max_explore:
                    max_explore = reward
                    best_split = feat_best
            else:
                if _best_split_strictly_above(
                    self, feat_best.feature, start, end, min_weight_leaf, max_explore, &feat_best
                ):
                    best_split = feat_best
                    break

        if best_split.pos >= end:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        _finalize_selected_split(self, impurity, start, end, &best_split)
        split[0] = best_split
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0


# ---------------------------------------------------------------------------
# (PI) ProphetSamplesSplitter
# ---------------------------------------------------------------------------

cdef class ProphetSamplesSplitter(BaseEarlyStopSplitter):
    """(PI) Prophet from samples: τ = median of random continuous threshold
    draws; take the best exact split with gain >= τ on the first feature that clears τ."""

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_features = self.n_features
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] constant_features = self.constant_features
        cdef float32_t[::1] feature_values = self.feature_values
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* random_state = &self.rand_r_state
        cdef float64_t impurity = parent.impurity
        cdef intp_t n_known_constants = parent.n_constant_features
        cdef intp_t f_i, f_j, p, i
        cdef intp_t n_visited = 0
        cdef intp_t n_found_constants = 0
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_total_constants = n_known_constants
        cdef intp_t partition_end
        cdef intp_t f_i_end
        cdef SplitRecord best_split, cur_split, best_sample
        cdef float64_t tau = -INFINITY
        cdef float64_t best_sample_gain = -INFINITY
        cdef intp_t n_gains = 0
        cdef float64_t gain_buf[256]  # N_GAIN_SAMPLES

        self.partitioner.init_node_split(start, end)
        _init_split_record(&best_split, end)
        f_i = n_features
        cdef intp_t n_avail = n_features - n_known_constants
        if n_avail <= 0:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0

        # Pass 1: random feature order; for each split with prob 1/e add gain to buffer
        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            cur_split.feature = features[f_j]
            _fill_and_sort_feature(self, cur_split.feature, start, end)
            if _sorted_feature_is_constant(self, start, end):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            _init_split_record(&best_sample, end)
            n_gains += _collect_gains_one_feature_random_sample(
                self, cur_split.feature, start, end, min_weight_leaf,
                &gain_buf[n_gains], 256 - n_gains, &best_sample, random_state)
            if best_sample.pos < end and best_sample.improvement > best_sample_gain:
                best_sample_gain = best_sample.improvement
                best_split = best_sample
        f_i_end = f_i
        if n_gains > 0:
            qsort(&gain_buf[0], n_gains, sizeof(float64_t), _cmp_float64)
            tau = gain_buf[n_gains // 2]
        else:
            tau = -INFINITY

        # Pass 2: same feature order; best exact split with gain >= tau on the first
        # feature that clears the threshold.
        i = n_features - 1
        while i >= f_i_end:
            cur_split.feature = features[i]
            _fill_and_sort_feature(self, cur_split.feature, start, end)
            if not _sorted_feature_is_constant(self, start, end):
                if _best_split_above_threshold(
                        self, cur_split.feature, start, end, min_weight_leaf, tau, &cur_split):
                    best_split = cur_split
                    break
            i -= 1

        if best_split.pos >= end and best_sample_gain > -INFINITY:
            best_split = best_sample
        if best_split.pos >= end:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        _finalize_selected_split(self, impurity, start, end, &best_split)
        split[0] = best_split
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0


# ---------------------------------------------------------------------------
# (PI-1) ProphetOneSampleSplitter: τ = one random split gain; exact selection uses
# one best-splitter-style scan per feature
# ---------------------------------------------------------------------------

cdef class ProphetOneSampleSplitter(BaseEarlyStopSplitter):
    """(PI-1) Classic secretary: exploration = one random split per feature (first n_features from
    different features); τ = max of those. Selection = best exact split above τ on the
    first feature that clears τ, skipping the exploration position; if none, use the
    split that achieved τ."""

    def __cinit__(self, *args, **kwargs):
        self.explore_gains = NULL
        self.explore_splits = NULL
    def __dealloc__(self):
        if self.explore_gains != NULL:
            free(self.explore_gains)
            self.explore_gains = NULL
        if self.explore_splits != NULL:
            free(self.explore_splits)
            self.explore_splits = NULL

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        BaseEarlyStopSplitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        if self.explore_gains != NULL:
            free(self.explore_gains)
            self.explore_gains = NULL
        if self.explore_splits != NULL:
            free(self.explore_splits)
            self.explore_splits = NULL
        if self.n_features > 0:
            self.explore_gains = <float64_t*>malloc(self.n_features * sizeof(float64_t))
            self.explore_splits = <SplitRecord*>malloc(self.n_features * sizeof(SplitRecord))
            if self.explore_gains == NULL or self.explore_splits == NULL:
                if self.explore_gains != NULL:
                    free(self.explore_gains)
                    self.explore_gains = NULL
                if self.explore_splits != NULL:
                    free(self.explore_splits)
                    self.explore_splits = NULL
                raise MemoryError("ProphetOneSampleSplitter: malloc explore_gains/explore_splits")
        return 0

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_features = self.n_features
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] constant_features = self.constant_features
        cdef float32_t[::1] feature_values = self.feature_values
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* explore_random_state = &self.explore_rand_r_state
        cdef float64_t impurity = parent.impurity
        cdef intp_t n_known_constants = parent.n_constant_features
        cdef intp_t f_i, f_j, idx
        cdef intp_t n_visited = 0
        cdef intp_t n_found_constants = 0
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_total_constants = n_known_constants
        cdef intp_t end_non_missing
        cdef SplitRecord cur_split, feat_split, tau_split, best_split, first_above
        cdef float32_t min_feature_value, max_feature_value
        cdef float64_t feat_imp
        cdef float64_t tau = -INFINITY
        cdef bint found_above = 0

        self.partitioner.init_node_split(start, end)
        for idx in range(n_features):
            self.explore_gains[idx] = -INFINITY
        _init_split_record(&tau_split, end)
        _init_split_record(&best_split, end)
        _init_split_record(&first_above, end)
        f_i = n_features
        # Pass 1 (exploration): one random split per feature; τ = max over those n_features gains.
        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, explore_random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            cur_split.feature = features[f_j]
            self.partitioner.find_min_max(
                cur_split.feature,
                &min_feature_value,
                &max_feature_value,
            )
            end_non_missing = end - self.partitioner.n_missing
            if (
                end_non_missing == start or
                max_feature_value <= min_feature_value + FEATURE_THRESHOLD
            ):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            # Exploration mirrors ExtraTree: one continuous threshold draw per feature.
            feat_imp = _draw_extratree_split_gain_from_minmax(
                self,
                cur_split.feature,
                start,
                end,
                min_weight_leaf,
                explore_random_state,
                min_feature_value,
                max_feature_value,
                &feat_split,
            )
            if feat_imp > tau:
                tau = feat_imp
                tau_split = feat_split
            self.explore_gains[cur_split.feature] = feat_imp
            self.explore_splits[cur_split.feature] = feat_split
        if tau <= -INFINITY:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        # Selection: replay the same randomized feature order used during
        # exploration, and on each visited feature do one exact best-splitter
        # scan while skipping the exploration position.
        for idx in range(n_features - 1, f_i - 1, -1):
            cur_split.feature = features[idx]
            if self.explore_gains[cur_split.feature] <= -INFINITY:
                continue
            _fill_and_sort_feature(self, cur_split.feature, start, end)
            if _sorted_feature_is_constant(self, start, end):
                continue
            if _best_split_above_threshold_skip_pos(
                self,
                cur_split.feature,
                start,
                end,
                min_weight_leaf,
                tau,
                self.explore_splits[cur_split.feature].pos,
                &cur_split,
            ):
                first_above = cur_split
                found_above = 1
                break
        if not found_above:
            best_split = tau_split
        else:
            best_split = first_above
        if best_split.pos >= end:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        _finalize_selected_split(self, impurity, start, end, &best_split)
        split[0] = best_split
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0


# ---------------------------------------------------------------------------
# BlockRankSplitter: block-best then secretary on block sequence
# ---------------------------------------------------------------------------

cdef class BlockRankSplitter(BaseEarlyStopSplitter):
    """Block-best stream with secretary-style acceptance.

    For each visited feature, admissible thresholds are grouped into blocks of
    size ~sqrt(n). The best threshold in each block is exposed as one stream
    item. The current implementation evaluates that block-best stream with a
    secretary-style rule while keeping the within-feature scan incremental.
    """

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_features = self.n_features
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] constant_features = self.constant_features
        cdef float32_t[::1] feature_values = self.feature_values
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* random_state = &self.rand_r_state
        cdef float64_t impurity = parent.impurity
        cdef intp_t n_known_constants = parent.n_constant_features
        cdef intp_t f_i, f_j, p, block_size, step
        cdef intp_t n_visited = 0
        cdef intp_t n_found_constants = 0
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_total_constants = n_known_constants
        cdef SplitRecord best_split, block_best, cur_split
        cdef float64_t block_best_imp
        cdef float64_t cur_imp
        cdef float64_t max_explore = -INFINITY
        cdef float64_t eff_frac
        cdef bint do_explore
        cdef bint found = 0
        cdef intp_t partition_end

        self.partitioner.init_node_split(start, end)
        _init_split_record(&best_split, end)
        f_i = n_features
        eff_frac = _effective_explore_frac(self.explore_frac, self.use_sqrt_n, end - start)
        cdef intp_t n_avail = n_features - n_known_constants
        if n_avail <= 0:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        if n_features > MAX_FEATURES_BACKUP:
            return _node_split_best_fallback(self, parent, split)

        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            block_best.feature = features[f_j]
            _fill_and_sort_feature(self, block_best.feature, start, end)
            if _sorted_feature_is_constant(self, start, end):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            block_size = max(1, <intp_t>sqrt(<float64_t>(end - start)))
            _init_split_record(&block_best, end)
            block_best.feature = features[f_i]
            cur_split.feature = block_best.feature
            self.criterion.init_missing(self.partitioner.n_missing)
            self.criterion.missing_go_to_left = 0
            self.criterion.reset()
            block_best_imp = -INFINITY
            step = 0
            p = start
            while p < end:
                while p + 1 < end and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD:
                    p += 1
                p += 1
                if p >= end:
                    break
                cur_imp = _eval_split(self, cur_split.feature, p, start, end, min_weight_leaf, &cur_split)
                if cur_imp > block_best_imp and cur_imp > -INFINITY:
                    block_best_imp = cur_imp
                    block_best = cur_split
                step += 1
                if step < block_size:
                    continue
                if block_best_imp > -INFINITY:
                    do_explore = rand_uniform(0.0, 1.0, random_state) < eff_frac
                    if do_explore:
                        if block_best_imp > max_explore:
                            max_explore = block_best_imp
                            best_split = block_best
                    else:
                        if block_best_imp > max_explore:
                            best_split = block_best
                            found = 1
                            break
                step = 0
                block_best_imp = -INFINITY
                _init_split_record(&block_best, end)
                block_best.feature = cur_split.feature
            if not found and step > 0 and block_best_imp > -INFINITY:
                do_explore = rand_uniform(0.0, 1.0, random_state) < eff_frac
                if do_explore:
                    if block_best_imp > max_explore:
                        max_explore = block_best_imp
                        best_split = block_best
                else:
                    if block_best_imp > max_explore:
                        best_split = block_best
                        found = 1
            if found:
                break

        if best_split.pos >= end:
            split.pos = end
            parent.n_constant_features = n_total_constants
            _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
            return 0
        _finalize_selected_split(self, impurity, start, end, &best_split)
        split[0] = best_split
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0


# ---------------------------------------------------------------------------
# (PI+par) ProphetParamSplitter
# ---------------------------------------------------------------------------

cdef class ProphetParamSplitter(BaseEarlyStopSplitter):
    """(PI+par) Prophet on known parametric gain distribution per covariate."""

    def __cinit__(self, Criterion criterion, intp_t max_features, intp_t min_samples_leaf,
                  float64_t min_weight_leaf, object random_state, const int8_t[:] monotonic_cst,
                  float64_t alpha=0.5, object criterion_kind=None, *argv):
        self.alpha = alpha

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        return _node_split_best_fallback(self, parent, split)


# ---------------------------------------------------------------------------
# (MAB+all), (MAB+S), (MAB+par)
# ---------------------------------------------------------------------------

cdef class MABAllSplitter(BaseEarlyStopSplitter):
    """(MAB+all) MAB for best covariate, then all thresholds."""

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        return _node_split_best_fallback(self, parent, split)


cdef class MABSecretarySplitter(BaseEarlyStopSplitter):
    """(MAB+S) MAB for best covariate, then secretary on thresholds."""

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        return _node_split_best_fallback(self, parent, split)


cdef class MABParamSplitter(BaseEarlyStopSplitter):
    """(MAB+par) MAB for best covariate, then parametric threshold."""

    def __cinit__(self, Criterion criterion, intp_t max_features, intp_t min_samples_leaf,
                  float64_t min_weight_leaf, object random_state, const int8_t[:] monotonic_cst,
                  float64_t alpha=0.5, object criterion_kind=None, *argv):
        self.alpha = alpha

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        self.n_split_calls += 1
        return _node_split_best_fallback(self, parent, split)


# ---------------------------------------------------------------------------
# Fallback: full best split (same logic as BestSplitter for dense axis-aligned)
# ---------------------------------------------------------------------------

cdef int _node_split_best_fallback(
    BaseEarlyStopSplitter splitter,
    ParentInfo* parent,
    SplitRecord* split,
) except -1 nogil:
    """Find best split over all features and thresholds (dense, axis-aligned). Used as fallback for unimplemented strategies."""
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t n_features = splitter.n_features
    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] samples = splitter.samples
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef float32_t[::1] feature_values = splitter.feature_values
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state
    cdef float64_t impurity = parent.impurity
    cdef intp_t n_known_constants = parent.n_constant_features
    cdef intp_t f_i = n_features
    cdef intp_t f_j, p, i
    cdef intp_t n_visited = 0
    cdef intp_t n_found_constants = 0
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_total_constants = n_known_constants
    cdef intp_t partition_end
    cdef SplitRecord best_split, current_split
    cdef float64_t best_imp = -INFINITY
    cdef float64_t cur_imp

    splitter.partitioner.init_node_split(start, end)
    _init_split_record(&best_split, end)

    while f_i > n_total_constants and (
        n_visited < splitter.max_features or n_visited <= n_found_constants + n_drawn_constants
    ):
        n_visited += 1
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)
        if f_j < n_known_constants:
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue
        f_j += n_found_constants
        current_split.feature = features[f_j]
        _fill_and_sort_feature(splitter, current_split.feature, start, end)
        if _sorted_feature_is_constant(splitter, start, end):
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
            n_found_constants += 1
            n_total_constants += 1
            continue
        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        splitter.criterion.reset()
        p = start
        while p < end:
            while p + 1 < end and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD:
                p += 1
            p += 1
            if p >= end:
                break
            cur_imp = _eval_split(splitter, current_split.feature, p, start, end, min_weight_leaf, &current_split)
            if cur_imp > best_imp:
                best_imp = cur_imp
                best_split = current_split

    if best_split.pos >= end:
        split.pos = end
        parent.n_constant_features = n_total_constants
        _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
        return 0

    _finalize_selected_split(splitter, impurity, start, end, &best_split)
    split[0] = best_split
    parent.n_constant_features = n_total_constants
    _sync_constant_features(features, constant_features, n_known_constants, n_found_constants)
    return 0
