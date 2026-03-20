# Early stopping splitters for CART: secretary, prophet inequality, and MAB strategies.
# Dense data only (sparse mirrors can be added later).

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._tree cimport ParentInfo
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int8_t, intp_t, uint8_t, uint32_t


cdef class BaseEarlyStopSplitter(Splitter):
    """Base class for early-stopping axis-aligned splitters.

    Subclasses implement node_split with secretary, prophet, or MAB strategies
    to avoid scanning all (feature, threshold) pairs while preserving a good
    probability of selecting a near-best split.
    """
    cdef const float32_t[:, ::1] X  # Dense feature matrix (set in init)
    cdef float64_t explore_frac      # Secretary exploration prob: (0,1]=use it; <0=1/e
    cdef bint use_sqrt_n            # If True, explore_frac = 1/sqrt(n_node)

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1


# --- Secretary family ---

cdef class SecretarySplitter(BaseEarlyStopSplitter):
    """(S) Secretary on splits: explore 1/e random (covariate, threshold) pairs,
    then select the next split with gain > max in exploration set, or best in set."""
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class SecretaryParamSplitter(BaseEarlyStopSplitter):
    """(S+par) Secretary on covariates; per-covariate best threshold via parametric
    (Gaussian / Gamma approximation or empirical) quantile alpha."""
    cdef float64_t alpha  # Quantile for threshold acceptance (e.g. 0.5)
    cdef int criterion_kind  # 0=empirical, 1=regression (Gaussian), 2=classification (Gamma approx)
    cdef float64_t p_thr_par  # Fraction of thresholds to sample for gain distribution (e.g. 0.1)
    cdef float64_t q_thr_par  # Quantile of fitted distribution for within-feature best (e.g. 0.9)
    cdef intp_t n_gain_samples_par  # Max number of gains to collect for param fit (e.g. 256)
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class CovariateSecretaryAllSplitter(BaseEarlyStopSplitter):
    """(S+all) Secretary on covariates; reward per covariate = max gain over all
    thresholds. Explore 1/e covariates (all thresholds), then take first better."""
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class DoubleSecretarySplitter(BaseEarlyStopSplitter):
    """(S^2) Double secretary: secretary on covariates, reward = secretary gain
    on thresholds (1/e thresholds per covariate in exploration)."""
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


# --- Prophet inequality family ---

cdef class ProphetSamplesSplitter(BaseEarlyStopSplitter):
    """(PI) Single-choice prophet inequality from samples (Rubinstein–Wang–Weinberg)."""
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class ProphetOneSampleSplitter(BaseEarlyStopSplitter):
    """(PI-1) Secretary with exploration = one random split per feature; τ = max; selection = first new split with gain ≥ τ."""
    cdef float64_t* explore_gains
    cdef SplitRecord* explore_splits
    cdef intp_t* sel_f
    cdef intp_t* sel_pos
    cdef intp_t sel_cap
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class BlockRankSplitter(BaseEarlyStopSplitter):
    """Block-rank: partition (feat, threshold) stream into blocks, best per block; secretary on block-best sequence."""
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class ProphetParamSplitter(BaseEarlyStopSplitter):
    """(PI+par) Prophet on known (parametric scaled chi-2) gain distribution per covariate."""
    cdef float64_t alpha
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


# --- MAB family ---

cdef class MABAllSplitter(BaseEarlyStopSplitter):
    """(MAB+all) MAB for best covariate, then test all thresholds for that covariate."""
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class MABSecretarySplitter(BaseEarlyStopSplitter):
    """(MAB+S) MAB for best covariate, then secretary on thresholds."""
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class MABParamSplitter(BaseEarlyStopSplitter):
    """(MAB+par) MAB for best covariate, then parametric (quantile alpha) threshold."""
    cdef float64_t alpha
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil
