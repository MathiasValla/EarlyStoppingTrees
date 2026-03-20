#!/usr/bin/env python
"""
Step 1: Build and run — test EarlyStopDecisionTreeRegressor and EarlyStopDecisionTreeClassifier
on Iris with splitter="best" and splitter="secretary" to confirm no runtime/API errors.

Uses numpy only for data and split to avoid triggering other editable installs (e.g. sklearn).
For full Iris, run with: python -c "from sklearn.datasets import load_iris; ..." or use the notebook.
"""
import numpy as np

from treeple.tree import EarlyStopDecisionTreeClassifier, EarlyStopDecisionTreeRegressor

RANDOM_STATE = 42


def _simple_train_test_split(X, y, test_size=0.3, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    train_idx, test_idx = idx[n_test:], idx[:n_test]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    # Minimal Iris-like data (150 x 4) so we don't need sklearn.datasets
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = np.asarray(iris.data, dtype=np.float32)
        y_class = iris.target
    except Exception:
        # Fallback: small synthetic data
        rng = np.random.default_rng(RANDOM_STATE)
        X = rng.standard_normal((150, 4)).astype(np.float32)
        y_class = (X[:, 0] + X[:, 1] > 0).astype(np.intp)
    # Regression: predict column 2
    y_reg = X[:, 2].copy()
    X_reg = np.delete(X, 2, axis=1)

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = _simple_train_test_split(
        X, y_class, test_size=0.3, random_state=RANDOM_STATE
    )
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = _simple_train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=RANDOM_STATE
    )

    print("=" * 60)
    print("EarlyStopDecisionTreeRegressor (Iris — predict petal length)")
    print("=" * 60)
    for splitter in ("best", "secretary"):
        est = EarlyStopDecisionTreeRegressor(splitter=splitter, random_state=RANDOM_STATE)
        est.fit(X_reg_train, y_reg_train)
        pred = est.predict(X_reg_test)
        mse = float(np.mean((pred - y_reg_test) ** 2))
        r2 = 1.0 - np.sum((y_reg_test - pred) ** 2) / np.sum((y_reg_test - np.mean(y_reg_test)) ** 2)
        print(f"  splitter={splitter:12s}  MSE={mse:.6f}  R²={r2:.4f}  n_leaves={est.get_n_leaves()}  OK")

    print()
    print("=" * 60)
    print("EarlyStopDecisionTreeClassifier (Iris — species)")
    print("=" * 60)
    for splitter in ("best", "secretary"):
        est = EarlyStopDecisionTreeClassifier(splitter=splitter, random_state=RANDOM_STATE)
        est.fit(X_clf_train, y_clf_train)
        pred = est.predict(X_clf_test)
        acc = float(np.mean(pred == y_clf_test))
        print(f"  splitter={splitter:12s}  accuracy={acc:.4f}  n_leaves={est.get_n_leaves()}  OK")

    print()
    print("All runs completed without errors.")


if __name__ == "__main__":
    main()
