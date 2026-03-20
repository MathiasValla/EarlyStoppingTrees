from .._lib.sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from ._classes import (
    EarlyStopDecisionTreeClassifier,
    EarlyStopDecisionTreeRegressor,
    ExtraObliqueDecisionTreeClassifier,
    ExtraObliqueDecisionTreeRegressor,
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeRegressor,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)
from ._honest_tree import HonestTreeClassifier
from ._multiview import MultiViewDecisionTreeClassifier
from ._neighbors import compute_forest_similarity_matrix

__all__ = [
    "EarlyStopDecisionTreeClassifier",
    "EarlyStopDecisionTreeRegressor",
    "ExtraObliqueDecisionTreeClassifier",
    "ExtraObliqueDecisionTreeRegressor",
    "compute_forest_similarity_matrix",
    "UnsupervisedDecisionTree",
    "UnsupervisedObliqueDecisionTree",
    "ObliqueDecisionTreeClassifier",
    "ObliqueDecisionTreeRegressor",
    "PatchObliqueDecisionTreeClassifier",
    "PatchObliqueDecisionTreeRegressor",
    "HonestTreeClassifier",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "MultiViewDecisionTreeClassifier",
]
