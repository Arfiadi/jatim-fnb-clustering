"""
Package init for food expenditure clustering project in East Java (Jawa Timur).
"""

from .data_loader import load_raw_data, clean_data, get_numeric_features
from .preprocessing import (
    analyze_dropped_feature,
    detect_outliers_iqr,
    scale_features,
    select_features_analysis,
)
from .clustering import (
    calculate_cophenetic_correlation,
    compare_linkage_methods,
    fit_hierarchical_clustering,
    compute_sample_silhouette,
    get_cluster_profiles,
    get_gap_analysis,
    get_business_recommendations,
)

__all__ = [
    "load_raw_data",
    "clean_data",
    "get_numeric_features",
    "analyze_dropped_feature",
    "detect_outliers_iqr",
    "scale_features",
    "select_features_analysis",
    "calculate_cophenetic_correlation",
    "compare_linkage_methods",
    "fit_hierarchical_clustering",
    "compute_sample_silhouette",
    "get_cluster_profiles",
    "get_gap_analysis",
    "get_business_recommendations",
]
