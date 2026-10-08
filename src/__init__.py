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
from . import visualization
from .visualization import (
    plot_dendrogram,
    plot_silhouette_sample_analysis,
    plot_boxplot_distribution,
    plot_linkage_comparison_interactive,
    plot_pca_2d_interactive,
    plot_cluster_donut_interactive,
    plot_cluster_distribution_interactive,
    plot_cluster_profile_heatmap_interactive,
    plot_top_features_interactive,
    plot_gap_analysis_interactive,
    plot_east_java_map_interactive,
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
    "visualization",
    "plot_dendrogram",
    "plot_silhouette_sample_analysis",
    "plot_boxplot_distribution",
    "plot_linkage_comparison_interactive",
    "plot_pca_2d_interactive",
    "plot_cluster_donut_interactive",
    "plot_cluster_distribution_interactive",
    "plot_cluster_profile_heatmap_interactive",
    "plot_top_features_interactive",
    "plot_gap_analysis_interactive",
    "plot_east_java_map_interactive",
]
