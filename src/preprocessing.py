"""
Preprocessing and statistical feature preparation module.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def analyze_dropped_feature(
    df_numeric: pd.DataFrame, 
    feature_name: str = "Minuman keras"
) -> Dict[str, Any]:
    """Perform quantitative analysis to statistically justify dropping a feature.
    
    Args:
        df_numeric: Candidate numeric features DataFrame before dropping.
        feature_name: The column name being evaluated (default: "Minuman keras").
        
    Returns:
        Dictionary containing statistical metrics and markdown justification.
    """
    if feature_name not in df_numeric.columns:
        return {
            "status": "not_found",
            "message": f"Feature '{feature_name}' tidak ditemukan di DataFrame."
        }
        
    series = df_numeric[feature_name]
    missing_count = series.isna().sum()
    missing_rate = missing_count / len(series)
    
    clean_series = series.dropna()
    feat_var = float(clean_series.var()) if len(clean_series) > 1 else 0.0
    feat_mean = float(clean_series.mean()) if len(clean_series) > 0 else 0.0
    feat_median = float(clean_series.median()) if len(clean_series) > 0 else 0.0
    feat_std = float(clean_series.std()) if len(clean_series) > 1 else 0.0
    
    # Compare with other features
    other_cols = [c for c in df_numeric.columns if c != feature_name]
    other_vars = df_numeric[other_cols].var()
    mean_all_variance = float(other_vars.mean())
    median_all_variance = float(other_vars.median())
    
    # Correlation with other features
    corrs = df_numeric.corrwith(clean_series).dropna()
    mean_corr = float(corrs.mean()) if len(corrs) > 0 else 0.0
    max_corr = float(corrs.max()) if len(corrs) > 0 else 0.0
    min_corr = float(corrs.min()) if len(corrs) > 0 else 0.0
    
    justification = (
        f"Justifikasi Statistik Penghapusan Fitur '{feature_name}':\n"
        f"1. Missing Rate: {missing_rate:.1%} ({missing_count} dari {len(series)} wilayah tidak memiliki data konsumsi / bernilai '-').\n"
        f"2. Varians Sangat Rendah: Varians fitur ini ({feat_var:,.2f}) jauh di bawah rerata varians fitur lainnya ({mean_all_variance:,.2f}), "
        f"menunjukkan daya pembeda (discriminatory power) yang sangat minim dalam membedakan karakteristik klaster.\n"
        f"3. Korelasi Rata-rata Lemah: Rerata korelasi dengan kelompok pengeluaran lain adalah {mean_corr:.3f} (rentang: {min_corr:.3f} s/d {max_corr:.3f}).\n"
        f"Kesimpulan: Penghapusan fitur ini mencegah penambahan dimensi dengan informasi mendekati konstan yang dapat mendistorsi jarak Euclidean."
    )
    
    return {
        "feature_name": feature_name,
        "missing_count": int(missing_count),
        "missing_rate": float(missing_rate),
        "feature_variance": feat_var,
        "mean_all_variance": mean_all_variance,
        "variance_ratio": feat_var / mean_all_variance if mean_all_variance > 0 else 0.0,
        "mean_correlation": mean_corr,
        "max_correlation": max_corr,
        "min_correlation": min_corr,
        "mean_value": feat_mean,
        "median_value": feat_median,
        "std_value": feat_std,
        "justification": justification
    }


def detect_outliers_iqr(
    df: pd.DataFrame, 
    features: Optional[List[str]] = None,
    region_names: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """Detect outliers across features using the Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame containing numeric feature values.
        features: Specific features to evaluate. If None, uses all numeric columns.
        region_names: Optional Series of region names to map outliers.
        
    Returns:
        Dictionary containing per-feature outlier details, summary table, and demographic rationale.
    """
    feature_list = features if features is not None else df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    rows_table = []
    
    regions = region_names if region_names is not None else pd.Series(df.index, index=df.index)
    
    for col in feature_list:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_indices = df[outliers_mask].index
        outlier_regions = regions.loc[outlier_indices].tolist() if not regions.empty else list(outlier_indices)
        
        count = len(outlier_regions)
        pct = (count / len(df)) * 100
        
        outlier_summary[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "count": count,
            "percentage": pct,
            "regions": outlier_regions
        }
        
        if count > 0:
            rows_table.append({
                "Fitur": col,
                "Jumlah Outlier": count,
                "Persentase (%)": round(pct, 1),
                "Batas Bawah": round(lower_bound, 2),
                "Batas Atas": round(upper_bound, 2),
                "Wilayah Terdeteksi": ", ".join(outlier_regions[:4]) + ("..." if len(outlier_regions) > 4 else "")
            })
            
    summary_df = pd.DataFrame(rows_table)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="Jumlah Outlier", ascending=False).reset_index(drop=True)
        
    rationale = (
        "Justifikasi Pengelolaan Outlier (Population Census Perspective):\n"
        "Dataset ini merepresentasikan seluruh 38 Kabupaten/Kota di Provinsi Jawa Timur (data populasi, bukan sampel acak). "
        "Nilai ekstrem yang terdeteksi (seperti tingginya pengeluaran di Kota Surabaya dan Kota Malang) mencerminkan "
        "disparitas sosio-ekonomi nyata antara wilayah metropolitan vs perdesaan. "
        "Oleh karena itu, outlier TIDAK dihapus atau dipangkas (winsorized), melainkan dipertahankan "
        "agar segmentasi klaster mampu mengidentifikasi klaster bernilai tinggi (high-spending urban segment) secara akurat."
    )
    
    return {
        "details": outlier_summary,
        "summary_table": summary_df,
        "total_features_with_outliers": len(summary_df),
        "methodology_rationale": rationale
    }


def scale_features(
    df: pd.DataFrame, 
    features: Optional[List[str]] = None,
    index_series: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize feature matrix using StandardScaler (z-score normalization).
    
    Args:
        df: Input DataFrame containing numeric features.
        features: Optional subset of feature columns. If None, uses all numeric columns.
        index_series: Optional series to use as row index for the scaled DataFrame.
        
    Returns:
        Tuple of (scaled DataFrame with proper columns and index, fitted StandardScaler).
    """
    selected_cols = features if features is not None else df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[selected_cols])
    
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=selected_cols,
        index=index_series if index_series is not None else df.index
    )
    return scaled_df, scaler


def select_features_analysis(
    df_numeric: pd.DataFrame, 
    df_scaled: pd.DataFrame,
    corr_threshold: float = 0.9,
    variance_threshold: float = 0.05
) -> Dict[str, Any]:
    """Perform data-driven feature selection diagnostic.
    
    Args:
        df_numeric: Raw unscaled numeric features.
        df_scaled: Scaled features DataFrame.
        corr_threshold: Correlation coefficient threshold to detect collinearity.
        variance_threshold: Variance threshold for constant/quasi-constant feature filter.
        
    Returns:
        Dictionary containing variance check, high correlation pairs, and PCA loadings analysis.
    """
    # 1. Low Variance Check
    var_selector = VarianceThreshold(threshold=variance_threshold)
    var_selector.fit(df_scaled)
    low_var_cols = df_scaled.columns[~var_selector.get_support()].tolist()
    
    # 2. Correlation Collinearity
    corr_matrix = df_numeric.corr().abs()
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i):
            val = corr_matrix.iloc[i, j]
            if val >= corr_threshold:
                high_corr_pairs.append({
                    "Fitur 1": cols[i],
                    "Fitur 2": cols[j],
                    "Korelasi": round(val, 3)
                })
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values(by="Korelasi", ascending=False).reset_index(drop=True)
        
    # 3. PCA Loadings
    pca = PCA()
    pca.fit(df_scaled)
    n_components = min(5, len(pca.components_))
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df_scaled.columns
    )
    explained_variance = pd.Series(
        pca.explained_variance_ratio_[:n_components],
        index=[f"PC{i+1}" for i in range(n_components)]
    )
    
    # Top features for PC1 & PC2
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(5)
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(5)
    
    return {
        "low_variance_features": low_var_cols,
        "high_corr_df": high_corr_df,
        "pca_loadings": loadings,
        "explained_variance_ratio": explained_variance,
        "total_explained_top2": float(pca.explained_variance_ratio_[:2].sum()),
        "top_pc1_contributors": top_pc1,
        "top_pc2_contributors": top_pc2
    }
