"""
Hierarchical clustering modeling, validation metrics, and business interpretation module.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score


LINKAGE_METHODS = ["ward", "complete", "average", "single"]


def calculate_cophenetic_correlation(
    X: np.ndarray, 
    methods: Optional[List[str]] = None
) -> Dict[str, float]:
    """Calculate the Cophenetic Correlation Coefficient for hierarchical clustering linkage methods.
    
    The Cophenetic correlation measures how faithfully the dendrogram preserves the original 
    pairwise Euclidean distances between observations (closer to 1.0 is better).
    
    Args:
        X: Scaled feature matrix (n_samples, n_features).
        methods: List of linkage methods to evaluate.
        
    Returns:
        Dictionary mapping linkage method name to its cophenetic coefficient.
    """
    eval_methods = methods or LINKAGE_METHODS
    dist_matrix = pdist(X)
    results = {}
    
    for method in eval_methods:
        try:
            Z = linkage(X, method=method)
            c, _ = cophenet(Z, dist_matrix)
            results[method] = float(c)
        except Exception as e:
            results[method] = np.nan
            
    return results


def compare_linkage_methods(
    X: np.ndarray, 
    k_range: range = range(2, 11),
    methods: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare silhouette scores and cophenetic correlation across multiple linkage methods and K values.
    
    Args:
        X: Scaled feature matrix.
        k_range: Range of cluster counts K to test (default 2 to 10).
        methods: Linkage methods to test.
        
    Returns:
        Tuple containing:
            - scores_df: DataFrame with columns ['Method', 'K', 'Silhouette']
            - summary_df: Summary comparison table with Best K, Max Silhouette, and Cophenetic Corr.
    """
    eval_methods = methods or LINKAGE_METHODS
    cophenetic_scores = calculate_cophenetic_correlation(X, eval_methods)
    
    records = []
    summary_rows = []
    
    for method in eval_methods:
        best_k = None
        best_score = -1.0
        
        for k in k_range:
            try:
                model = AgglomerativeClustering(n_clusters=k, linkage=method)
                labels = model.fit_predict(X)
                score = float(silhouette_score(X, labels))
            except Exception:
                score = np.nan
                
            records.append({
                "Method": method,
                "K": k,
                "Silhouette": score
            })
            
            if score > best_score:
                best_score = score
                best_k = k
                
        c_score = cophenetic_scores.get(method, np.nan)
        summary_rows.append({
            "Metode Linkage": method.capitalize(),
            "Best K": best_k,
            "Best Silhouette": round(best_score, 4),
            "Cophenetic Corr": round(c_score, 4),
            "Status": "Dipilih (Ward)" if method == "ward" else "Komparator"
        })
        
    scores_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summary_rows).sort_values(by="Best Silhouette", ascending=False).reset_index(drop=True)
    return scores_df, summary_df


def fit_hierarchical_clustering(
    X: np.ndarray, 
    n_clusters: int = 3, 
    linkage_method: str = "ward"
) -> Tuple[AgglomerativeClustering, np.ndarray, float]:
    """Fit Agglomerative Hierarchical Clustering model on feature matrix.
    
    Args:
        X: Scaled feature matrix.
        n_clusters: Number of clusters.
        linkage_method: Linkage criterion ('ward', 'complete', 'average', 'single').
        
    Returns:
        Tuple of (fitted model, cluster labels, silhouette score).
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    score = float(silhouette_score(X, labels))
    return model, labels, score


def compute_sample_silhouette(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calculate silhouette coefficient for each individual sample."""
    return silhouette_samples(X, labels)


def get_cluster_profiles(
    df_features: pd.DataFrame, 
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute median and mean expenditure profiles per cluster.
    
    Args:
        df_features: Unscaled feature values.
        labels: Cluster assignment array.
        label_names: Optional mapping from cluster index to human-readable name.
        
    Returns:
        Tuple of (median_profile_df, mean_profile_df).
    """
    df_temp = df_features.copy()
    if label_names:
        cluster_col = [label_names.get(lbl, f"Klaster {lbl}") for lbl in labels]
    else:
        cluster_col = [f"Klaster {lbl}" for lbl in labels]
        
    df_temp["Cluster_Name"] = cluster_col
    median_profile = df_temp.groupby("Cluster_Name").median()
    mean_profile = df_temp.groupby("Cluster_Name").mean()
    return median_profile, mean_profile


def get_gap_analysis(
    cluster_profile: pd.DataFrame, 
    top_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """Analyze the expenditure disparity (gap ratio and absolute gap) across clusters.
    
    Args:
        cluster_profile: Median expenditure DataFrame per cluster.
        top_features: Optional list of top features to analyze.
        
    Returns:
        DataFrame ranking commodities by disparity ratio.
    """
    features = top_features or cluster_profile.columns.tolist()
    gap_data = []
    
    for feat in features:
        if feat in cluster_profile.columns:
            series = cluster_profile[feat]
            max_cluster = series.idxmax()
            min_cluster = series.idxmin()
            max_val = float(series.max())
            min_val = float(series.min())
            
            gap_ratio = (max_val / min_val) if min_val > 0 else np.nan
            gap_abs = max_val - min_val
            
            gap_data.append({
                "Komoditas / Fitur": feat,
                "Klaster Tertinggi": max_cluster,
                "Nilai Tertinggi (Rp)": round(max_val, 2),
                "Klaster Terendah": min_cluster,
                "Nilai Terendah (Rp)": round(min_val, 2),
                "Gap Ratio (X kali)": round(gap_ratio, 2) if not np.isnan(gap_ratio) else 0.0,
                "Gap Absolut (Rp)": round(gap_abs, 2)
            })
            
    gap_df = pd.DataFrame(gap_data)
    if not gap_df.empty:
        gap_df = gap_df.sort_values(by="Gap Ratio (X kali)", ascending=False).reset_index(drop=True)
    return gap_df


def assign_business_cluster_names(
    cluster_profile: pd.DataFrame
) -> Dict[int, str]:
    """Automatically rank and name clusters based on median total spending power.
    
    Returns a dictionary mapping cluster index (or cluster identifier) to descriptive business label.
    """
    total_spending = cluster_profile.sum(axis=1).sort_values(ascending=False)
    clusters_sorted = total_spending.index.tolist()
    
    naming_templates = [
        "🏙️ Urban High-Spender (Metropolitan)",
        "🏘️ Suburban Moderate (Sentra Industri & Pertumbuhan)",
        "🌾 Rural Conservative (Perdesaan & Agrikultur)",
        "🌿 Semi-Urban Developing (Wilayah Berkembang)",
        "🏭 Industrial Growth Zone"
    ]
    
    mapping = {}
    for i, cl_name in enumerate(clusters_sorted):
        label = naming_templates[i] if i < len(naming_templates) else f"Klaster Segment {i+1}"
        mapping[cl_name] = label
        
    return mapping


def get_business_recommendations() -> Dict[str, Dict[str, Any]]:
    """Provide comprehensive, structured business recommendations framed for FMCG and Consumer Analytics.
    
    Covers Who, What, So What, and Now What across FMCG/Retail, Food Delivery, and Regional Government.
    """
    return {
        "🏙️ Urban High-Spender (Metropolitan)": {
            "who": "Pusat pertumbuhan ekonomi kota besar (cth: Kota Surabaya, Kota Malang, Sidoarjo). Karakteristik penduduk berpendapatan tinggi, mobilitas cepat, dan gaya hidup instan.",
            "what": "Pengeluaran tertinggi pada: Nasi campur/rames, Mie bakso, Ayam/daging olahan matang, dan Minuman jadi (kopi susu & teh siap saji).",
            "so_what": "Pasar dengan purchasing power tertinggi. Konsumen mengutamakan kepraktisan, brand reputation, dan higienitas dibandingkan harga murah.",
            "now_what": {
                "fmcg": "Perbanyak penetrasi produk Ready-to-Drink (RTD) premium, frozen meal siap saji, dan paket bumbu gourmet di modern trade (supermarket/minimarket).",
                "food_delivery": "Prioritaskan ekspansi mitra restoran premium, bundling menu makan siang perkantoran, dan fitur pesan terjadwal.",
                "pemprov": "Fokus pada pengawasan higienitas sanitasi kuliner jalanan dan edukasi pengurangan kadar gula pada minuman kekinian."
            }
        },
        "🏘️ Suburban Moderate (Sentra Industri & Pertumbuhan)": {
            "who": "Kabupaten penyangga ekonomi dan koridor industri (cth: Gresik, Pasuruan, Mojokerto). Karakteristik masyarakat pekerja industri dan keluarga muda produktif.",
            "what": "Pengeluaran moderat namun stabil pada makanan pokok olahan, lauk pauk olahan terjangkau, dan mie instan.",
            "so_what": "Pasar dengan volume konsumsi sangat tinggi yang sensitif terhadap perbandingan 'value-for-money'.",
            "now_what": {
                "fmcg": "Gunakan strategi 'Family Pack' dan multipack mie instan/bumbu siap pakai dengan harga terjangkau di warung tradisional dan minimarket.",
                "food_delivery": "Optimalkan promo free ongkir radius dekat, perluas mitra warteg/kedai lokal, dan loyalty points ramah kantong pekerja.",
                "pemprov": "Pantau stabilitas harga bahan pokok makanan jadi di kantin pabrik dan pasar tradisional untuk menjaga daya beli pekerja."
            }
        },
        "🌾 Rural Conservative (Perdesaan & Agrikultur)": {
            "who": "Kabupaten dengan basis agraris, maritim, dan tapal kuda (cth: Pacitan, Ponorogo, Sampang, Bondowoso). Pola belanja berorientasi hemat dan esensial.",
            "what": "Pengeluaran makanan jadi lebih rendah; didominasi gorengan, jajanan tradisional, dan mie instan sesekali, dengan porsi memasak sendiri di rumah lebih tinggi.",
            "so_what": "Konsumen sangat price-sensitive. Makanan jadi hanya dikonsumsi untuk keperluan praktis tertentu atau momen sosial.",
            "now_what": {
                "fmcg": "Fokus pada produk kemasan sachet ekonomis (Rp 1.000 - Rp 2.500) dan distribusi penetrasi ke warung kelontong pedesaan (GT channel).",
                "food_delivery": "Model cloud kitchen sederhana di pusat kota kecamatan atau integrasi kurir lokal berbasis komunitas/WhatsApp.",
                "pemprov": "Tingkatkan program bantuan ketahanan pangan, diversifikasi pangan bergizi seimbang, dan intervensi gizi untuk pencegahan stunting."
            }
        }
    }
