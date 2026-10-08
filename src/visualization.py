"""
Visualization module providing both publication-quality static charts (Matplotlib/Seaborn)
and rich interactive charts (Plotly).
"""

from typing import Dict, List, Optional
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


# Approximate centroid coordinates for all 38 Kabupaten/Kota in Jawa Timur
EAST_JAVA_COORDINATES: Dict[str, Dict[str, float]] = {
    "Bangkalan": {"lat": -7.0456, "lon": 112.9348},
    "Banyuwangi": {"lat": -8.2192, "lon": 114.3692},
    "Blitar": {"lat": -8.0983, "lon": 112.1681},
    "Bojonegoro": {"lat": -7.1502, "lon": 111.8817},
    "Bondowoso": {"lat": -7.9135, "lon": 113.8214},
    "Gresik": {"lat": -7.1566, "lon": 112.6555},
    "Jember": {"lat": -8.1724, "lon": 113.7007},
    "Jombang": {"lat": -7.5459, "lon": 112.2331},
    "Kediri": {"lat": -7.8480, "lon": 112.0178},
    "Kota Batu": {"lat": -7.8712, "lon": 112.5271},
    "Kota Blitar": {"lat": -8.0954, "lon": 112.1609},
    "Kota Kediri": {"lat": -7.8228, "lon": 112.0119},
    "Kota Madiun": {"lat": -7.6298, "lon": 111.5239},
    "Kota Malang": {"lat": -7.9797, "lon": 112.6304},
    "Kota Mojokerto": {"lat": -7.4726, "lon": 112.4381},
    "Kota Pasuruan": {"lat": -7.6453, "lon": 112.9075},
    "Kota Probolinggo": {"lat": -7.7543, "lon": 113.2159},
    "Kota Surabaya": {"lat": -7.2575, "lon": 112.7521},
    "Lamongan": {"lat": -7.1282, "lon": 112.4132},
    "Lumajang": {"lat": -8.1337, "lon": 113.2246},
    "Madiun": {"lat": -7.6167, "lon": 111.6500},
    "Magetan": {"lat": -7.6534, "lon": 111.3281},
    "Malang": {"lat": -8.1667, "lon": 112.6667},
    "Mojokerto": {"lat": -7.5500, "lon": 112.5000},
    "Nganjuk": {"lat": -7.6047, "lon": 111.9042},
    "Ngawi": {"lat": -7.4039, "lon": 111.4452},
    "Pacitan": {"lat": -8.2066, "lon": 111.0928},
    "Pamekasan": {"lat": -7.1593, "lon": 113.4748},
    "Pasuruan": {"lat": -7.7000, "lon": 112.8333},
    "Ponorogo": {"lat": -7.9697, "lon": 111.4644},
    "Probolinggo": {"lat": -7.8167, "lon": 113.2833},
    "Sampang": {"lat": -7.1872, "lon": 113.2394},
    "Sidoarjo": {"lat": -7.4478, "lon": 112.7183},
    "Situbondo": {"lat": -7.7063, "lon": 114.0044},
    "Sumenep": {"lat": -7.0167, "lon": 113.8667},
    "Trenggalek": {"lat": -8.0500, "lon": 111.7167},
    "Tuban": {"lat": -6.8972, "lon": 112.0649},
    "Tulungagung": {"lat": -8.0667, "lon": 111.9000},
}


# ==========================================
# 1. MATPLOTLIB / SEABORN STATIS (High Quality)
# ==========================================

def plot_dendrogram(
    X: np.ndarray, 
    labels: List[str], 
    method: str = "ward", 
    color_threshold: Optional[float] = None
) -> plt.Figure:
    """Generate high-resolution hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    Z = linkage(X, method=method)
    
    dendrogram(
        Z,
        ax=ax,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold,
        above_threshold_color="#7f8c8d"
    )
    ax.set_title(f"Dendrogram Hierarchical Clustering (Metode: {method.capitalize()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Kabupaten / Kota di Jawa Timur", fontsize=10)
    ax.set_ylabel("Jarak Euklides / Varians Gabungan", fontsize=10)
    plt.tight_layout()
    return fig


def plot_silhouette_sample_analysis(
    X: np.ndarray, 
    labels: np.ndarray, 
    n_clusters: int,
    cluster_names_map: Optional[Dict[int, str]] = None
) -> plt.Figure:
    """Plot per-sample silhouette coefficients with cluster bands and average score line."""
    sample_silhouette_values = silhouette_samples(X, labels)
    avg_score = float(silhouette_score(X, labels))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    y_lower = 10
    
    unique_labels = sorted(np.unique(labels))
    colors = cm.nipy_spectral(np.linspace(0, 0.85, n_clusters))
    
    for i, cluster_idx in enumerate(unique_labels):
        ith_cluster_values = sample_silhouette_values[labels == cluster_idx]
        ith_cluster_values.sort()
        size_cluster = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster
        
        color = colors[i]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
            label=cluster_names_map.get(cluster_idx, f"Klaster {cluster_idx}") if cluster_names_map else f"Klaster {cluster_idx}"
        )
        
        ax.text(-0.06, y_lower + 0.5 * size_cluster, f"K{cluster_idx} (n={size_cluster})", fontsize=9, fontweight="bold")
        y_lower = y_upper + 10
        
    ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=1.5, label=f"Rata-rata: {avg_score:.3f}")
    ax.set_title(f"Analisis Silhouette Per Sampel (K = {n_clusters})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Koefisien Silhouette", fontsize=10)
    ax.set_ylabel("Klaster", fontsize=10)
    ax.set_yticks([])
    ax.set_xlim([-0.2, 1.0])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_boxplot_distribution(
    df: pd.DataFrame, 
    features: List[str], 
    title: str = "Distribusi Pengeluaran"
) -> plt.Figure:
    """Plot horizontal boxplots for selected expenditure features."""
    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.4)), dpi=120)
    sns.boxplot(data=df[features], orient="h", linewidth=0.8, fliersize=3, palette="Set3", ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Nilai Pengeluaran (Rp)", fontsize=9)
    plt.tight_layout()
    return fig


# ==========================================
# 2. PLOTLY INTERAKTIF
# ==========================================

def plot_linkage_comparison_interactive(
    scores_df: pd.DataFrame, 
    chosen_method: str = "ward", 
    chosen_k: int = 3
) -> go.Figure:
    """Interactive line chart comparing silhouette scores across all linkage methods and K values."""
    fig = px.line(
        scores_df,
        x="K",
        y="Silhouette",
        color="Method",
        markers=True,
        title="Perbandingan Silhouette Score Antar Metode Linkage (K = 2 s/d 10)",
        labels={"K": "Jumlah Klaster (K)", "Silhouette": "Silhouette Score", "Method": "Metode Linkage"},
        color_discrete_map={
            "ward": "#1f77b4",
            "complete": "#ff7f0e",
            "average": "#2ca02c",
            "single": "#d62728"
        }
    )
    
    # Add vertical indicator for selected configuration
    fig.add_vline(
        x=chosen_k, 
        line_width=2, 
        line_dash="dash", 
        line_color="crimson",
        annotation_text=f"Terpilih: K={chosen_k}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_pca_2d_interactive(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    region_names: pd.Series,
    cluster_names_map: Dict[int, str],
    df_raw_features: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Interactive PCA 2D scatter plot with rich hover tooltips."""
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(df_scaled)
    var_exp = pca.explained_variance_ratio_
    
    plot_df = pd.DataFrame({
        "Kabupaten/Kota": region_names.values,
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "Cluster_ID": labels,
        "Nama_Klaster": [cluster_names_map.get(lbl, f"Klaster {lbl}") for lbl in labels]
    })
    
    # Add top spending category per region if raw features available
    if df_raw_features is not None:
        numeric_only = df_raw_features.select_dtypes(include=[np.number])
        top_cats = numeric_only.idxmax(axis=1)
        plot_df["Top_Pengeluaran"] = top_cats.values
        plot_df["Total_Pengeluaran_Mingguan"] = numeric_only.sum(axis=1).apply(lambda x: f"Rp {x:,.0f}").values
    else:
        plot_df["Top_Pengeluaran"] = "-"
        plot_df["Total_Pengeluaran_Mingguan"] = "-"
        
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Nama_Klaster",
        hover_name="Kabupaten/Kota",
        hover_data={
            "Nama_Klaster": True,
            "Top_Pengeluaran": True,
            "Total_Pengeluaran_Mingguan": True,
            "PC1": ":.2f",
            "PC2": ":.2f"
        },
        title=f"Visualisasi 2D Klastering via PCA ({var_exp.sum()*100:.1f}% Total Varians Dijelaskan)",
        labels={
            "PC1": f"PC 1 ({var_exp[0]*100:.1f}% Varians)",
            "PC2": f"PC 2 ({var_exp[1]*100:.1f}% Varians)",
            "Nama_Klaster": "Segmen Klaster"
        },
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return fig


def plot_cluster_distribution_interactive(
    df: pd.DataFrame, 
    cluster_col: str = "Nama_Klaster"
) -> go.Figure:
    """Interactive bar chart displaying member distribution per cluster."""
    counts = df[cluster_col].value_counts().reset_index()
    counts.columns = ["Segmen Klaster", "Jumlah Wilayah"]
    counts["Persentase (%)"] = (counts["Jumlah Wilayah"] / len(df) * 100).round(1)
    
    fig = px.bar(
        counts,
        x="Segmen Klaster",
        y="Jumlah Wilayah",
        color="Segmen Klaster",
        text="Jumlah Wilayah",
        hover_data={"Persentase (%)": True},
        title="Distribusi Jumlah Kabupaten/Kota per Segmen Klaster",
        template="plotly_white"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Jumlah Kabupaten/Kota")
    return fig


def plot_cluster_profile_heatmap_interactive(
    profile_df: pd.DataFrame,
    title: str = "Heatmap Median Pengeluaran per Segmen Klaster"
) -> go.Figure:
    """Interactive heatmap showing median expenditure across commodities for each cluster."""
    fig = px.imshow(
        profile_df.T,
        labels=dict(x="Segmen Klaster", y="Komoditas Pengeluaran", color="Median (Rp)"),
        x=profile_df.index,
        y=profile_df.columns,
        color_continuous_scale="YlGnBu",
        title=title,
        aspect="auto"
    )
    fig.update_layout(template="plotly_white", height=750)
    return fig


def plot_top_features_interactive(
    profile_df: pd.DataFrame, 
    top_n: int = 10
) -> go.Figure:
    """Interactive grouped bar chart comparing the top expenditure items across clusters."""
    top_cols = profile_df.max(axis=0).sort_values(ascending=False).head(top_n).index
    df_subset = profile_df[top_cols].reset_index()
    df_melt = df_subset.melt(id_vars=profile_df.index.name or "index", var_name="Komoditas", value_name="Median_Pengeluaran")
    cluster_var = profile_df.index.name or "index"
    
    fig = px.bar(
        df_melt,
        x="Komoditas",
        y="Median_Pengeluaran",
        color=cluster_var,
        barmode="group",
        title=f"Top {top_n} Komoditas Pengeluaran Tertinggi per Klaster",
        labels={"Median_Pengeluaran": "Median Pengeluaran Mingguan (Rp)", "Komoditas": "Jenis Makanan/Minuman", cluster_var: "Segmen Klaster"},
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=-0.35),
        height=500
    )
    return fig


def plot_gap_analysis_interactive(gap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Interactive horizontal bar chart ranking commodities by gap disparity ratio."""
    df_plot = gap_df.head(top_n).sort_values(by="Gap Ratio (X kali)", ascending=True)
    
    fig = px.bar(
        df_plot,
        x="Gap Ratio (X kali)",
        y="Komoditas / Fitur",
        orientation="h",
        text="Gap Ratio (X kali)",
        hover_data={"Klaster Tertinggi": True, "Klaster Terendah": True, "Gap Absolut (Rp)": True},
        title=f"Disparitas Konsumsi Antar Klaster (Top {top_n} Rasio Kesenjangan Terbesar)",
        template="plotly_white",
        color="Gap Ratio (X kali)",
        color_continuous_scale="Tealgrn"
    )
    fig.update_traces(texttemplate="%{text:.1f}x", textposition="outside")
    fig.update_layout(height=450, xaxis_title="Rasio Kesenjangan (Kelipatan)", yaxis_title="")
    return fig


def plot_east_java_map_interactive(
    df_clustered: pd.DataFrame,
    region_col: str = "Kabupaten/Kota",
    cluster_col: str = "Nama_Klaster",
    total_spending_col: Optional[str] = None
) -> go.Figure:
    """Interactive geospatial bubble map of East Java with real coordinates for all 38 regions."""
    map_data = []
    
    for _, row in df_clustered.iterrows():
        name = str(row[region_col]).strip()
        coords = EAST_JAVA_COORDINATES.get(name)
        
        # Fallback partial matching if needed
        if not coords:
            for k, v in EAST_JAVA_COORDINATES.items():
                if k.lower() in name.lower() or name.lower() in k.lower():
                    coords = v
                    break
                    
        if coords:
            map_data.append({
                "Kabupaten/Kota": name,
                "lat": coords["lat"],
                "lon": coords["lon"],
                "Segmen Klaster": row[cluster_col],
                "Total Pengeluaran": row[total_spending_col] if total_spending_col and total_spending_col in row else 50000
            })
            
    df_map = pd.DataFrame(map_data)
    
    # Adaptively support modern Plotly (>=6.0: scatter_map) and legacy Plotly (scatter_mapbox)
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            df_map,
            lat="lat",
            lon="lon",
            color="Segmen Klaster",
            hover_name="Kabupaten/Kota",
            hover_data={"Segmen Klaster": True, "lat": False, "lon": False},
            zoom=7.3,
            center={"lat": -7.7, "lon": 112.7},
            map_style="open-street-map",
            title="🗺️ Peta Persebaran Geospasial Segmen Pengeluaran di Jawa Timur",
            template="plotly_white"
        )
    elif hasattr(px, "scatter_mapbox"):
        fig = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            color="Segmen Klaster",
            hover_name="Kabupaten/Kota",
            hover_data={"Segmen Klaster": True, "lat": False, "lon": False},
            zoom=7.3,
            center={"lat": -7.7, "lon": 112.7},
            mapbox_style="open-street-map",
            title="🗺️ Peta Persebaran Geospasial Segmen Pengeluaran di Jawa Timur",
            template="plotly_white"
        )
    else:
        fig = px.scatter(
            df_map,
            x="lon",
            y="lat",
            color="Segmen Klaster",
            hover_name="Kabupaten/Kota",
            title="🗺️ Peta Persebaran Geospasial Segmen Pengeluaran di Jawa Timur",
            labels={"lon": "Bujur (Longitude)", "lat": "Lintang (Latitude)"},
            template="plotly_white"
        )

    fig.update_traces(marker=dict(size=14, opacity=0.88))
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=550)
    return fig

