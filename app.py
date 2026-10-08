"""
Streamlit Application: Platform Analisis Klaster Pengeluaran Makanan & Minuman Jawa Timur
Framed for FMCG & Consumer Analytics Insights.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path across all cloud environments
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import src
from src.clustering import assign_business_cluster_names, get_business_recommendations
from src.visualization import (
    plot_boxplot_distribution,
    plot_cluster_donut_interactive,
    plot_cluster_distribution_interactive,
    plot_cluster_profile_heatmap_interactive,
    plot_dendrogram,
    plot_east_java_map_interactive,
    plot_gap_analysis_interactive,
    plot_linkage_comparison_interactive,
    plot_pca_2d_interactive,
    plot_silhouette_sample_analysis,
    plot_top_features_interactive,
)

# Set Streamlit page config
st.set_page_config(
    page_title="Analisis Klaster Konsumsi Makanan & Minuman Jawa Timur",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme Adaptive for Light & Dark Mode)
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_pipeline_data():
    """Cache data loading, cleaning, and diagnostics."""
    df_raw = src.load_raw_data()
    df_clean, regions, df_numeric = src.clean_data(df_raw)
    
    # Quantitative drop evaluation
    drop_info = src.analyze_dropped_feature(df_numeric, "Minuman keras")
    
    # Numeric features used for modeling
    df_features = df_numeric.drop(columns=["Minuman keras"], errors="ignore")
    feature_names = df_features.columns.tolist()
    
    # Outliers
    outlier_info = src.detect_outliers_iqr(df_features, region_names=regions)
    
    # Scaling
    df_scaled, scaler = src.scale_features(df_features, index_series=regions)
    
    # Diagnostic feature selection
    fs_diag = src.select_features_analysis(df_features, df_scaled)
    
    # Linkage comparison
    scores_df, summary_df = src.compare_linkage_methods(df_scaled.values)
    
    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "regions": regions,
        "df_numeric": df_numeric,
        "df_features": df_features,
        "feature_names": feature_names,
        "drop_info": drop_info,
        "outlier_info": outlier_info,
        "df_scaled": df_scaled,
        "fs_diag": fs_diag,
        "scores_df": scores_df,
        "summary_df": summary_df,
    }


pipeline_data = get_cached_pipeline_data()
df_raw = pipeline_data["df_raw"]
df_features = pipeline_data["df_features"]
df_scaled = pipeline_data["df_scaled"]
regions = pipeline_data["regions"]
feature_names = pipeline_data["feature_names"]
drop_info = pipeline_data["drop_info"]
outlier_info = pipeline_data["outlier_info"]
scores_df = pipeline_data["scores_df"]
summary_df = pipeline_data["summary_df"]
fs_diag = pipeline_data["fs_diag"]

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model Klaster")
    linkage_method = st.selectbox(
        "Metode Linkage:",
        options=["ward", "complete", "average", "single"],
        index=0,
        help="Ward meminimalkan varians gabungan (rekomendasi standard). Complete menggunakan jarak maksimum, Average menggunakan rata-rata jarak."
    )
    
    n_clusters = st.slider(
        "Jumlah Klaster (K):",
        min_value=2,
        max_value=10,
        value=3,
        help="Pilih jumlah klaster berdasarkan Silhouette Score dan analisis dendrogram."
    )
    
    # Fit model on selected parameters
    model, cluster_labels, sil_score = src.fit_hierarchical_clustering(
        df_scaled.values, 
        n_clusters=n_clusters, 
        linkage_method=linkage_method
    )
    
    # Calculate profiles
    median_profile_raw, mean_profile_raw = src.get_cluster_profiles(df_features, cluster_labels)
    default_business_names = assign_business_cluster_names(median_profile_raw)
    
    st.markdown("---")
    st.markdown("### 🏷️ Kustomisasi Label Klaster")
    use_business_names = st.toggle("Gunakan Label Deskriptif Bisnis", value=True)
    
    cluster_names_map = {}
    with st.expander("Ubah Nama Klaster Manual", expanded=False):
        for i in range(n_clusters):
            curr_id = f"Klaster {i}"
            suggested_label = default_business_names.get(curr_id, f"Segmen {i+1}") if use_business_names else f"Klaster {i}"
            cluster_names_map[i] = st.text_input(
                f"Label Klaster {i}:",
                value=suggested_label,
                key=f"cluster_name_input_{i}"
            )
            
    st.markdown("---")
    st.markdown("### 📊 Filter Fitur Boxplot")
    default_box_cols = [
        "Nasi campur/rames",
        "Mie bakso, mie rebus, mie goreng",
        "Ayam/daging matang (ayam goreng, rendang, dsb)",
        "Minuman jadi (kopi, kopi susu, teh, susu coklat, dsb)",
        "Mie instan"
    ]
    valid_box_defaults = [c for c in default_box_cols if c in feature_names]
    selected_box_features = st.multiselect(
        "Pilih fitur untuk diinspeksi:",
        options=feature_names,
        default=valid_box_defaults
    )
    
    st.markdown("---")
    st.caption("Proyek Pemodelan Statistik Terapan — Analisis Klaster Jawa Timur 2024")

# Update mapped names
named_labels = [cluster_names_map[lbl] for lbl in cluster_labels]
df_result = df_features.copy()
df_result["Kabupaten/Kota"] = regions.values
df_result["Cluster_ID"] = cluster_labels
df_result["Nama_Klaster"] = named_labels
df_result["Total_Pengeluaran"] = df_features.sum(axis=1)

# Named profiles
median_profile_named, mean_profile_named = src.get_cluster_profiles(
    df_features, 
    cluster_labels, 
    label_names=cluster_names_map
)
gap_analysis_df = src.get_gap_analysis(median_profile_named)

# ==========================================
# HEADER & EXECUTIVE METRICS
# ==========================================
st.markdown('<div class="main-title">🍜 Segmentasi Pengeluaran Makanan & Minuman Jawa Timur</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hierarchical Clustering 38 Kabupaten/Kota Berdasarkan Pola Pengeluaran Makanan & Minuman Jadi — Perspektif FMCG & Consumer Analytics</div>', unsafe_allow_html=True)

# KPI row
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Total Wilayah", f"{len(regions)} Kab/Kota")
with col_m2:
    st.metric("Jumlah Klaster", f"K = {n_clusters}")
with col_m3:
    st.metric("Silhouette Score", f"{sil_score:.3f}")
with col_m4:
    c_score_val = summary_df.loc[summary_df["Metode Linkage"] == linkage_method.capitalize(), "Cophenetic Corr"].values
    c_disp = f"{c_score_val[0]:.3f}" if len(c_score_val) > 0 else "-"
    st.metric("Cophenetic Corr", c_disp)
with col_m5:
    st.metric("Fitur Dimodelkan", f"{len(feature_names)} Komoditas")

st.markdown("---")

# ==========================================
# MAIN NAVIGATION TABS
# ==========================================
tab_intro, tab_eda, tab_clustering, tab_business, tab_export = st.tabs([
    "ℹ️ 1. Ringkasan & Konteks Industri",
    "🔍 2. Eksplorasi & Preprocessing",
    "📊 3. Hasil & Evaluasi Klaster",
    "💼 4. Profil & Rekomendasi Bisnis",
    "📥 5. Tabel Data & Ekspor"
])

# ----------------------------------------------------
# TAB 1: RINGKASAN & KONTEKS INDUSTRI
# ----------------------------------------------------
with tab_intro:
    st.info("💡 **Executive Summary:** Segmentasi ini memetakan heterogenitas belanja konsumsi makanan dan minuman jadi di 38 Kabupaten/Kota Provinsi Jawa Timur. Hasil klasterisasi memberikan dasar penetapan strategi distribusi regional, optimasi bauran produk (product portfolio), dan pricing berkeadilan.")
    
    col_intro1, col_intro2 = st.columns([3, 2])
    with col_intro1:
        st.subheader("🎯 Nilai Strategis untuk Industri FMCG & Consumer Analytics")
        st.markdown("""
        Analisis ini mensimulasikan alur kerja tim **Consumer Insights & Regional Strategy** di perusahaan FMCG multinasional maupun nasional (seperti *Unilever Indonesia, Indofood, Wings Group*) serta platform On-Demand Services (*GoFood / GrabFood*):
        
        1. **Regional Pricing & Packaging Strategy:** Menentukan wilayah mana yang cocok untuk kemasan sachet ekonomis vs multipack family size.
        2. **Trade Marketing & Channel Prioritization:** Mengalokasikan proporsi pasokan ke *General Trade* (warung tradisional) vs *Modern Trade* (minimarket & supermarket).
        3. **Supply Chain & Cold Storage Placement:** Memetakan sentra pengeluaran makanan basah dan olahan daging untuk menentukan titik hub logistik rantai dingin.
        4. **Pencegahan Food Wastage & Intervensi Kebijakan Publik:** Membantu pemangku kebijakan daerah mendeteksi disparitas konsumsi gizi siap saji antar daerah.
        """)
        
    with col_intro2:
        st.subheader("📐 Ringkasan Metodologi")
        st.markdown("""
        - **Sumber Data:** Badan Pusat Statistik (BPS) Jawa Timur (Survei Sosial Ekonomi Nasional 2024).
        - **Cakupan Data:** Data sensus populasi 38 Kabupaten/Kota (tidak ada sampling bias).
        - **Metode Utama:** *Agglomerative Hierarchical Clustering* dengan standardisasi Z-score.
        - **Evaluasi Matematis:** Silhouette Analysis (rata-rata & per sampel) serta Koefisien Korelasi Kofenet (*Cophenetic Correlation Coefficient*).
        - **Daya Saing Model:** Komparasi 4 metode linkage (*Ward, Complete, Average, Single*) secara transparan.
        """)
        
    st.markdown("---")
    st.subheader("🗺️ Ringkasan Visualisasi Spasial Singkat")
    fig_map_preview = plot_east_java_map_interactive(
        df_result,
        region_col="Kabupaten/Kota",
        cluster_col="Nama_Klaster",
        total_spending_col="Total_Pengeluaran"
    )
    st.plotly_chart(fig_map_preview, use_container_width=True, key="preview_map_chart")

# ----------------------------------------------------
# TAB 2: EKSPLORASI & PREPROCESSING
# ----------------------------------------------------
with tab_eda:
    eda_sub1, eda_sub2, eda_sub3, eda_sub4 = st.tabs([
        "Tinjauan Data Mentah",
        "Justifikasi Drop Fitur",
        "Deteksi Outlier (Metode IQR)",
        "Seleksi Fitur & Korelasi"
    ])
    
    with eda_sub1:
        st.subheader("1. Tinjauan Data Awal BPS")
        st.dataframe(df_raw.head(10), use_container_width=True)
        col_raw1, col_raw2 = st.columns(2)
        with col_raw1:
            st.info(f"Jumlah Observasi: **{df_raw.shape[0]} Wilayah** | Jumlah Kolom Mentah: **{df_raw.shape[1]} Kolom**")
        with col_raw2:
            nan_summary = df_raw.isna().sum()
            missing_cols = nan_summary[nan_summary > 0]
            st.caption(f"Kolom dengan nilai kosong awal: {len(missing_cols)} kolom")

    with eda_sub2:
        st.subheader("2. Analisis Kuantitatif & Justifikasi Drop 'Minuman Keras'")
        st.markdown(f"""
        Dalam metodologi pemodelan yang baik, penghapusan kolom harus didukung oleh pengujian empiris kuantitatif:
        """)
        
        c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
        c_stat1.metric("Missing Rate", f"{drop_info['missing_rate']:.1%}", f"{drop_info['missing_count']} dari 38 wilayah")
        c_stat2.metric("Varians Fitur", f"{drop_info['feature_variance']:,.1f}")
        c_stat3.metric("Rerata Varians Fitur Lain", f"{drop_info['mean_all_variance']:,.1f}")
        c_stat4.metric("Rasio Varians", f"{drop_info['variance_ratio']:.4f}")
        
        st.markdown(f"""
        > **Kesimpulan Statistik:**
        > Nilai varians fitur ini sangat mendekati 0 relatif terhadap kelompok pengeluaran lainnya (rasio varians hanya {drop_info['variance_ratio']:.4f}). Ditambah dengan 34.2% data berupa nilai '-' (tidak ada konsumsi), memasukkan fitur ini akan menambah dimensi bising (*noise*) tanpa memberikan daya pembeda (*discriminatory power*) yang bermakna bagi algoritma pengelompokan berbasis jarak.
        """)

    with eda_sub3:
        st.subheader("3. Deteksi Outlier Berdasarkan IQR Method")
        st.markdown(f"""
        Pengujian menggunakan batas interkuartil $[Q1 - 1.5 \\times IQR, Q3 + 1.5 \\times IQR]$ mendeteksi **{outlier_info['total_features_with_outliers']} fitur** yang memiliki nilai ekstrem.
        """)
        
        st.info(
            "💡 **Justifikasi Ilmiah Penanganan Outlier (Data Sensus Wilayah):**\n\n"
            "Karena data ini mencakup 100% populasi Kabupaten/Kota di Jawa Timur (bukan sampel acak survei), "
            "nilai pencilan tinggi di wilayah seperti **Kota Surabaya** dan **Kota Malang** merefleksikan realitas "
            "sosio-ekonomi perkotaan metropolitan. Memangkas (trimming/winsorizing) outlier akan menghilangkan kemampuan "
            "model untuk mendeteksi segmen *High-Spend Urban*. Oleh karena itu, seluruh observasi dipertahankan secara utuh."
        )
        
        st.dataframe(outlier_info["summary_table"], use_container_width=True)

    with eda_sub4:
        st.subheader("4. Analisis Multikolinieritas & Kontribusi Komponen Utama (PCA)")
        st.markdown("Deteksi pasangan komoditas dengan korelasi Pearson sangat kuat ($r > 0.90$):")
        if not fs_diag["high_corr_df"].empty:
            st.dataframe(fs_diag["high_corr_df"], use_container_width=True)
        else:
            st.info("Tidak ada pasangan fitur dengan korelasi di atas ambang batas 0.90.")
            
        st.markdown(f"**Proporsi Varians Dijelaskan oleh 2 Komponen Utama Teratas:** **{fs_diag['total_explained_top2']*100:.1f}%**")
        st.dataframe(fs_diag["pca_loadings"].head(10), use_container_width=True)

# ----------------------------------------------------
# TAB 3: HASIL & EVALUASI KLASTER
# ----------------------------------------------------
with tab_clustering:
    clust_sub1, clust_sub2, clust_sub3 = st.tabs([
        "Evaluasi Komprehensif (Linkage & K)",
        "Visualisasi Gambaran Klaster (PCA & Distribusi)",
        "Peta Persebaran Geospasial"
    ])
    
    with clust_sub1:
        st.subheader("1. Komparasi 4 Metode Linkage & Silhouette Score")
        
        col_comp1, col_comp2 = st.columns([3, 2])
        with col_comp1:
            fig_linkage = plot_linkage_comparison_interactive(
                scores_df, 
                chosen_method=linkage_method, 
                chosen_k=n_clusters
            )
            st.plotly_chart(fig_linkage, use_container_width=True, key="linkage_curve_chart")
        with col_comp2:
            st.markdown("##### Tabel Evaluasi Komparatif")
            st.dataframe(summary_df, use_container_width=True)
            st.markdown("""
            **Catatan Metodologi:**
            - **Ward's Method:** Menghasilkan klaster paling seimbang dan homogen dari sisi varians internal.
            - **Cophenetic Correlation:** Mengukur seberapa baik struktur hierarki pohon mempertahankan jarak Euclidean asli.
            """)
            
        st.markdown("---")
        col_dendro, col_sil_sample = st.columns(2)
        with col_dendro:
            st.markdown("##### Dendrogram Hierarki")
            fig_d = plot_dendrogram(df_scaled.values, labels=regions.tolist(), method=linkage_method, dark_mode=True)
            st.pyplot(fig_d)
        with col_sil_sample:
            st.markdown(f"##### Silhouette Analysis Per Sampel (K={n_clusters})")
            fig_s = plot_silhouette_sample_analysis(
                df_scaled.values, 
                cluster_labels, 
                n_clusters, 
                cluster_names_map=cluster_names_map,
                dark_mode=True
            )
            st.pyplot(fig_s)

        with st.expander("🔍 Perbesar Tampilan Dendrogram (Full Width - 38 Wilayah)", expanded=False):
            fig_d_wide = plot_dendrogram(df_scaled.values, labels=regions.tolist(), method=linkage_method, dark_mode=True)
            st.pyplot(fig_d_wide)

    with clust_sub2:
        st.subheader(f"2. Sebaran Klaster & Proyeksi PCA 2D (K={n_clusters})")
        
        # 1. Sleek KPI Metric Cards per Cluster
        sorted_clusters = sorted(df_result["Nama_Klaster"].unique())
        metric_cols = st.columns(len(sorted_clusters))
        for idx, cl_name in enumerate(sorted_clusters):
            subset = df_result[df_result["Nama_Klaster"] == cl_name]
            count = len(subset)
            pct = (count / len(df_result)) * 100
            med_val = subset["Total_Pengeluaran"].median() if "Total_Pengeluaran" in subset.columns else 0
            with metric_cols[idx]:
                st.metric(
                    label=cl_name,
                    value=f"{count} Wilayah ({pct:.1f}%)",
                    delta=f"Median: Rp {med_val:,.0f}/mgg" if med_val > 0 else None,
                    delta_color="off"
                )

        # 2. Main Visual Grid: Dominant PCA (2.2) + Proportional Donut Chart (1)
        col_pca, col_dist = st.columns([2.2, 1])
        with col_pca:
            fig_pca = plot_pca_2d_interactive(
                df_scaled, 
                cluster_labels, 
                regions, 
                cluster_names_map,
                df_raw_features=df_features
            )
            st.plotly_chart(fig_pca, use_container_width=True, key="pca_scatter_chart")
        with col_dist:
            fig_donut = plot_cluster_donut_interactive(df_result, cluster_col="Nama_Klaster")
            st.plotly_chart(fig_donut, use_container_width=True, key="cluster_donut_chart")
            
            with st.container(border=True):
                st.markdown("**💡 Interpretasi Klaster & PCA:**")
                st.caption(
                    "Dimensi PC1 dan PC2 mereduksi 24 fitur konsumsi menjadi koordinat 2D (varians gabungan ~59%). "
                    "Klaster terpisah dengan batas wajar tanpa keberadaan singleton atau klaster outlier ekstrem."
                )

        st.markdown("---")
        st.markdown("##### 🏛️ Rincian Kabupaten / Kota Anggota di Setiap Klaster")
        cols_members = st.columns(min(n_clusters, 4))
        for idx, cl_name in enumerate(sorted_clusters):
            members = df_result[df_result["Nama_Klaster"] == cl_name]["Kabupaten/Kota"].tolist()
            with cols_members[idx % len(cols_members)]:
                with st.expander(f"📍 {cl_name} ({len(members)} Wilayah)", expanded=True):
                    for m in members:
                        st.markdown(f"- **{m}**")

    with clust_sub3:
        st.subheader("3. Peta Geospasial 38 Kabupaten/Kota di Jawa Timur")
        fig_map = plot_east_java_map_interactive(
            df_result,
            region_col="Kabupaten/Kota",
            cluster_col="Nama_Klaster",
            total_spending_col="Total_Pengeluaran"
        )
        st.plotly_chart(fig_map, use_container_width=True, key="east_java_map_chart")

# ----------------------------------------------------
# TAB 4: PROFIL & REKOMENDASI BISNIS
# ----------------------------------------------------
with tab_business:
    biz_sub1, biz_sub2, biz_sub3, biz_sub4 = st.tabs([
        "Rekomendasi Strategis (FMCG/Food Delivery/Pemda)",
        "Analisis Disparitas (Gap Analysis)",
        "Profil Median (Heatmap & Bar Chart)",
        "Distribusi Komparatif (Boxplot)"
    ])
    
    with biz_sub1:
        st.subheader("📋 Rekomendasi Strategis Berbasis Klaster")
        st.markdown("Pilih segmen klaster di bawah ini untuk menelaah profil konsumen dan strategi aksi yang proporsional:")
        
        recs_data = get_business_recommendations()
        cluster_list = sorted(df_result["Nama_Klaster"].unique())
        
        # Tabs for each cluster + overview matrix tab
        cluster_tab_names = [f"{cl.split('(')[0].strip()}" for cl in cluster_list] + ["📊 Matriks Komparatif (Overview)"]
        sub_recs_tabs = st.tabs(cluster_tab_names)
        
        for idx, cluster_id in enumerate(cluster_list):
            with sub_recs_tabs[idx]:
                members_list = sorted(df_result[df_result["Nama_Klaster"] == cluster_id]["Kabupaten/Kota"].tolist())
                spending_subset = df_result[df_result["Nama_Klaster"] == cluster_id]["Total_Pengeluaran"] if "Total_Pengeluaran" in df_result.columns else pd.Series([0])
                med_val = spending_subset.median()
                
                # Find matching recommendation key
                rec_entry = None
                for key, val in recs_data.items():
                    if key in cluster_id or cluster_id in key:
                        rec_entry = val
                        break
                
                # Row 1: Balanced Header Information (Two proportional cards)
                c_info1, c_info2 = st.columns([1.15, 1])
                with c_info1:
                    with st.container(border=True):
                        st.markdown(f"#### {cluster_id}")
                        st.markdown(f"**👤 Profil Sasaran (Who):**\n{rec_entry['who'] if rec_entry else '-'}")
                        st.markdown(f"**🍜 Pola Konsumsi (What):**\n{rec_entry['what'] if rec_entry else '-'}")
                        st.markdown(f"**💼 Implikasi Pasar (So What):**\n{rec_entry['so_what'] if rec_entry else '-'}")
                
                with c_info2:
                    with st.container(border=True):
                        st.markdown(f"#### 🏛️ Wilayah Anggota ({len(members_list)} Kab/Kota)")
                        st.caption(f"Median Total Pengeluaran F&B: **Rp {med_val:,.0f} / kapita / minggu**")
                        
                        # Render members as neat tags/badges
                        badge_html = " ".join([
                            f"<span style='display:inline-block; background:rgba(128,128,128,0.15); border-radius:6px; padding:3px 8px; margin:2px 3px; font-size:0.85rem;'>📍 {m}</span>"
                            for m in members_list
                        ])
                        st.markdown(badge_html, unsafe_allow_html=True)
                
                # Row 2: 3 Proportional Action Cards (Now What)
                if rec_entry and "now_what" in rec_entry:
                    st.markdown("##### 🎯 Rekomendasi Aksi Konkret (Now What)")
                    c_fmcg, c_food, c_gov = st.columns(3)
                    with c_fmcg:
                        with st.container(border=True):
                            st.markdown("##### 🛒 FMCG / Retail")
                            st.markdown(rec_entry['now_what']['fmcg'])
                    with c_food:
                        with st.container(border=True):
                            st.markdown("##### 🍔 Food Delivery & QSR")
                            st.markdown(rec_entry['now_what']['food_delivery'])
                    with c_gov:
                        with st.container(border=True):
                            st.markdown("##### 🏛️ Kebijakan Pemprov / Dinkes")
                            st.markdown(rec_entry['now_what']['pemprov'])

        # Final Tab: Matriks Komparatif Cross-Cluster
        with sub_recs_tabs[-1]:
            st.markdown("#### 📊 Matriks Strategis Lintas Segmen")
            st.markdown("Perbandingan komprehensif arah strategi bisnis dan kebijakan publik antar klaster:")
            
            matrix_data = []
            for cl_name in cluster_list:
                rec = None
                for k, v in recs_data.items():
                    if k in cl_name or cl_name in k:
                        rec = v
                        break
                cnt = (df_result["Nama_Klaster"] == cl_name).sum()
                med = df_result[df_result["Nama_Klaster"] == cl_name]["Total_Pengeluaran"].median() if "Total_Pengeluaran" in df_result.columns else 0
                matrix_data.append({
                    "Segmen Klaster": cl_name,
                    "Jumlah Wilayah": f"{cnt} Kab/Kota",
                    "Median Belanja": f"Rp {med:,.0f}/mgg",
                    "Strategi FMCG / Retail": rec['now_what']['fmcg'] if rec else "-",
                    "Strategi Foodservice": rec['now_what']['food_delivery'] if rec else "-",
                    "Fokus Kebijakan Pemda": rec['now_what']['pemprov'] if rec else "-"
                })
            st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

    with biz_sub2:
        st.subheader("📊 Analisis Kesenjangan Konsumsi Antar Wilayah (Disparity Gap)")
        st.markdown("Tabel berikut memeringkat komoditas dengan rasio kesenjangan pengeluaran terbesar antara klaster konsumsi tertinggi dan terendah:")
        
        fig_gap = plot_gap_analysis_interactive(gap_analysis_df, top_n=10)
        st.plotly_chart(fig_gap, use_container_width=True, key="gap_analysis_chart")
        st.dataframe(gap_analysis_df, use_container_width=True)

    with biz_sub3:
        st.subheader("🔥 Profil Median Komparatif Antar Segmen")
        fig_top = plot_top_features_interactive(median_profile_named, top_n=10)
        st.plotly_chart(fig_top, use_container_width=True, key="top_commodities_chart")
        
        fig_heat = plot_cluster_profile_heatmap_interactive(median_profile_named)
        st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_profile_chart")

    with biz_sub4:
        st.subheader("📦 Distribusi Pengeluaran Fitur Pilihan (Boxplot)")
        if selected_box_features:
            fig_box_custom = plot_boxplot_distribution(
                df_features, 
                selected_box_features, 
                title="Distribusi Pengeluaran untuk Fitur Pilihan"
            )
            st.pyplot(fig_box_custom)
        else:
            st.warning("Silakan pilih setidaknya satu fitur di sidebar untuk melihat boxplot.")

# ----------------------------------------------------
# TAB 5: TABEL DATA & EKSPOR
# ----------------------------------------------------
with tab_export:
    st.subheader("📥 Data Lengkap Hasil Segmentasi")
    
    # Filter by cluster
    selected_filter_cluster = st.multiselect(
        "Filter Tampilan Berdasarkan Klaster:",
        options=sorted(df_result["Nama_Klaster"].unique()),
        default=sorted(df_result["Nama_Klaster"].unique())
    )
    
    df_filtered = df_result[df_result["Nama_Klaster"].isin(selected_filter_cluster)]
    
    # Reorder columns nicely
    display_cols = ["Kabupaten/Kota", "Nama_Klaster", "Total_Pengeluaran"] + [c for c in feature_names if c in df_filtered.columns]
    df_filtered_display = df_filtered[display_cols].sort_values(by=["Nama_Klaster", "Total_Pengeluaran"], ascending=[True, False]).reset_index(drop=True)
    
    st.dataframe(df_filtered_display, use_container_width=True, height=500)
    
    csv_bytes = df_filtered_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Unduh Data Hasil Klastering (.CSV)",
        data=csv_bytes,
        file_name=f"hasil_klaster_konsumsi_jatim_k{n_clusters}.csv",
        mime="text/csv"
    )
