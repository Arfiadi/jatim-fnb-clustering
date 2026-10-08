"""Script to generate reproducible Jupyter Notebook for East Java clustering analysis."""

from pathlib import Path
import nbformat as nbf


def generate_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# 🍜 Analisis Klaster Pola Pengeluaran Makanan & Minuman Jadi di Jawa Timur 2024\n"
            "**Mata Kuliah:** Pemodelan Statistik Terapan  \n"
            "**Framing Industri:** FMCG & Consumer Analytics Insights (Regional Distribution & Pricing Strategy)  \n"
            "**Metodologi:** Agglomerative Hierarchical Clustering (StandardScaler, Linkage Comparison, Cophenetic Correlation, Per-Sample Silhouette Analysis)"
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Add project root to sys.path\n"
            "root_path = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(root_path) not in sys.path:\n"
            "    sys.path.append(str(root_path))\n\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import plotly.express as px\n"
            "import plotly.graph_objects as go\n\n"
            "import src\n"
            "from src.clustering import (\n"
            "    calculate_cophenetic_correlation,\n"
            "    compare_linkage_methods,\n"
            "    fit_hierarchical_clustering,\n"
            "    compute_sample_silhouette,\n"
            "    get_cluster_profiles,\n"
            "    get_gap_analysis,\n"
            "    assign_business_cluster_names,\n"
            "    get_business_recommendations\n"
            ")\n"
            "from src.visualization import (\n"
            "    plot_dendrogram,\n"
            "    plot_silhouette_sample_analysis,\n"
            "    plot_pca_2d_interactive,\n"
            "    plot_cluster_distribution_interactive,\n"
            "    plot_cluster_profile_heatmap_interactive,\n"
            "    plot_top_features_interactive,\n"
            "    plot_gap_analysis_interactive,\n"
            "    plot_east_java_map_interactive\n"
            ")\n\n"
            "print('Libraries and custom modules successfully loaded.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Data Ingestion & Inspeksi Awal\n"
            "Dataset bersumber dari Badan Pusat Statistik (BPS) Jawa Timur yang mencatat rata-rata pengeluaran per kapita seminggu untuk komoditas makanan dan minuman jadi di seluruh 38 Kabupaten/Kota tahun 2024."
        ),
        nbf.v4.new_code_cell(
            "df_raw = src.load_raw_data()\n"
            "print(f'Dimensi Data Mentah: {df_raw.shape[0]} Wilayah x {df_raw.shape[1]} Kolom')\n"
            "df_raw.head()"
        ),
        nbf.v4.new_code_cell(
            "df_clean, regions, df_numeric = src.clean_data(df_raw)\n"
            "print(f'Fitur Numerik Teridentifikasi: {df_numeric.shape[1]} kolom')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Preprocessing & Justifikasi Metodologis\n"
            "### 2.1 Justifikasi Statistik Penghapusan Fitur `Minuman keras`\n"
            "Sebelum melakukan penghapusan fitur, kami melakukan pengujian empiris kuantitatif pada missing rate, rasio varians, dan korelasi."
        ),
        nbf.v4.new_code_cell(
            "drop_diag = src.analyze_dropped_feature(df_numeric, 'Minuman keras')\n"
            "print(drop_diag['justification'])"
        ),
        nbf.v4.new_code_cell(
            "# Drop fitur Minuman keras secara terjustifikasi\n"
            "df_features = df_numeric.drop(columns=['Minuman keras'], errors='ignore')\n"
            "feature_names = df_features.columns.tolist()\n"
            "print(f'Jumlah fitur final untuk pemodelan: {len(feature_names)} komoditas')"
        ),
        nbf.v4.new_markdown_cell(
            "### 2.2 Deteksi & Rationale Penanganan Outlier (IQR Method)\n"
            "Karena data ini merupakan data sensus populasi (38 Kabupaten/Kota se-Jawa Timur), pencilan tinggi pada Kota Surabaya dan Kota Malang mencerminkan fenomena ekonomi nyata. Outlier dipertahankan untuk mendeteksi segmen metropolitan."
        ),
        nbf.v4.new_code_cell(
            "outlier_res = src.detect_outliers_iqr(df_features, region_names=regions)\n"
            "print(outlier_res['methodology_rationale'])\n"
            "outlier_res['summary_table'].head(10)"
        ),
        nbf.v4.new_markdown_cell(
            "### 2.3 Standardisasi Fitur (Z-score Normalization)\n"
            "Menggunakan `StandardScaler` agar perbedaan skala nominal antar komoditas tidak mendominasi perhitungan jarak Euclidean."
        ),
        nbf.v4.new_code_cell(
            "df_scaled, scaler = src.scale_features(df_features, index_series=regions)\n"
            "df_scaled.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Eksplorasi Metodologis: Evaluasi 4 Metode Linkage\n"
            "Kami mengevaluasi 4 metode linkage (`Ward`, `Complete`, `Average`, `Single`) berdasarkan Silhouette Score (K=2 s/d 10) dan *Cophenetic Correlation Coefficient*."
        ),
        nbf.v4.new_code_cell(
            "scores_df, summary_eval = src.compare_linkage_methods(df_scaled.values)\n"
            "summary_eval"
        ),
        nbf.v4.new_code_cell(
            "fig_linkage = src.visualization.plot_linkage_comparison_interactive(scores_df, chosen_method='ward', chosen_k=3)\n"
            "fig_linkage.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Pemodelan Hierarchical Clustering (Ward, K=3)\n"
            "Metode Ward dipilih karena menghasilkan klaster yang homogen dengan ukuran klaster yang berimbang secara manajerial."
        ),
        nbf.v4.new_code_cell(
            "fig_dendro = plot_dendrogram(df_scaled.values, labels=regions.tolist(), method='ward')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "model, labels, sil_score = fit_hierarchical_clustering(df_scaled.values, n_clusters=3, linkage_method='ward')\n"
            "print(f'Silhouette Score Ward (K=3): {sil_score:.4f}')\n\n"
            "fig_sil_sample = plot_silhouette_sample_analysis(df_scaled.values, labels, n_clusters=3)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Visualisasi Hasil Klastering\n"
            "### 5.1 Proyeksi 2D PCA & Distribusi Anggota"
        ),
        nbf.v4.new_code_cell(
            "median_prof_raw, _ = get_cluster_profiles(df_features, labels)\n"
            "cluster_name_map = assign_business_cluster_names(median_prof_raw)\n\n"
            "named_labels = [cluster_name_map.get(f'Klaster {lbl}', f'Klaster {lbl}') for lbl in labels]\n"
            "df_result = df_features.copy()\n"
            "df_result['Kabupaten/Kota'] = regions.values\n"
            "df_result['Cluster_ID'] = labels\n"
            "df_result['Nama_Klaster'] = named_labels\n"
            "df_result['Total_Pengeluaran'] = df_features.sum(axis=1)\n\n"
            "fig_pca = plot_pca_2d_interactive(df_scaled, labels, regions, cluster_name_map, df_raw_features=df_features)\n"
            "fig_pca.show()"
        ),
        nbf.v4.new_code_cell(
            "fig_dist = plot_cluster_distribution_interactive(df_result, cluster_col='Nama_Klaster')\n"
            "fig_dist.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### 5.2 Peta Persebaran Spasial Jawa Timur"
        ),
        nbf.v4.new_code_cell(
            "fig_map = plot_east_java_map_interactive(df_result, total_spending_col='Total_Pengeluaran')\n"
            "fig_map.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Business Insights & FMCG Strategic Recommendations\n"
            "### 6.1 Analisis Disparitas Pengeluaran (Gap Analysis)"
        ),
        nbf.v4.new_code_cell(
            "median_prof_named, mean_prof_named = get_cluster_profiles(df_features, labels, label_names=cluster_name_map)\n"
            "gap_df = get_gap_analysis(median_prof_named)\n"
            "gap_df.head(10)"
        ),
        nbf.v4.new_code_cell(
            "fig_gap = plot_gap_analysis_interactive(gap_df, top_n=10)\n"
            "fig_gap.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### 6.2 Top Komoditas & Rekomendasi Aksi Industri"
        ),
        nbf.v4.new_code_cell(
            "fig_top = plot_top_features_interactive(median_prof_named, top_n=10)\n"
            "fig_top.show()"
        ),
        nbf.v4.new_code_cell(
            "recs = get_business_recommendations()\n"
            "for cluster_title, rec_body in recs.items():\n"
            "    print('=' * 80)\n"
            "    print(f'SEGMEN: {cluster_title}')\n"
            "    print(f'WHO: {rec_body[\"who\"]}')\n"
            "    print(f'WHAT: {rec_body[\"what\"]}')\n"
            "    print(f'SO WHAT: {rec_body[\"so_what\"]}')\n"
            "    print('NOW WHAT:')\n"
            "    print(f'  - FMCG Strategy: {rec_body[\"now_what\"][\"fmcg\"]}')\n"
            "    print(f'  - Food Delivery: {rec_body[\"now_what\"][\"food_delivery\"]}')\n"
            "    print(f'  - Kebijakan Pemda: {rec_body[\"now_what\"][\"pemprov\"]}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Ekspor Data Hasil Segmentasi"
        ),
        nbf.v4.new_code_cell(
            "output_path = Path('data/processed/hasil_klaster_konsumsi_jatim_2024.csv')\n"
            "output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "df_result.to_csv(output_path, index=False)\n"
            "print(f'Data hasil segmentasi berhasil diekspor ke: {output_path}')"
        )
    ]

    nb['cells'] = cells
    target_file = Path("notebooks/01_eda_and_clustering.ipynb")
    with open(target_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {target_file} successfully!")


if __name__ == "__main__":
    generate_notebook()
