# --- Load dan Preprocessing Data (Simplified Version) ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
@st.cache_data
def load_and_preprocess():
    """Memuat, membersihkan, dan memproses data pengeluaran."""
    try:
        file_path = "Rata-rata Pengeluaran Perkapita Seminggu  Menurut Kelompok Makanan Minuman Jadi di Jawa Timur, 2024.xlsx"
        df = pd.read_excel(file_path, sheet_name="Sheet1")
    except FileNotFoundError:
        st.error(f"File Excel '{file_path}' tidak ditemukan.")
        return None, None, None, None, None, None

    # Proses data sesuai dengan kode asli
    df_original_raw = df.copy()
    nama_daerah = df["Kabupaten/Kota"]
    
    # Handle "Minuman keras" column specifically
    df["Minuman keras"] = df["Minuman keras"].replace("-", np.nan)
    df["Minuman keras"] = pd.to_numeric(df["Minuman keras"])
    df["Minuman keras"] = df["Minuman keras"].fillna(df["Minuman keras"].mean())
    
    # Drop non-numeric columns and prepare numeric data
    data_numerik = df.drop(columns=["Kabupaten/Kota"])
    df_for_viz = data_numerik.copy()  # For visualization before scaling
    
    # Drop "Minuman keras" column as in original code
    data_numerik = data_numerik.drop(columns=['Minuman keras'])
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data_numerik), 
                           columns=data_numerik.columns, 
                           index=nama_daerah)
    
    # Prepare original processed data (without scaling)
    df_original_processed = df[['Kabupaten/Kota']].copy()
    df_original_processed = pd.concat([df_original_processed, data_numerik], axis=1)
    
    return df_original_raw, df_original_processed, df_scaled, nama_daerah, data_numerik.columns.tolist(), df_for_viz

# --- Memuat Data ---
result = load_and_preprocess()
if result:
    df_raw, df_original, df_scaled, nama_daerah, all_features, df_for_viz = result
else:
    st.error("Gagal memuat atau memproses data. Aplikasi dihentikan.")
    st.stop()
# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Coat_of_arms_of_East_Java.svg/1200px-Coat_of_arms_of_East_Java.svg.png", width=80)
    st.title("⚙️ Kontrol Analisis")
    st.markdown("---")
    st.header("⚙️ Pengaturan Klaster")
    n_clusters = st.slider("Jumlah klaster:", 2, 10, 3, help="Pilih jumlah klaster berdasarkan dendogram dan silhouette score atau sesuai keinginan Anda.")

    st.header("🏷️ Penamaan Klaster")
    cluster_names = {}
    with st.expander("Edit Nama Klaster"):
        for i in range(n_clusters):
            cluster_names[i] = st.text_input(f"Klaster {i}", value=f"Klaster {i}", key=f"cluster_{i}")

    st.header("📊 Pengaturan Visualisasi")
    default_features_req = [
        "Nasi campur/rames", "Mie bakso, mie rebus, mie goreng",
        "Ayam/daging matang (ayam goreng, rendang, dsb)",
        "Minuman jadi (kopi, kopi susu, teh, susu coklat, dsb)", "Mie instan"
    ]
    default_features = [f for f in default_features_req if f in all_features]
    selected_features = st.multiselect(
        "Fitur default untuk boxplot:",
        options=all_features,
        default=default_features
    )
    st.markdown("---")
    st.info("Aplikasi ini memvisualisasikan analisis klastering.")

# --- Fungsi Pembantu & Klastering ---
def map_cluster_names(labels):
    return [cluster_names.get(label, f"Klaster {label}") for label in labels]

model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = model.fit_predict(df_scaled)
df = df_original.copy()
df['Cluster'] = cluster_labels
df['Nama_Klaster'] = map_cluster_names(cluster_labels)

# --- Judul Utama ---
st.title("📊 Platform Analisis Klaster: Pola Pengeluaran Makanan & Minuman di Jawa Timur")

# --- Tab Utama  ---
# Nama-nama tab 
about_project_tab, data_exploration_tab, clustering_results_tab, cluster_profile_tab, data_export_tab = st.tabs([
    "ℹ️ Tentang Proyek", 
    "1. Explorasi & Preprocessing",
    "2. Hasil & Gambaran Klaster",
    "3. Profil & Interpretasi",
    "4. Data & Ekspor"
])

# ==================================
# === TAB : Tentang Proyek ===
# ==================================
with about_project_tab:
    st.header("Selamat Datang di Platform Analisis Klaster")
    st.markdown("""
    <div class="project-info">
    <h4>Informasi Proyek</h4>
    <p>Dasbor ini menyajikan hasil analisis klastering untuk mengidentifikasi pola pengeluaran per kapita mingguan masyarakat 
       di berbagai Kabupaten/Kota di Provinsi Jawa Timur terhadap kelompok makanan dan minuman jadi menggunakan.</p>
    
    **Tujuan Utama:**
    <ul>
        <li>Mengelompokkan wilayah di Jawa Timur berdasarkan pola pengeluaran makanan dan minuman jadi menggunakan metode hierarchical clustering. .</li>
        <li>Menyajikan visualisasi dan interpretasi hasil clustering berbasis data pengeluaran, sehingga informasi yang diperoleh dapat dengan mudah dipahami 
            dan digunakan sebagai dasar pengambilan keputusan oleh berbagai pihak terkait.</li>
    </ul>
    
    **Data & Metodologi:**
    <ul>
        <li>Sumber Data: Badan Pusat Statistik (BPS) (<em>Melalui Website bps.go.id</em>).</li>
        <li>Metode Klastering: <em>Agglomerative Hierarchical Clustering</em> dengan metode Ward.</li>
        <li>Validasi & Visualisasi: Dibantu dengan <em>Silhouette Score</em> dan <em>Dendrogram</em> untuk menentukan jumlah Klaster, 
            dan <em>Principal Component Analysis (PCA) </em>untuk visualisasi klaster dalam plot 2D.</li>
    </ul>
    
    <h4>Panduan Penggunaan Platform:</h4>
    <ul>
        <li><b>Panel Kontrol (Sidebar Kiri):</b> Gunakan untuk mengatur jumlah klaster yang diinginkan, memberi nama deskriptif pada setiap klaster, 
               dan memilih fitur-fitur awal yang ingin dilihat distribusinya pada analisis boxplot.</li>
        <li><b>Navigasi Tab Utama (Di Atas):</b>
            <ul>
                <li><b>Explorasi & Preprocessing:</b> Menampilkan data awal, visualisasi sebelum transformasi, dan data setelah proses standardisasi.</li>
                <li><b>Hasil & Gambaran Klaster:</b> Menyajikan alat bantu penentuan jumlah klaster (Dendrogram & Silhouette Score), serta visualisasi umum hasil klaster seperti distribusi anggota dan peta PCA.</li>
                <li><b>Profil Detail & Interpretasi:</b> Memungkinkan Anda untuk mendalami karakteristik spesifik setiap klaster melalui heatmap, fitur dominan, dan perbandingan distribusi fitur.</li>
                <li><b>Data & Ekspor:</b> Menampilkan tabel data lengkap hasil klastering dan menyediakan opsi untuk mengunduh data dalam format CSV.</li>
            </ul>
        </li>
    </ul>
    <p>Semoga dasbor ini memberikan <i>insight</i> yang bermanfaat!</p>
    </div>
    """, unsafe_allow_html=True)

# ==================================
# === TAB 1 (Sekarang menjadi data_exploration_tab): Explorasi & Preprocessing ===
# ==================================
with data_exploration_tab: # Nama variabel tab diubah
    subtab1_tinjauan, subtab1_pra_transform, subtab1_pasca_transform = st.tabs([
        "Tinjauan Data Awal",
        "Eksplorasi Data Awal",
        "Boxplot Setelah Standardisasi"
    ])

    with subtab1_tinjauan:
        st.subheader("1. Tinjauan Data Awal")
        if df_raw is not None:
            st.dataframe(df_raw.head())
            with st.expander("Tampilkan Info Data Mentah"):
                st.write("Jumlah `NaN` sebelum proses:")
                st.dataframe(df_raw.isna().sum().to_frame(name='Jumlah NaN'))
                st.info("Catatan: '-' diganti NaN. 'Minuman Keras' diimputasi mean. Namun, akhirnya 'Minuman Keras' dihapus.")
        else:
            st.warning("Data mentah tidak dapat ditampilkan.")

    with subtab1_pra_transform:
        st.subheader("2. Visualisasi Data Sebelum Transformasi")
        col_tab1_1, col_tab1_2 = st.columns(2)
        with col_tab1_1:
            st.markdown("##### Distribusi Fitur (Sebelum Scaling)")
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            sns.boxplot(data=df_for_viz, orient="h", linewidth=0.7, fliersize=2, ax=ax1)
            ax1.set_title("Boxplot Sebelum Standardisasi")
            st.pyplot(fig1)

        with col_tab1_2:
            st.markdown("##### Korelasi Antar Fitur")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_for_viz.corr(), cmap='coolwarm', annot=False, ax=ax2)
            ax2.set_title("Heatmap Korelasi Antar Fitur")
            st.pyplot(fig2)

    with subtab1_pasca_transform:
        st.subheader("3. Data Setelah Standardisasi")
        st.info(
            "**Catatan Penting:** Variabel 'Minuman keras' **telah dihapus** sebelum tahap standardisasi ini. "
            "Alasannya adalah karena data pada variabel ini bernilai sangat rendah, dan sebelumnya sebagian datanya bernilai '-' pada awalnya."
            " selain itu korelasi dari fitur ini juga sangat rendah."
        )
        st.markdown("##### Distribusi Fitur (Setelah Scaling)")
        fig3, ax3 = plt.subplots(figsize=(18, 10))
        sns.boxplot(data=df_scaled, orient="h", linewidth=0.7, fliersize=2, ax=ax3)
        ax3.set_title("Boxplot Setelah Standardisasi")
        st.pyplot(fig3)


# ==================================
# === TAB 2 (Sekarang menjadi clustering_results_tab): Hasil & Gambaran Klaster ===
# ==================================
with clustering_results_tab: 
    subtab2_penentuan, subtab2_gambaran = st.tabs(["Penentuan Jumlah Kluster", "Gambaran Hasil Klaster"])

    with subtab2_penentuan:
        st.subheader("4. Penentuan Jumlah Klaster Optimal (K)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Analisis Dendrogram")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            dendrogram(linkage(df_scaled, method='ward'), ax=ax4, labels=df_scaled.index.tolist(), leaf_rotation=90, leaf_font_size=8)
            ax4.set_title("Dendrogram Hierarchical Clustering")
            st.pyplot(fig4)

        with col2:
            st.markdown("##### Analisis Silhouette Score")
            cluster_range = range(2, 11)
            silhouette_scores = []
            with st.spinner("Menghitung Silhouette Scores..."):
                for n_calc in cluster_range:
                    model_sil = AgglomerativeClustering(n_clusters=n_calc, linkage='ward')
                    labels_sil = model_sil.fit_predict(df_scaled)
                    silhouette_scores.append(silhouette_score(df_scaled, labels_sil))

            fig5, ax5 = plt.subplots(figsize=(10, 6))
            ax5.plot(cluster_range, silhouette_scores, marker='o')
            ax5.axvline(n_clusters, color='r', linestyle='--', label=f'K = {n_clusters} (Dipilih)')
            ax5.set_title("Silhouette Score (Optimalitas K)"); ax5.set_xlabel("Jumlah Klaster"); ax5.set_ylabel("Silhouette Score")
            ax5.grid(True); ax5.legend()
            st.pyplot(fig5)

    with subtab2_gambaran:
        st.subheader(f"5. Gambaran Umum hasil Klaster (K = {n_clusters})")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### Distribusi Anggota Klaster")
            fig6, ax6 = plt.subplots(figsize=(8, 6))
            cluster_counts_named = df['Nama_Klaster'].value_counts().sort_index()
            sns.barplot(x=cluster_counts_named.index, y=cluster_counts_named.values, palette='viridis', ax=ax6)
            plt.xticks(rotation=45)
            for p in ax6.patches:
                ax6.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            ax6.set_title('Jumlah Kabupaten/Kota per Klaster')
            st.pyplot(fig6)

        with col4:
            st.markdown("##### Visualisasi Klaster 2D (PCA)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_scaled)
            df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
            df_pca['Nama_Klaster'] = df['Nama_Klaster']
            df_pca['Kabupaten/Kota'] = df_scaled.index

            fig10, ax10 = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='PC1', y='PC2', hue='Nama_Klaster', data=df_pca, palette='Set2', s=80, ax=ax10, alpha=0.9)
            ax10.set_title(f'Peta Klaster PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% Varians)')
            ax10.legend(title='Klaster', bbox_to_anchor=(1.05, 1), loc='upper left')

            show_labels = st.checkbox("Tampilkan Label Wilayah pada Plot PCA", value=False, key="pca_labels_subtab")

            if show_labels:
                for line in range(0, df_pca.shape[0]):
                    ax10.annotate(
                        df_pca['Kabupaten/Kota'].iloc[line],
                        (df_pca['PC1'].iloc[line], df_pca['PC2'].iloc[line]),
                        textcoords="offset points", xytext=(0, 5), ha='center',
                        fontsize=7, alpha=0.7
                    )
                st.caption("Label wilayah ditampilkan di setiap titik.")
            st.pyplot(fig10)


# ==================================
# === TAB 3 Profil Detail & Interpretasi ===
# ==================================
with cluster_profile_tab: # Nama variabel tab diubah
    # st.header("Analisis Karakteristik Klaster")
    subtab3_1, subtab3_2, subtab3_3 = st.tabs(["Profil Klaster Lengkap", "Perbandingan Fitur Dominan", "Distribusi Fitur (Boxplot)"])

    numeric_cols = df.select_dtypes(include=np.number).columns.drop('Cluster', errors='ignore')
    cluster_profile = df.groupby('Nama_Klaster')[numeric_cols].median()

    with subtab3_1:
        st.subheader("Profil Perbandingan (Heatmap)")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_profile.T, annot=True, cmap='YlGnBu', fmt=".1f", ax=ax)
        ax.set_title("Heatmap Perbandingan Median per Fitur")
        st.pyplot(fig)

        st.markdown("---")

        st.subheader("Profil per Klaster (Top 10 Fitur & Wilayah)")
        try:
            top_features_per_cluster = {}
            for cluster in sorted(df['Nama_Klaster'].unique()):
                cluster_data = df[df['Nama_Klaster'] == cluster]
                median_values = cluster_data[all_features].median()
                top_features_per_cluster[cluster] = median_values.sort_values(ascending=False).head(10)

            cols = st.columns(min(n_clusters, 3))
            for idx, cluster in enumerate(top_features_per_cluster.keys()):
                with cols[idx % len(cols)]:
                    st.markdown(f"**{cluster}**")
                    fig_sub, ax_sub = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=top_features_per_cluster[cluster].values, y=top_features_per_cluster[cluster].index, palette='viridis', ax=ax_sub)
                    ax_sub.set_title(f'Top 10 Fitur'); ax_sub.set_xlabel("Median"); ax_sub.set_ylabel("")
                    plt.tight_layout(); st.pyplot(fig_sub)
                    with st.expander(f"Lihat Wilayah ({len(df[df['Nama_Klaster'] == cluster])})"):
                         st.write(", ".join(df[df['Nama_Klaster'] == cluster]['Kabupaten/Kota'].values))
        except Exception as e:
            st.error(f"Error: {str(e)}")

    with subtab3_2:
        st.subheader("10 Fitur Pengeluaran Dominan (Secara Keseluruhan)")
        try:
            top_features = cluster_profile.max(axis=0).sort_values(ascending=False).head(10).index
            fig, ax = plt.subplots(figsize=(12, 6))
            cluster_profile[top_features].T.plot(kind='bar', ax=ax, width=0.8)
            plt.title("Fitur dengan Pengeluaran Tertinggi di Seluruh Klaster", fontsize=12)
            plt.xlabel("Jenis Pengeluaran"); plt.ylabel("Nilai Median Pengeluaran")
            plt.xticks(rotation=45, ha='right'); plt.legend(title="Klaster", bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
            st.pyplot(fig)
            st.markdown("""<div style="background-color:#f8f9fa; padding:12px; border-radius:8px; margin-top:12px">
            <b>Interpretasi:</b> Menunjukkan perbandingan 10 fitur dengan median tertinggi.</div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    with subtab3_3:
        st.subheader("Perbandingan Distribusi Fitur Pilihan")
        try:
            col_tab3_3_1, col_tab3_3_2 = st.columns([3, 1])
            with col_tab3_3_1:
                selected_box_features = st.multiselect(
                    "Pilih fitur untuk dibandingkan:",
                    options=all_features,
                    default=selected_features,
                    key="boxplot_features_merged"
                )
            with col_tab3_3_2:
                max_features = st.slider(
                    "Maksimal fitur per baris:", 1, 5, 3,
                    key="slider_box_merged"
                )

            if selected_box_features:
                n_rows = int(np.ceil(len(selected_box_features) / max_features))
                for i in range(n_rows):
                    features_slice = selected_box_features[i*max_features : (i+1)*max_features]
                    fig_box, axes_box = plt.subplots(1, len(features_slice), figsize=(5*len(features_slice), 4), squeeze=False)
                    axes_box = axes_box.flatten()
                    for j, feature in enumerate(features_slice):
                        sns.boxplot(data=df, x='Nama_Klaster', y=feature, palette='Set2', ax=axes_box[j], order=sorted(df['Nama_Klaster'].unique()))
                        axes_box[j].set_title(feature[:25] + "..." if len(feature) > 25 else feature)
                        axes_box[j].tick_params(axis='x', rotation=45); axes_box[j].set_xlabel("")
                    plt.tight_layout(); st.pyplot(fig_box)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==================================
# === TAB 4 : Data & Ekspor ===
# ==================================
with data_export_tab: # Nama variabel tab diubah
    st.header("Data Hasil Klastering & Ekspor")
    df_display = df[['Kabupaten/Kota', 'Nama_Klaster', 'Cluster'] + all_features].sort_values(['Nama_Klaster', 'Kabupaten/Kota']).reset_index(drop=True)
    st.dataframe(df_display, use_container_width=True, height=600)

    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df_display)
    st.download_button("📥 Unduh Data sebagai CSV", csv, f'hasil_klaster_jatim_{n_clusters}_klaster.csv', 'text/csv')