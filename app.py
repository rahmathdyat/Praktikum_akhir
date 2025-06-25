import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Predikdi Data Jumalah Kunjungan Wisatawan Mancanegara ke Indonesia Tahun (2008-2025)",
    page_icon="📊",
    layout="wide"
)

# --- Judul Aplikasi Utama ---
st.title("💡 Predikdi Data Jumalah Kunjungan Wisatawan Mancanegara ke Indonesia Tahun (2008-2025)")
st.markdown("---")
st.write("Aplikasi ini memungkinkan Anda untuk mengunggah dataset, melakukan analisis data eksploratif, membangun model Machine Learning (Supervised dan Unsupervised), serta memvisualisasikan hasilnya dengan mudah.")

# --- Sidebar (Menu Navigasi) ---
st.sidebar.header("📋 Menu Navigasi")
menu_selection = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda & Unggah Dataset", "🔍 Analisis Data Eksploratif (EDA)", "🤖 Pemodelan Machine Learning", "📈 Visualisasi Hasil Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Panduan Singkat")
st.sidebar.markdown("""
1.  *Unggah Dataset*: Mulai dengan mengunggah file CSV Anda.
2.  *Analisis Data*: Pahami karakteristik data Anda.
3.  *Pemodelan*: Pilih algoritma, tentukan variabel, dan latih model.
4.  *Visualisasi*: Lihat performa model dan hasil klastering.
""")

st.sidebar.markdown("### 🔧 Algoritma Didukung")
st.sidebar.markdown("""
-   *Supervised Learning (Klasifikasi & Regresi)*:
    -   Regresi Linier
    -   Regresi Logistik
    -   Naive Bayes
    -   Support Vector Machine (SVM)
    -   K-Nearest Neighbors (KNN)
    -   Decision Tree
-   *Unsupervised Learning (Klastering)*:
    -   K-Means
""")

# --- Inisialisasi Session State ---
if 'data_original' not in st.session_state:
    st.session_state.data_original = None
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state: # 'classification', 'regression', 'clustering'
    st.session_state.model_type = None
if 'algorithm_name' not in st.session_state:
    st.session_state.algorithm_name = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None


# --- Fungsi Pra-pemrosesan Data ---
def preprocess_data(df):
    """
    Menangani missing values (imputasi mean/mode) dan encoding variabel kategorikal.
    Mengembalikan DataFrame yang telah diproses dan dictionary label encoders.
    """
    df_processed = df.copy()
    label_encoders = {}

    for col in df_processed.columns:
        # Imputasi Missing Values
        if df_processed[col].isnull().any():
            if df_processed[col].dtype == 'object' or df_processed[col].nunique() < 20: # Heuristik untuk kategorikal
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Missing'
                df_processed[col].fillna(mode_val, inplace=True)
                st.info(f"Missing values di kolom '{col}' (kategorikal) diisi dengan modus: '{mode_val}'.")
            else:
                mean_val = df_processed[col].mean()
                df_processed[col].fillna(mean_val, inplace=True)
                st.info(f"Missing values di kolom '{col}' (numerik) diisi dengan rata-rata: {mean_val:.2f}.")

        # Encoding Kategorikal
        if df_processed[col].dtype == 'object':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            st.info(f"Kolom '{col}' telah di-encode menjadi numerik.")
            
    return df_processed, label_encoders

# Fungsi untuk menentukan tipe problem
def determine_problem_type(target_series):
    """
    Menentukan apakah problem adalah klasifikasi atau regresi berdasarkan kolom target.
    """
    # Jika target adalah tipe objek (string/kategorikal asli) atau memiliki kurang dari 20 nilai unik
    # dianggap klasifikasi. Ambang batas 20 bisa disesuaikan.
    if target_series.dtype == 'object' or target_series.nunique() <= 20:
        return 'classification'
    else:
        return 'regression'

# --- 1. Beranda & Unggah Dataset ---
if menu_selection == "🏠 Beranda & Unggah Dataset":
    st.header("Selamat Datang! Unggah Dataset Anda 📂")
    st.markdown("Silakan unggah file CSV Anda untuk memulai analisis data mining.")

    uploaded_file = st.file_uploader(
        "Pilih file CSV:",
        type=['csv'],
        help="Ukuran file maksimal disarankan sekitar 200MB untuk performa optimal."
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data_original = data.copy() # Simpan salinan asli

            st.success("✅ Dataset berhasil diunggah!")

            st.subheader("📊 Informasi Umum Dataset")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jumlah Baris", st.session_state.data_original.shape[0])
            with col2:
                st.metric("Jumlah Kolom", st.session_state.data_original.shape[1])
            
            st.subheader("👀 Preview Data (5 Baris Pertama)")
            st.dataframe(st.session_state.data_original.head())

            st.subheader("🔍 Tipe Data per Kolom")
            st.dataframe(st.session_state.data_original.dtypes.to_frame('Tipe Data'))

            st.subheader("📈 Statistik Deskriptif Kolom Numerik")
            st.dataframe(st.session_state.data_original.describe())

            # Reset model dan hasil saat dataset baru diunggah
            st.session_state.data_processed = None
            st.session_state.model = None
            st.session_state.predictions = None
            st.session_state.model_type = None
            st.session_state.algorithm_name = None
            st.session_state.target_column = None
            st.session_state.feature_columns = None
            st.session_state.label_encoders = {}
            st.session_state.scaler = None


        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}. Pastikan ini adalah file CSV yang valid.")
    else:
        st.info("⬆️ Silakan unggah file CSV untuk memulai analisis.")

# --- 2. Analisis Data Eksploratif (EDA) ---
elif menu_selection == "🔍 Analisis Data Eksploratif (EDA)":
    st.header("Eksplorasi Dataset Anda 📊")
    st.markdown("Pahami karakteristik dataset Anda melalui berbagai analisis eksploratif.")

    if st.session_state.data_original is not None:
        data = st.session_state.data_original

        analysis_option = st.selectbox(
            "Pilih Jenis Analisis:",
            ["Ringkasan Statistik", "Distribusi Data", "Korelasi Antar Fitur", "Missing Values", "Deteksi Outlier"]
        )

        st.markdown("---")

        if analysis_option == "Ringkasan Statistik":
            st.subheader("📝 Ringkasan Statistik Dataset")
            st.write(data.describe(include='all')) # include='all' untuk juga melihat statistik kategorikal

        elif analysis_option == "Distribusi Data":
            st.subheader("📊 Visualisasi Distribusi Data")
            
            all_cols = data.columns.tolist()
            if not all_cols:
                st.warning("Dataset kosong atau tidak memiliki kolom.")
            else:
                selected_col_dist = st.selectbox("Pilih Kolom untuk Visualisasi Distribusi:", all_cols)
                
                if data[selected_col_dist].dtype in ['int64', 'float64']:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(data[selected_col_dist].dropna(), kde=True, ax=ax)
                        ax.set_title(f'Histogram Distribusi {selected_col_dist}')
                        ax.set_xlabel(selected_col_dist)
                        ax.set_ylabel('Frekuensi')
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.boxplot(y=data[selected_col_dist].dropna(), ax=ax)
                        ax.set_title(f'Box Plot {selected_col_dist}')
                        ax.set_ylabel(selected_col_dist)
                        st.pyplot(fig)
                else:
                    st.info(f"Kolom '{selected_col_dist}' adalah kategorikal. Menampilkan hitungan nilai unik.")
                    value_counts = data[selected_col_dist].value_counts().reset_index()
                    value_counts.columns = [selected_col_dist, 'Count']
                    fig = px.bar(value_counts, x=selected_col_dist, y='Count', 
                                 title=f'Distribusi Kategori {selected_col_dist}')
                    st.plotly_chart(fig, use_container_width=True)

        elif analysis_option == "Korelasi Antar Fitur":
            st.subheader("🔗 Matriks Korelasi Kolom Numerik")
            numeric_data = data.select_dtypes(include=np.number)

            if not numeric_data.empty and numeric_data.shape[1] > 1:
                corr_matrix = numeric_data.corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                ax.set_title('Matriks Korelasi')
                st.pyplot(fig)
            else:
                st.info("Tidak ada cukup kolom numerik untuk membuat matriks korelasi.")

        elif analysis_option == "Missing Values":
            st.subheader("❓ Analisis Missing Values")
            missing_info = data.isnull().sum()
            missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
            
            if not missing_info.empty:
                missing_df = pd.DataFrame({
                    'Kolom': missing_info.index,
                    'Jumlah Missing': missing_info.values,
                    'Persentase Missing (%)': (missing_info.values / len(data)) * 100
                })
                st.dataframe(missing_df.style.background_gradient(cmap='YlOrRd', subset=['Persentase Missing (%)']))
                
                st.warning("⚠️ Perhatikan kolom dengan missing values. Mereka akan diimputasi atau dihapus selama pemodelan.")
            else:
                st.success("🎉 Tidak ada missing values dalam dataset ini!")

        elif analysis_option == "Deteksi Outlier":
            st.subheader("🎯 Deteksi Outlier (dengan Box Plot)")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols:
                selected_outlier_col = st.selectbox("Pilih Kolom Numerik untuk Deteksi Outlier:", numeric_cols)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.boxplot(y=data[selected_outlier_col].dropna(), ax=ax)
                ax.set_title(f'Box Plot untuk {selected_outlier_col}')
                ax.set_ylabel(selected_outlier_col)
                st.pyplot(fig)

                Q1 = data[selected_outlier_col].quantile(0.25)
                Q3 = data[selected_outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_count = data[(data[selected_outlier_col] < lower_bound) | (data[selected_outlier_col] > upper_bound)].shape[0]
                st.info(f"Ditemukan *{outliers_count}* outlier pada kolom '{selected_outlier_col}' berdasarkan metode IQR.")
                st.write(f"Batas Bawah: {lower_bound:.2f}")
                st.write(f"Batas Atas: {upper_bound:.2f}")

            else:
                st.info("Tidak ada kolom numerik untuk deteksi outlier.")
    else:
        st.warning("Silakan unggah dataset terlebih dahulu di menu 'Beranda & Unggah Dataset'.")

# --- 3. Pemodelan Machine Learning ---
elif menu_selection == "🤖 Pemodelan Machine Learning":
    st.header("Pilih & Latih Model Machine Learning 🚀")
    st.markdown("Pilih algoritma, tentukan fitur dan target (untuk Supervised Learning), lalu latih model.")

    if st.session_state.data_original is not None:
        st.subheader("⚙️ Pra-pemrosesan Data Otomatis")
        # Lakukan pra-pemrosesan setiap kali halaman pemodelan diakses
        st.session_state.data_processed, st.session_state.label_encoders = preprocess_data(st.session_state.data_original)
        
        if st.session_state.data_processed.empty:
            st.error("Dataset kosong setelah pra-pemrosesan. Tidak dapat melanjutkan pemodelan.")
        else:
            st.success("✅ Dataset telah berhasil dipra-proses (missing values & encoding).")
            st.write("Preview Data Setelah Pra-pemrosesan:")
            st.dataframe(st.session_state.data_processed.head())

            st.markdown("---")

            st.subheader("Pilihan Algoritma")
            algorithm_options = [
                "Regresi Linier", "Regresi Logistik", "Naive Bayes",
                "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)",
                "Decision Tree", "K-Means Clustering"
            ]
            st.session_state.algorithm_name = st.selectbox(
                "Pilih Algoritma Machine Learning:",
                algorithm_options,
                index=algorithm_options.index(st.session_state.algorithm_name) if st.session_state.algorithm_name else 0
            )

            # --- Supervised Learning Setup ---
            if st.session_state.algorithm_name != "K-Means Clustering":
                st.subheader("🎯 Pengaturan untuk Supervised Learning")

                available_cols = st.session_state.data_processed.columns.tolist()
                
                # Menjaga pilihan kolom target jika sudah ada
                default_target_idx = 0
                if st.session_state.target_column and st.session_state.target_column in available_cols:
                    default_target_idx = available_cols.index(st.session_state.target_column)

                st.session_state.target_column = st.selectbox(
                    "Pilih kolom target (variabel dependen):",
                    available_cols,
                    index=default_target_idx
                )

                # Menentukan tipe problem berdasarkan kolom target dari original_df
                # Ini penting karena data_processed sudah numerik semua
                st.session_state.model_type = determine_problem_type(st.session_state.data_original[st.session_state.target_column])
                st.info(f"Tipe Masalah yang Terdeteksi: *{st.session_state.model_type.replace('classification', 'Klasifikasi').replace('regression', 'Regresi').upper()}*")

                st.session_state.feature_columns = [col for col in available_cols if col != st.session_state.target_column]

                if not st.session_state.feature_columns:
                    st.error("Tidak ada fitur yang tersedia setelah memilih kolom target. Pastikan dataset memiliki lebih dari satu kolom.")
                else:
                    st.write(f"Fitur yang akan digunakan ({len(st.session_state.feature_columns)}): *{', '.join(st.session_state.feature_columns)}*")

                    X = st.session_state.data_processed[st.session_state.feature_columns]
                    y = st.session_state.data_processed[st.session_state.target_column]

                    test_size = st.slider("Ukuran Data Testing (%)", 10, 50, 20) / 100
                    
                    # Cek ukuran data setelah split
                    min_samples = 2 # Minimal sampel untuk train_test_split
                    if len(X) < min_samples:
                        st.error(f"Ukuran dataset terlalu kecil ({len(X)} sampel) untuk split. Minimal {min_samples} sampel diperlukan.")
                    elif len(X) * (1 - test_size) < 1 or len(X) * test_size < 1:
                         st.error(f"Ukuran data training atau testing akan kosong dengan rasio {test_size*100}%. Sesuaikan rasio atau gunakan dataset yang lebih besar.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, 
                            stratify=y if st.session_state.model_type == 'classification' and y.nunique() > 1 else None
                        )

                        # Scaling fitur numerik (penting untuk SVM, KNN, Regresi Logistik)
                        st.session_state.scaler = StandardScaler()
                        X_train_scaled = st.session_state.scaler.fit_transform(X_train)
                        X_test_scaled = st.session_state.scaler.transform(X_test)
                        
                        # Simpan X_test asli (tidak discale) untuk visualisasi fitur importance
                        st.session_state.predictions = {
                            'y_test': y_test,
                            'X_test': X_test, # Original X_test for feature names
                            'X_train_scaled': X_train_scaled,
                            'X_test_scaled': X_test_scaled,
                            'y_train': y_train
                        }

                        st.markdown("---")
                        st.subheader("Konfigurasi & Latih Model")

                        model = None
                        if st.session_state.algorithm_name == "Regresi Linier":
                            if st.session_state.model_type == 'regression':
                                model = LinearRegression()
                            else:
                                st.warning("Regresi Linier hanya cocok untuk masalah Regresi. Pilih algoritma lain atau kolom target numerik.")
                        elif st.session_state.algorithm_name == "Regresi Logistik":
                            if st.session_state.model_type == 'classification':
                                model = LogisticRegression(max_iter=1000, random_state=42)
                            else:
                                st.warning("Regresi Logistik hanya cocok untuk masalah Klasifikasi. Pilih algoritma lain atau kolom target kategorikal.")
                        elif st.session_state.algorithm_name == "Naive Bayes":
                            if st.session_state.model_type == 'classification':
                                model = GaussianNB()
                            else:
                                st.warning("Naive Bayes umumnya cocok untuk masalah Klasifikasi. Pilih algoritma lain atau kolom target kategorikal.")
                        elif st.session_state.algorithm_name == "Support Vector Machine (SVM)":
                            if st.session_state.model_type == 'classification':
                                model = SVC(random_state=42)
                            else:
                                model = SVR()
                                st.info("SVM akan digunakan untuk Regresi (SVR).")
                        elif st.session_state.algorithm_name == "K-Nearest Neighbors (KNN)":
                            max_k = min(20, len(X_train) - 1)
                            if max_k < 1:
                                st.error("Ukuran data training terlalu kecil untuk K-NN.")
                                model = None
                            else:
                                n_neighbors = st.slider("Jumlah Tetangga (K) untuk KNN:", 1, max_k, min(5, max_k))
                                if st.session_state.model_type == 'classification':
                                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                                else:
                                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                        elif st.session_state.algorithm_name == "Decision Tree":
                            max_depth = st.slider("Kedalaman Maksimum Pohon Keputusan:", 1, 20, 5)
                            if st.session_state.model_type == 'classification':
                                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                            else:
                                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                        
                        if st.button("🚀 Latih Model"):
                            if model:
                                try:
                                    model.fit(X_train_scaled, y_train)
                                    y_pred = model.predict(X_test_scaled)
                                    
                                    st.session_state.model = model
                                    st.session_state.predictions['y_pred'] = y_pred
                                    st.success(f"Model {st.session_state.algorithm_name} berhasil dilatih!")

                                    # Tampilkan metrik evaluasi singkat
                                    st.subheader("✅ Metrik Evaluasi Singkat")
                                    if st.session_state.model_type == 'classification':
                                        accuracy = accuracy_score(y_test, y_pred)
                                        st.write(f"*Akurasi:* {accuracy:.4f}")
                                    elif st.session_state.model_type == 'regression':
                                        mae = mean_absolute_error(y_test, y_pred)
                                        mse = mean_squared_error(y_test, y_pred)
                                        r2 = r2_score(y_test, y_pred)
                                        st.write(f"*MAE:* {mae:.4f}")
                                        st.write(f"*MSE:* {mse:.4f}")
                                        st.write(f"*R² Score:* {r2:.4f}")
                                        
                                    st.info("Anda dapat melihat visualisasi lengkap di menu 'Visualisasi Hasil Model'.")
                                except Exception as e:
                                    st.error(f"❌ Terjadi kesalahan saat melatih model: {e}")
                            else:
                                st.warning("Silakan pilih algoritma yang sesuai dengan tipe masalah Anda.")

            # --- Unsupervised Learning Setup (K-Means) ---
            else: # K-Means Clustering
                st.subheader("🌟 Pengaturan untuk Unsupervised Learning (K-Means)")
                
                numeric_cols_for_clustering = st.session_state.data_processed.select_dtypes(include=np.number).columns.tolist()
                
                if not numeric_cols_for_clustering:
                    st.error("Tidak ada kolom numerik di dataset untuk K-Means. Pastikan dataset Anda memiliki data numerik.")
                else:
                    st.write("Fitur yang akan digunakan untuk clustering (semua kolom numerik yang diproses):")
                    st.write(numeric_cols_for_clustering)

                    n_clusters = st.slider("Jumlah Klaster (K):", 2, min(10, len(st.session_state.data_processed) -1), 3)
                    
                    if n_clusters < 2:
                        st.error("Jumlah klaster minimal 2.")
                    elif n_clusters > len(st.session_state.data_processed):
                        st.error("Jumlah klaster tidak boleh melebihi jumlah sampel data.")
                    else:
                        st.markdown("---")
                        if st.button("🚀 Jalankan K-Means Clustering"):
                            try:
                                X_clustering = st.session_state.data_processed[numeric_cols_for_clustering]
                                
                                # Scaling untuk K-Means
                                st.session_state.scaler = StandardScaler()
                                X_scaled_clustering = st.session_state.scaler.fit_transform(X_clustering)

                                model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                                clusters = model.fit_predict(X_scaled_clustering)

                                st.session_state.model = model
                                st.session_state.model_type = "clustering"
                                st.session_state.predictions = {
                                    'clusters': clusters,
                                    'X_original_for_clustering': X_clustering # Simpan X asli untuk visualisasi
                                }
                                st.success(f"Model K-Means dengan {n_clusters} klaster berhasil dilatih!")

                                st.subheader("✅ Hasil Klastering Singkat")
                                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                                st.write("*Distribusi Anggota Klaster:*")
                                st.dataframe(cluster_counts.to_frame(name='Jumlah Anggota'))
                                st.write(f"*Inertia (Sum of squared distances):* {model.inertia_:.2f}")

                                st.info("Anda dapat melihat visualisasi klaster di menu 'Visualisasi Hasil Model'.")

                            except Exception as e:
                                st.error(f"❌ Terjadi kesalahan saat menjalankan K-Means: {e}")
    else:
        st.warning("Silakan unggah dataset terlebih dahulu di menu 'Beranda & Unggah Dataset'.")

# --- 4. Visualisasi Hasil Model ---
elif menu_selection == "📈 Visualisasi Hasil Model":
    st.header("Visualisasi Hasil Model 📊")
    st.markdown("Lihat metrik performa dan visualisasi interaktif dari model yang telah Anda latih.")

    if st.session_state.model is not None and st.session_state.predictions is not None:
        
        # --- Visualisasi untuk Supervised Learning (Klasifikasi & Regresi) ---
        if st.session_state.model_type in ['classification', 'regression']:
            y_test = st.session_state.predictions['y_test']
            y_pred = st.session_state.predictions['y_pred']
            X_test_original = st.session_state.predictions['X_test']
            
            if st.session_state.model_type == 'classification':
                st.subheader("🎯 Metrik Evaluasi Klasifikasi")
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Akurasi", f"{accuracy:.4f}")
                col2.metric("Presisi", f"{precision:.4f}")
                col3.metric("Recall", f"{recall:.4f}")
                col4.metric("F1-Score", f"{f1:.4f}")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                # Menggunakan LabelEncoder untuk mendapatkan label asli jika ada
                target_col_name = st.session_state.target_column
                if target_col_name in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[target_col_name]
                    display_labels = le.inverse_transform(np.unique(y_test).astype(int))
                else:
                    display_labels = np.unique(y_test) # Gunakan nilai numerik jika tidak di-encode

                fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            xticklabels=display_labels, yticklabels=display_labels)
                ax_cm.set_title('Confusion Matrix')
                ax_cm.set_xlabel('Prediksi')
                ax_cm.set_ylabel('Aktual')
                st.pyplot(fig_cm)

                st.subheader("📋 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            elif st.session_state.model_type == 'regression':
                st.subheader("📈 Metrik Evaluasi Regresi")
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{mae:.4f}")
                col2.metric("MSE", f"{mse:.4f}")
                col3.metric("RMSE", f"{rmse:.4f}")
                col4.metric("R² Score", f"{r2:.4f}")

                st.subheader("Scatter Plot: Aktual vs. Prediksi")
                fig_reg_scatter = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Nilai Aktual', 'y': 'Nilai Prediksi'},
                    title='Perbandingan Nilai Aktual dan Prediksi'
                )
                fig_reg_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                                   y=[y_test.min(), y_test.max()],
                                                   mode='lines', name='Ideal', 
                                                   line=dict(color='red', dash='dash')))
                st.plotly_chart(fig_reg_scatter, use_container_width=True)

                st.subheader("Residual Plot")
                residuals = y_test - y_pred
                fig_res_scatter = px.scatter(
                    x=y_pred,
                    y=residuals,
                    labels={'x': 'Nilai Prediksi', 'y': 'Residuals'},
                    title='Residual Plot'
                )
                fig_res_scatter.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res_scatter, use_container_width=True)

            # Visualisasi Decision Tree (jika algoritma adalah Decision Tree)
            if st.session_state.algorithm_name == "Decision Tree":
                st.subheader("🌳 Visualisasi Pohon Keputusan")
                if st.session_state.model:
                    try:
                        # Re-scaling X_test back if needed for feature_names (though plot_tree takes names directly)
                        # Ensure X_test in session_state.predictions is the non-scaled one
                        feature_names_for_tree = X_test_original.columns.tolist()
                        
                        class_names_for_tree = None
                        if st.session_state.model_type == 'classification':
                            target_col_name = st.session_state.target_column
                            if target_col_name in st.session_state.label_encoders:
                                le = st.session_state.label_encoders[target_col_name]
                                class_names_for_tree = [str(x) for x in le.inverse_transform(st.session_state.model.classes_)]
                            else:
                                class_names_for_tree = [str(x) for x in st.session_state.model.classes_]

                        fig_tree, ax_tree = plt.subplots(figsize=(25, 15))
                        plot_tree(
                            st.session_state.model,
                            feature_names=feature_names_for_tree,
                            class_names=class_names_for_tree,
                            filled=True,
                            rounded=True,
                            fontsize=8,
                            ax=ax_tree
                        )
                        ax_tree.set_title('Visualisasi Pohon Keputusan', fontsize=20)
                        st.pyplot(fig_tree)
                    except Exception as e:
                        st.warning(f"⚠️ Gagal memvisualisasikan Decision Tree. Mungkin karena kedalaman pohon terlalu besar atau masalah lainnya: {e}")

            # Feature Importance (untuk model yang mendukung)
            if hasattr(st.session_state.model, 'feature_importances_') and st.session_state.feature_columns:
                st.subheader("📊 Pentingnya Fitur (Feature Importance)")
                importances = st.session_state.model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': st.session_state.feature_columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                fig_feat_imp = px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Pentingnya Fitur dalam Model',
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Plasma
                )
                fig_feat_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_feat_imp, use_container_width=True)

        # --- Visualisasi untuk Unsupervised Learning (K-Means) ---
        elif st.session_state.model_type == 'clustering':
            st.subheader("🎨 Visualisasi Hasil Klastering K-Means")
            clusters = st.session_state.predictions['clusters']
            X_original_for_clustering = st.session_state.predictions['X_original_for_clustering']
            
            # Tambahkan kolom cluster ke DataFrame asli untuk visualisasi mudah dengan Plotly
            df_clustered = X_original_for_clustering.copy()
            df_clustered['Cluster'] = clusters
            df_clustered['Cluster'] = df_clustered['Cluster'].astype(str) # Agar Plotly memperlakukan sebagai kategori

            numeric_cols_for_plot = X_original_for_clustering.columns.tolist()

            if len(numeric_cols_for_plot) >= 2:
                col_x = st.selectbox("Pilih Kolom X untuk Scatter Plot:", numeric_cols_for_plot, index=0)
                col_y = st.selectbox("Pilih Kolom Y untuk Scatter Plot:", numeric_cols_for_plot, index=1 if len(numeric_cols_for_plot) > 1 else 0)

                if col_x and col_y:
                    fig_cluster = px.scatter(
                        df_clustered,
                        x=col_x,
                        y=col_y,
                        color='Cluster',
                        title=f'Visualisasi Klaster K-Means ({col_x} vs {col_y})',
                        hover_data=df_clustered.columns,
                        template="plotly_white"
                    )
                    
                    # Tambahkan centroid ke plot
                    centroids_scaled = st.session_state.model.cluster_centers_
                    # Skalakan balik centroid ke skala fitur asli
                    centroids_original_scale = st.session_state.scaler.inverse_transform(centroids_scaled)
                    
                    centroids_df = pd.DataFrame(centroids_original_scale, columns=numeric_cols_for_plot)

                    fig_cluster.add_trace(go.Scatter(
                        x=centroids_df[col_x],
                        y=centroids_df[col_y],
                        mode='markers',
                        marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='DarkSlateGrey')),
                        name='Centroid Klaster'
                    ))
                    
                    st.plotly_chart(fig_cluster, use_container_width=True)

                st.subheader("Distribusi Anggota Klaster")
                cluster_distribution = df_clustered['Cluster'].value_counts().sort_index()
                fig_dist_cluster = px.bar(
                    cluster_distribution,
                    x=cluster_distribution.index,
                    y=cluster_distribution.values,
                    labels={'x':'Klaster', 'y':'Jumlah Anggota'},
                    title='Distribusi Jumlah Anggota per Klaster'
                )
                st.plotly_chart(fig_dist_cluster, use_container_width=True)
                
            else:
                st.warning("Tidak cukup kolom numerik untuk membuat visualisasi scatter plot klaster. Minimal 2 kolom diperlukan.")
                
        else:
            st.info("Pilih algoritma di menu 'Pemodelan' dan latih model untuk melihat visualisasi.")
    else:
        st.warning("Silakan latih model terlebih dahulu di menu 'Pemodelan' untuk melihat hasil visualisasi.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
    Aplikasi Data Mining Interaktif © 2023. Dibuat dengan Streamlit dan Scikit-learn.
    <br>
    Dirancang untuk mempermudah eksplorasi data dan pemodelan Machine Learning.
</div>
""", unsafe_allow_html=True)