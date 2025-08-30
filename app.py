import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
from streamlit_folium import st_folium
import folium

# --- KONFIGURasi HALAMAN ---
st.set_page_config(
    page_title="California Property Insights",
    page_icon="🏠",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT DATA & MODEL ---
# Menggunakan cache agar model dan data hanya dimuat sekali untuk performa lebih baik.
@st.cache_resource
def load_model():
    """Memuat model prediksi yang sudah dilatih."""
    try:
        # Pastikan file model ini ada di direktori yang sama dengan skrip Anda.
        model = joblib.load('california_house_price_model.joblib')
        return model
    except FileNotFoundError:
        st.error("File model 'california_house_price_model.joblib' tidak ditemukan. Pastikan file berada di folder yang sama dengan skrip ini.")
        return None

@st.cache_data
def load_data():
    """Memuat dataset asli untuk visualisasi."""
    try:
        # Pastikan file CSV ini ada di direktori yang sama dengan skrip Anda.
        df = pd.read_csv('BI - California Housing Dataset.csv', delimiter=';')
        df.dropna(inplace=True) # Membersihkan data dari nilai kosong untuk visualisasi yang stabil
        return df
    except FileNotFoundError:
        st.error("File dataset 'BI - California Housing Dataset.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan skrip ini.")
        return None

# Memuat model dan data di awal
model = load_model()
df = load_data()

# --- FUNGSI TAMPILAN UNTUK INVESTOR ---
def investor_view():
    """
    Menampilkan halaman untuk investor yang memungkinkan mereka melakukan prediksi harga
    berdasarkan input fitur properti.
    """
    st.header("📈 Prediktor Harga Properti untuk Investor")
    st.write("Klik titik di peta untuk memilih lokasi, atau masukkan koordinat manual. Lengkapi spesifikasi properti di bawah untuk mendapatkan estimasi harga.")

    # Inisialisasi session state untuk menyimpan lokasi yang diklik peta
    if 'clicked_location' not in st.session_state:
        st.session_state.clicked_location = {'lat': 34.05, 'lng': -118.24} # Default: Los Angeles

    # --- Peta Interaktif untuk Memilih Lokasi ---
    st.subheader("Pilih Lokasi di Peta")
    m = folium.Map(location=[36.77, -119.41], zoom_start=6) # Titik tengah California
    
    # Tambahkan event handler agar peta bisa merespon klik dan menampilkan popup koordinat
    folium.LatLngPopup().add_to(m)

    map_data = st_folium(m, width='100%', height=500) # Atur tinggi peta agar tidak terlalu dominan

    # Perbarui session state jika ada lokasi baru yang diklik di peta
    if map_data and map_data['last_clicked']:
        st.session_state.clicked_location = map_data['last_clicked']

    # --- INPUT FITUR PROPERTI ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Lokasi Properti")
        # Gunakan nilai dari session state sebagai nilai default input
        lat_val = st.session_state.clicked_location['lat']
        lon_val = st.session_state.clicked_location['lng']
        
        latitude = st.number_input('Latitude', value=lat_val, format="%.4f")
        longitude = st.number_input('Longitude', value=lon_val, format="%.4f")

        ocean_proximity = st.selectbox(
            'Kedekatan dengan Laut',
            ('<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'),
            help="Pilih kategori lokasi berdasarkan kedekatannya dengan laut."
        )

    with col2:
        st.subheader("📋 Spesifikasi Blok Properti")
        housing_median_age = st.slider('Usia Median Rumah (Tahun)', 1, 52, 25)
        # PERBAIKAN: Mengganti "Total Kamar" menjadi "Total Ruang" agar konsisten dan lebih akurat.
        total_rooms = st.number_input('Total Ruang', min_value=0, value=2100, step=100)
        total_bedrooms = st.number_input('Total Kamar Tidur', min_value=0, value=440, step=50)
        population = st.number_input('Populasi', min_value=0, value=1200, step=100)
        households = st.number_input('Jumlah Rumah Tangga', min_value=1, value=410, step=50)
        median_income = st.number_input('Pendapatan Median (dalam puluhan ribu USD)', min_value=0.0, value=3.2, step=0.5, format="%.2f")

    if st.button('Prediksi Harga', type="primary", use_container_width=True):
        if model is not None:
            # --- Preprocessing Input ---
            # 1. Feature Engineering
            # PERBAIKAN: Pengecekan pembagian dengan nol yang lebih aman
            if households > 0 and total_rooms > 0 and total_bedrooms > 0:
                rooms_per_household = total_rooms / households
                bedrooms_per_room = total_bedrooms / total_rooms
                population_per_household = population / households
            else:
                st.error("Jumlah Rumah Tangga, Total Ruang, dan Total Kamar Tidur harus lebih dari 0 untuk perhitungan rasio.")
                return

            # 2. One-Hot Encoding manual
            # SARAN: Untuk aplikasi produksi, lebih baik menggunakan pipeline Scikit-Learn
            # yang menyertakan encoder yang sudah di-'fit' untuk menghindari kesalahan.
            ocean_features = {f'ocean_{cat}': 0 for cat in ('<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND')}
            if f'ocean_{ocean_proximity}' in ocean_features:
                ocean_features[f'ocean_{ocean_proximity}'] = 1

            # Kolom harus sesuai persis dengan yang digunakan saat training model
            expected_columns = [
                'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                'population', 'households', 'median_income', 'rooms_per_household',
                'bedrooms_per_room', 'population_per_household',
                'ocean_<1H OCEAN', 'ocean_INLAND', 'ocean_ISLAND', 'ocean_NEAR BAY',
                'ocean_NEAR OCEAN'
            ]

            # Membuat DataFrame dari input
            input_data = pd.DataFrame([[
                longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                population, households, median_income, rooms_per_household,
                bedrooms_per_room, population_per_household,
                ocean_features['ocean_<1H OCEAN'], ocean_features['ocean_INLAND'],
                ocean_features['ocean_ISLAND'], ocean_features['ocean_NEAR BAY'],
                ocean_features['ocean_NEAR OCEAN']
            ]], columns=expected_columns)

            try:
                # Melakukan prediksi
                prediction = model.predict(input_data)[0]
                st.success(f"Prediksi Median Harga Rumah: **${prediction:,.2f}**")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.warning("Pastikan urutan dan nama kolom pada input data sudah sesuai dengan model yang dilatih.")
        else:
            st.error("Model tidak dapat dimuat, prediksi tidak bisa dilakukan.")

# --- FUNGSI TAMPILAN UNTUK Market Survey ---
def buyer_view():
    """
    Menampilkan halaman untuk Market Survey yang memungkinkan mereka memfilter dan
    mengeksplorasi properti yang ada di dataset.
    """
    st.header("🏡 Eksplorasi Properti untuk Market Survey")
    st.write("Gunakan filter di sidebar untuk menemukan area properti yang sesuai dengan kriteria Anda.")

    # --- Filter di Sidebar ---
    st.sidebar.title("Filter Properti")
    
    # PERBAIKAN: Menggunakan nilai min/max dari data secara dinamis
    min_price = int(df['median_house_value'].min())
    max_price = int(df['median_house_value'].max())
    price_range = st.sidebar.slider(
        'Rentang Harga ($)',
        min_value=min_price,
        max_value=max_price,
        value=(min_price, 200000)
    )

    # PERBAIKAN BUG: Mengoreksi typo dari 'multilibelect' menjadi 'multiselect'.
    location_filter = st.sidebar.multiselect(
        'Pilih Lokasi',
        options=df['ocean_proximity'].unique(),
        default=df['ocean_proximity'].unique()
    )
    
    # PERBAIKAN: Batas atas slider dibuat dinamis tapi dibatasi agar lebih usable
    max_rooms = int(df['total_rooms'].max())
    rooms_range = st.sidebar.slider(
        'Rentang Total Ruang',
        min_value=int(df['total_rooms'].min()),
        max_value=min(max_rooms, 10000), # Batasi max value agar slider tidak terlalu lebar
        value=(1000, 3000)
    )

    max_bedrooms = int(df['total_bedrooms'].max())
    bedrooms_range = st.sidebar.slider(
        'Rentang Total Kamar Tidur',
        min_value=int(df['total_bedrooms'].min()),
        max_value=min(max_bedrooms, 2500),
        value=(200, 600)
    )

    # Melakukan filter pada DataFrame berdasarkan input dari sidebar
    filtered_df = df[
        (df['median_house_value'].between(price_range[0], price_range[1])) &
        (df['ocean_proximity'].isin(location_filter)) &
        (df['total_rooms'].between(rooms_range[0], rooms_range[1])) &
        (df['total_bedrooms'].between(bedrooms_range[0], bedrooms_range[1]))
    ]

    st.write(f"Ditemukan **{len(filtered_df)}** blok properti yang sesuai dengan kriteria Anda.")

    # --- Visualisasi Peta ---
    if not filtered_df.empty:
        # SOLUSI TOOLTIP: Mengganti HexagonLayer dengan ScatterplotLayer
        st.pydeck_chart(pdk.Deck(
            map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
            initial_view_state=pdk.ViewState(
                latitude=filtered_df['latitude'].mean(),
                longitude=filtered_df['longitude'].mean(),
                zoom=6,
                pitch=45, # Sedikit mengurangi pitch untuk tampilan scatterplot
            ),
            layers=[
                pdk.Layer(
                   'ScatterplotLayer',
                   data=filtered_df,
                   get_position='[longitude, latitude]',
                   get_fill_color='[255, (255 - (median_house_value / 2000)), 0, 140]', # Warna berdasarkan harga
                   get_radius=1500, # Radius setiap titik
                   pickable=True,
                ),
            ],
            # PERBAIKAN TOOLTIP: Menghapus format string (:,.0f) yang tidak valid agar nilai dapat ditampilkan.
            tooltip={"html": "Median Harga: <b>${median_house_value}</b><br/>Lokasi: {ocean_proximity}"}
        ))

        # PENAMBAHAN: Menambahkan catatan untuk menjelaskan arti warna pada peta
        st.info("💡 **Catatan Peta:** Warna titik merepresentasikan harga properti. **Kuning** untuk harga lebih rendah, dan semakin **merah** untuk harga yang lebih tinggi.")

        # PERBAIKAN: Menghapus "(Contoh Data)" dari subheader
        st.subheader("Detail Properti yang Difilter")
        st.dataframe(filtered_df[[
            'longitude', 'latitude', 'median_house_value', 'ocean_proximity', 
            'total_rooms', 'total_bedrooms', 'population', 'households'
        ]].head(10))
    else:
        st.warning("Tidak ada properti yang ditemukan dengan kriteria tersebut. Silakan sesuaikan filter Anda.")


# --- LAYOUT UTAMA APLIKASI ---
st.title("🏠 California Property Insights Dashboard")

# Navigasi di sidebar
selected_view = st.sidebar.radio(
    "Pilih Tampilan Anda:",
    ("Investor", "Market Survey"),
    key="view_selector"
)

# Menampilkan halaman yang dipilih berdasarkan navigasi
if df is not None and model is not None:
    if selected_view == "Investor":
        investor_view()
    elif selected_view == "Market Survey":
        buyer_view()
else:
    st.error("Gagal memuat data atau model. Aplikasi tidak dapat berjalan. Mohon periksa file di direktori Anda.")

