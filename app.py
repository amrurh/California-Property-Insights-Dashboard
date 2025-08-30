import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import joblib

# Set page configuration
st.set_page_config(page_title="California Property Insights",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('BI - California Housing Dataset.csv', delimiter=';')
    return data

df = load_data()

# Load the pre-trained model
model = joblib.load('california_house_price_model.joblib')

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Main Menu',
        ['Home', 'Market Survey', 'Price Prediction'],
        icons=['house', 'bar-chart-line', 'graph-up-arrow'],
        menu_icon="cast",
        default_index=0
    )

# Home Page
if selected == 'Home':
    st.title('🏡 California Property Insights Dashboard')
    st.markdown("""
        Selamat datang di Dasbor Wawasan Properti California. Platform ini memberikan analisis komprehensif 
        tentang pasar perumahan di California. Dasbor ini dirancang untuk memberikan wawasan berharga bagi 
        pembeli rumah, investor, dan analis real estat.
    """)
    st.markdown("---")

    # Display key metrics
    st.header('Key Metrics')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Properti", value=f"{df.shape[0]:,}")
    with col2:
        st.metric(label="Harga Rumah Rata-rata", value=f"${df['median_house_value'].mean():,.0f}")
    with col3:
        st.metric(label="Pendapatan Rata-rata", value=f"${df['median_income'].mean()*10000:,.0f}")
    with col4:
        st.metric(label="Populasi Rata-rata per Blok", value=f"{df['population'].mean():,.0f}")
    
    st.markdown("---")

    # Data Preview
    st.header('Pratinjau Data')
    st.dataframe(df.head())
    
# Market Survey Page
elif selected == 'Market Survey':
    st.title('📊 Analisis Survei Pasar')
    st.markdown("Halaman ini menyajikan analisis mendalam tentang berbagai aspek pasar perumahan California.")

    # Geographical Distribution of House Prices
    st.header('Distribusi Geografis Properti')
    
    # Define the color palette for ocean proximity
    ocean_palette = {
        'NEAR BAY': '#78ADD2',
        '<1H OCEAN': '#FFB26E',
        'INLAND': '#80C680',
        'NEAR OCEAN': '#E67D7E',
        'ISLAND': '#BEA4D7'
    }

    # Add a radio button to toggle map coloring
    color_by = st.radio(
        "Pilih pewarnaan plot:",
        ('Intensitas Harga', 'Kategori Lokasi'),
        horizontal=True
    )

    if color_by == 'Intensitas Harga':
        # Create a scatter mapbox plot colored by price
        fig_map = px.scatter_mapbox(df,
                                lat="latitude",
                                lon="longitude",
                                color="median_house_value",
                                size="population",
                                color_continuous_scale=px.colors.cyclical.IceFire,
                                size_max=15,
                                zoom=4,
                                mapbox_style="carto-positron",
                                hover_name="ocean_proximity",
                                hover_data=["median_income", "population", "median_house_value"],
                                title="Distribusi Properti Berdasarkan Intensitas Harga")
    else:  # Kategori Lokasi
        # Create a scatter mapbox plot colored by location category
        fig_map = px.scatter_mapbox(df,
                                lat="latitude",
                                lon="longitude",
                                color="ocean_proximity",
                                size="population",
                                color_discrete_map=ocean_palette,
                                size_max=15,
                                zoom=4,
                                mapbox_style="carto-positron",
                                hover_name="ocean_proximity",
                                hover_data=["median_income", "population", "median_house_value"],
                                title="Distribusi Properti Berdasarkan Kategori Lokasi")
    
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # Other visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.header('Harga Rumah vs Pendapatan')
        fig_scatter = px.scatter(df, x='median_income', y='median_house_value',
                                 color='ocean_proximity',
                                 title='Harga Rumah vs. Pendapatan Rata-rata',
                                 labels={'median_income': 'Pendapatan Rata-rata ($10k)', 'median_house_value': 'Harga Rumah Rata-rata'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.header('Distribusi Harga Rumah')
        fig_hist = px.histogram(df, x='median_house_value',
                                nbins=50,
                                title='Distribusi Harga Rumah Rata-rata',
                                labels={'median_house_value': 'Harga Rumah Rata-rata'})
        st.plotly_chart(fig_hist, use_container_width=True)

# Price Prediction Page
elif selected == 'Price Prediction':
    st.title('📈 Prediksi Harga Rumah')
    st.markdown("""
        Gunakan model machine learning kami untuk memprediksi harga rumah di California berdasarkan berbagai fitur. 
        Silakan masukkan nilai-nilai di bawah ini untuk mendapatkan estimasi harga.
    """)

    # Create input fields for user
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            longitude = st.number_input('Longitude', value=-122.23)
            latitude = st.number_input('Latitude', value=37.88)
            housing_median_age = st.number_input('Usia Rata-rata Rumah', min_value=1, max_value=100, value=41)
            total_rooms = st.number_input('Total Kamar', min_value=1, value=880)
            total_bedrooms = st.number_input('Total Kamar Tidur', min_value=1, value=129)
        with col2:
            population = st.number_input('Populasi', min_value=1, value=322)
            households = st.number_input('Rumah Tangga', min_value=1, value=126)
            median_income = st.number_input('Pendapatan Rata-rata ($10k)', min_value=0.1, value=8.3252)
            ocean_proximity = st.selectbox('Kedekatan dengan Laut',
                                           options=['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])
        
        submit_button = st.form_submit_button(label='Prediksi Harga')

    if submit_button:
        # Create a dataframe from user input
        ocean_proximity_mapping = {
            '<1H OCEAN': 0, 'INLAND': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4
        }
        
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity_mapping[ocean_proximity]]
        })

        # Make prediction
        prediction = model.predict(input_data)
        predicted_price = prediction[0]

        st.success(f'**Prediksi Harga Rumah:** `${predicted_price:,.2f}`')
