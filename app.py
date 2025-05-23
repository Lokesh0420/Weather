import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load models and data
clf = joblib.load('weather_classifier.pkl')
reg = joblib.load('temperature_regressor.pkl')
le_weather = joblib.load('label_encoder.pkl')
df = pd.read_csv("final_dataset_Insights.csv")
radar_model = load_model('radar_cnn_model.h5')
radar_classes = ['Clear', 'Rain', 'Storm', 'Cloudy']

# Page configuration
st.set_page_config(page_title="Weather Dashboard", layout="wide")

# Theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
            .main { background-color: #1e1e1e; color: #e0e0e0; }
            .stSidebar { background-color: #2e2e2e; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .main { background-color: #f4f4f4; color: #333; }
            .stSidebar { background-color: #2E7D32; }
        </style>
    """, unsafe_allow_html=True)

# Page selection
page = st.sidebar.selectbox("Choose Page", ["Prediction", "Data Insights", "Real-Time Prediction", "Radar Image Classifier"])

if page == "Prediction":
    st.title("Real-Time Weather Prediction Dashboard")
    st.markdown("Predict the **weather type** and **temperature** based on your inputs.")

    st.sidebar.header("Input Weather Conditions")

    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 10)
    pressure = st.sidebar.slider("Pressure (hPa)", 900, 1100, 1013)
    temperature = st.sidebar.slider("Temperature (°C)", -20, 50, 25)
    city = st.sidebar.selectbox("City", ["New York", "London", "Tokyo", "Sydney", "Mumbai"])
    day_night = st.sidebar.radio("Time of Day", ["Day", "Night"])

    city_encoded = {"New York": 0, "London": 1, "Tokyo": 2, "Sydney": 3, "Mumbai": 4}[city]
    day_night_encoded = {"Day": 1, "Night": 0}[day_night]

    features_classification = [[humidity, wind_speed, pressure, temperature, city_encoded, day_night_encoded]]
    features_regression = [[humidity, wind_speed, pressure, city_encoded, day_night_encoded]]

    predicted_temp = reg.predict(features_regression)[0]
    predicted_probs = clf.predict_proba(features_classification)[0]
    predicted_class = clf.predict(features_classification)[0]
    predicted_label = le_weather.inverse_transform([predicted_class])[0]
    confidence = max(predicted_probs) * 100

    st.subheader("Prediction Results")
    st.info(f"**Predicted Temperature:** {round(predicted_temp, 2)} °C")
    st.success(f"**Predicted Weather Type:** {predicted_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # CSV Download
    prediction_df = pd.DataFrame({
        "Humidity": [humidity],
        "Wind Speed": [wind_speed],
        "Pressure": [pressure],
        "Temperature Input": [temperature],
        "Predicted Temperature": [predicted_temp],
        "Predicted Weather": [predicted_label],
        "Confidence": [confidence]
    })

    st.download_button(
        label="Download Prediction as CSV",
        data=prediction_df.to_csv(index=False),
        file_name="weather_prediction.csv",
        mime="text/csv"
    )

elif page == "Data Insights":
    st.title("Weather Data Insights")
    st.markdown("Explore patterns and trends from the dataset.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weather Type Distribution")
        fig1, ax1 = plt.subplots()
        df['weather_type'].value_counts().plot(kind='bar', ax=ax1, color='#81D4FA')
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Weather Type")
        st.pyplot(fig1)

    with col2:
        st.subheader("Temperature vs Humidity")
        fig2, ax2 = plt.subplots()
        sns.set_style("whitegrid")
        sns.scatterplot(data=df, x='humidity', y='temperature', hue='weather_type', ax=ax2, palette='coolwarm')
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Pressure vs Wind Speed")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x='pressure', y='wind_speed', hue='weather_type', ax=ax3, palette='viridis')
        st.pyplot(fig3)

    with col4:
        st.subheader("Weather Type by City")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df, x='city', y='temperature', hue='weather_type', ax=ax4, palette='Set2')
        st.pyplot(fig4)

    st.success("Insights generated successfully!")

    # CSV Download for insights data
    st.download_button(
        label="Download Full Dataset as CSV",
        data=df.to_csv(index=False),
        file_name='weather_insights_data.csv',
        mime='text/csv'
    )

elif page == "Real-Time Prediction":
    st.title("Real-Time Weather Prediction using WeatherAPI")
    st.markdown("Fetch **live weather data** and predict the weather type using the trained model.")

    # Get default city from IP
    def get_user_city():
        try:
            ip_info = requests.get("https://ipinfo.io/json").json()
            return ip_info.get("city", "")
        except:
            return ""

    location = st.text_input("Enter location (e.g., London, Tokyo):", value=get_user_city())
    api_key = "5bc16a3aeee0437d822184413250704"

    def get_weatherapi_data(location, api_key):
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
        response = requests.get(url)
        data = response.json()
        return {
            "humidity": data["current"]["humidity"],
            "wind_speed": data["current"]["wind_kph"],
            "pressure": data["current"]["pressure_mb"],
            "temperature": data["current"]["temp_c"],
            "city_encoded": 0,  # Ideally use a dynamic encoder
            "day_night_encoded": 1 if data["current"]["is_day"] == 1 else 0
        }

    if st.button("Predict Real-Time Weather Type"):
        if location and api_key:
            try:
                weather_data = get_weatherapi_data(location, api_key)
                st.subheader("Fetched Real-Time Data")
                for key, value in weather_data.items():
                    if key != "city_encoded":
                        st.write(f"{key.replace('_', ' ').title()}: {value}")

                input_df = pd.DataFrame([weather_data])
                prediction = clf.predict(input_df)
                predicted_weather_label = le_weather.inverse_transform(prediction)[0]
                st.success(f"Predicted Real-Time Weather Type: **{predicted_weather_label}**")

                st.download_button(
                    label="Download Real-Time Prediction as CSV",
                    data=input_df.to_csv(index=False),
                    file_name="real_time_weather_prediction.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f" Error: {e}")
        else:
            st.warning("Please enter a valid location.")

elif page == "Radar Image Classifier":
    st.title("Radar Image Weather Classification")
    uploaded = st.file_uploader("Upload a radar image", type=["jpg", "png"])

    if uploaded:
        img = image.load_img(uploaded, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = radar_model.predict(img_array)
        label = radar_classes[np.argmax(prediction)]

        st.image(uploaded, caption="Uploaded Radar Image", use_column_width=True)
        st.success(f"Predicted Weather Type from Radar: **{label}**")
