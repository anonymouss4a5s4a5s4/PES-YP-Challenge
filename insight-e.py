import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import holidays
from zipfile import ZipFile
from io import BytesIO
import os

PROJECT_NAME = "Project Insight-E"
OUTPUT_FILENAME = "insight-e_correlation_heatmap.png"

LOCATION_META = {"latitude": 48.78, "longitude": 2.29, "country": "FR"}
DATE_RANGE = {"start": "2006-12-16", "end": "2010-11-26"}


def load_energy_data():
    """Downloads and prepares the primary household energy consumption data."""
    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    local_file = "household_power_consumption.txt"

    if not os.path.exists(local_file):
        print(f"Downloading energy data to {local_file}...")
        response = requests.get(zip_url, stream=True)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to download energy data. Status: {response.status_code}")
        with ZipFile(BytesIO(response.content)) as z:
            z.extractall()

    print("Loading and preprocessing energy data...")
    df = pd.read_csv(
        local_file, sep=';', na_values=['?'],
        parse_dates={'datetime': ['Date', 'Time']},
        infer_datetime_format=True, dayfirst=True, index_col='datetime'
    )
    
    df.fillna(method='ffill', inplace=True)

    df_hourly = df.resample('h').agg({
        'Global_active_power': 'sum', 'Global_reactive_power': 'sum',
        'Voltage': 'mean', 'Global_intensity': 'mean',
        'Sub_metering_1': 'sum', 'Sub_metering_2': 'sum', 'Sub_metering_3': 'sum'
    })
    df_hourly.dropna(inplace=True)
    return df_hourly


def fetch_weather_data(lat, lon, start_date, end_date):
    """Fetches historical weather data from the Open-Meteo API."""
    print("Fetching weather data from API...")
    api_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relativehumidity_2m,apparent_temperature,precipitation,windspeed_10m"
    }
    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch weather data. Status: {response.status_code}")
    
    data = response.json()['hourly']
    df_weather = pd.DataFrame(data)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather.set_index('time', inplace=True)
    return df_weather


def add_calendar_features(df, country_code):
    """Adds holiday and time-based features for socio-economic context."""
    print("Adding calendar-based features...")
    country_holidays = holidays.CountryHoliday(country_code)
    df['is_holiday'] = df.index.to_series().apply(lambda x: x in country_holidays).astype(int)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    return df


def create_correlation_heatmap(df):
    """Generates and saves a correlation heatmap for key features."""
    print("Generating feature correlation heatmap...")
    features_of_interest = [
        'Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'temperature_2m', 'apparent_temperature', 'relativehumidity_2m',
        'is_holiday', 'is_weekend', 'hour'
    ]
    
    df_plot = df[features_of_interest].copy()
    df_plot.rename(columns={
        'Global_active_power': 'Total Power', 'Sub_metering_1': 'Kitchen',
        'Sub_metering_2': 'Laundry', 'Sub_metering_3': 'HVAC/Heater',
        'temperature_2m': 'Temp (°C)', 'apparent_temperature': 'Apparent Temp (°C)',
        'relativehumidity_2m': 'Humidity (%)'
    }, inplace=True)

    correlation_matrix = df_plot.corr()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
        linewidths=.5, ax=ax, annot_kws={"size": 10}
    )
    ax.set_title(f'{PROJECT_NAME}: Feature Correlation Matrix', fontsize=18, pad=20)
    
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight')
    print(f"Analysis complete. Heatmap saved as '{OUTPUT_FILENAME}'")


def main():
    """Main pipeline to load, merge, and analyze the data."""
    print(f"--- Initializing {PROJECT_NAME} Data Pipeline ---")
    
    df_energy = load_energy_data()
    
    df_weather = fetch_weather_data(
        LOCATION_META["latitude"], LOCATION_META["longitude"],
        DATE_RANGE["start"], DATE_RANGE["end"]
    )

    print("Merging data sources...")
    df_merged = df_energy.join(df_weather, how='inner')
    
    df_final = add_calendar_features(df_merged, LOCATION_META["country"])
    
    create_correlation_heatmap(df_final)

    print(f"--- Pipeline Finished Successfully ---")


if __name__ == '__main__':
    main()