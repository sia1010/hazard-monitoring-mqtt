import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ====================================================
# Helper Functions
# ====================================================
@st.cache_data(ttl=60)
def load_data(path: str, mtime) -> pd.DataFrame:
    """Load and preprocess the sensor CSV file."""
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['heat_index'] = compute_heat_index(df['temp'], df['humidity'])
    return df

def compute_heat_index(temp, humidity):
    """
    Compute heat index using NOAA formula (simplified).
    Temp in °C, humidity in %.
    """
    T, R = temp, humidity
    HI = (
        -8.78469475556
        + 1.61139411 * T
        + 2.33854883889 * R
        - 0.14611605 * T * R
        - 0.012308094 * T**2
        - 0.0164248277778 * R**2
        + 0.002211732 * T**2 * R
        + 0.00072546 * T * R**2
        - 0.000003582 * T**2 * R**2
    )
    return HI

@st.cache_data(ttl=60)
def filter_data(df: pd.DataFrame, selected_ids, start_date, end_date):
    """Apply device and datetime filters."""
    mask = (
        df['unique_id'].isin(selected_ids)
        & (df['datetime'] >= start_date)
        & (df['datetime'] <= end_date)
    )
    return df[mask]

@st.cache_data(ttl=60)
def resample_data(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    return (
        df.set_index("datetime")
        .groupby("unique_id")
        .resample(freq)
        .agg({
            "decibels": leq,        # proper time-weighted exposure
            "temp": "mean",         # still normal avg for temp
            "humidity": "mean",
            "heat_index": "mean"
        })
        .reset_index()
    )

def leq(series):
    # convert dB to linear, average, convert back to dB
    return 10 * np.log10(np.mean(10 ** (series / 10)))

# ====================================================
# Streamlit App
# ====================================================
st.set_page_config(page_title="ESP32 Sensor Dashboard", layout="wide")
st.title("📊 ESP32 IoT Sensor Dashboard")

# Load data
file_mtime = os.path.getmtime("data.csv")
df = load_data("data.csv", file_mtime)
user_df = pd.read_csv("device_log.csv")
df = df.merge(user_df, on='unique_id', how='left')

# Initialize session state for tab
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Heat Index"

# ====================================================
# Sidebar Filters
# ====================================================
st.sidebar.header("Filters")

# --- Device filter ---
st.sidebar.subheader("Device Selection")

selected_users = st.sidebar.multiselect(
    "Select User",
    placeholder="Select one or more users",
    options=df["username"].unique(),  # unique values
    default=None  # all selected by default
)

# If no selection, use all users
if selected_users:
    selected_ids = df[df["username"].isin(selected_users)]["unique_id"].unique()
else:
    selected_ids = df["unique_id"].unique()

# --- Date range filter ---
st.sidebar.subheader("Datetime Range")
min_date, max_date = df['datetime'].min().date(), df['datetime'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Ensure tuple unpacking
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# --- Resample frequency ---
freq = st.sidebar.selectbox(
    "Resample Frequency",
    options=["1min", "5min", "10min", "30min", "1h", "24h"],
    index=3,  # default = "30min"
)

# Apply filters
filtered_df = filter_data(df, selected_ids, start_date, end_date)
df_resampled = resample_data(filtered_df, freq=freq).merge(user_df, on='unique_id', how='left')

# ====================================================
# Tabs
# ====================================================

st.session_state.active_tab = st.tabs(["Heat Index", "Sound Exposure", "Raw Data"])

with st.session_state.active_tab[0]:
    # ====================================================
    # Heat Index
    # ====================================================
    st.subheader("🌡️ Heat Index")
    fig2 = px.line(
        df_resampled,
        x="datetime",
        y="heat_index",
        color="unique_id",
        markers=True,
        title="Heat Index per Device",
        hover_data={
            "unique_id": False,
            "datetime": True,
            "temp": True,
            "humidity": True,
            "heat_index": True,
            "username": True
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

with st.session_state.active_tab[1]:
    # ====================================================
    # Decibels Chart
    # ====================================================
    st.subheader("🔊 Sound Exposure Level (dB)")
    fig2 = px.line(
        df_resampled,
        x="datetime",
        y="decibels",
        color="unique_id",
        markers=True,
        title="Decibels per Device",
        hover_data={
            "unique_id": False,
            "datetime": True,
            "decibels": True,
            "username": True
        }
        
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Extract date
    df_resampled['date'] = df_resampled['datetime'].dt.strftime('%d/%m')  # format as you want

    # Compute daily Leq per device
    daily_exposure = (
        df_resampled.groupby(['unique_id', 'date'])
        .agg({'decibels': leq})  # Leq is already computed in df_resampled
        .reset_index()
        .rename(columns={'decibels': 'Leq (dB)'})
    )

    # Pivot table: rows = device, columns = date
    pivot_exposure = daily_exposure.pivot(index='unique_id', columns='date', values='Leq (dB)')

    st.subheader("📅 Daily Sound Exposure per Device")
    st.dataframe(pivot_exposure)

with st.session_state.active_tab[2]:
    # ====================================================
    # Raw Data Table
    # ====================================================
    st.subheader("📄 Device-User log")
    st.dataframe(user_df)
    st.subheader("📄 Filtered Data Table")
    st.dataframe(filtered_df)