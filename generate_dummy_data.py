import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
start_time = 8
end_time = 17
num_devices = 5
device_ids = pd.read_csv("device_log.csv")['unique_id'].tolist()
start_date = datetime(2025, 9, 1, start_time, 0, 0)
end_date = datetime(2025, 9, 7, end_time, 0, 0)  # 1 week
time_interval = timedelta(seconds=15)  # every 15 seconds

# Generate timestamps (only between start_time - 5pm each day)
timestamps = []
current = start_date
while current <= end_date:
    if current.hour >= start_time and current.hour < end_time:
        timestamps.append(current)
    current += time_interval

utp_locations = [
    (4.3850, 100.9755),  # Chancellor Hall
    (4.3858, 100.9750),  # IRC (Library)
    (4.3862, 100.9738),  # Pocket D (Lecture Complex)
    (4.3842, 100.9729),  # Mosque
    (4.3890, 100.9715),  # Village 4 (Hostel area)
    (4.3905, 100.9740),  # Village 5 (Hostel area)
]

# Generate data with Malaysian climate ranges
data = []

def temp_with_peak(dt, temp_base, device_index):
    # seconds since midnight
    seconds_since_midnight = dt.hour*3600 + dt.minute*60 + dt.second
    
    # desired peak at 12:00 (12 PM)
    peak_seconds = 12 * 3600
    
    # 24-hour period
    period_seconds = 24 * 3600
    
    # compute angle, add phase shift of π/2 so peak aligns at 12 PM
    angle = 2 * np.pi * (seconds_since_midnight - peak_seconds) / period_seconds
    phase_shift = np.pi / 2  # shift sine so peak = 12 PM
    
    # temperature with small device-dependent amplitude variation + noise
    temp = temp_base + (4 + 0.2*device_index) * np.sin(angle + phase_shift) \
           + 2 * np.sin(2*angle) \
           + np.random.normal(-0.5, 0.5)
    return temp


def humidity_with_variation(dt, hum_base, device_index):
    # inverse relation to temperature, smooth daily cycle
    seconds_since_midnight = dt.hour*3600 + dt.minute*60 + dt.second
    period_seconds = 24 * 3600
    angle = 2 * np.pi * (seconds_since_midnight - 13*3600) / period_seconds
    
    humidity = hum_base - (3 + 0.1*device_index) * np.sin(angle) \
               + 2 * np.sin(2*angle) \
               + np.random.normal(-0.8, 0.8)
    return humidity


for device_index, device in enumerate(device_ids):
    # Device-specific baselines
    temp_base = np.random.uniform(27, 30)
    hum_base = np.random.uniform(70, 85)

    # Starting GPS from UTP list
    lat, lon = utp_locations[device_index % len(utp_locations)]

    for i, ts in enumerate(timestamps):
        # Smooth temperature/humidity
        temp = temp_with_peak(ts, temp_base, device_index)
        humidity = humidity_with_variation(ts, hum_base, device_index)

        temp = round(min(max(temp, 26), 34), 2)
        humidity = round(min(max(humidity, 65), 90), 2)

        # Decibels
        decibels = round(np.random.uniform(50, 70), 2)

        # Random walk for GPS (small step in degrees ≈ meters)
        lat += np.random.normal(0, 0.00005)   # ~5m step
        lon += np.random.normal(0, 0.00005)   # ~5m step

        latitude = round(lat, 7)
        longitude = round(lon, 7)

        data.append([ts, device, temp, humidity, decibels, latitude, longitude])


# Create DataFrame
df = pd.DataFrame(data, columns=["datetime", "unique_id", "temp", "humidity", "decibels", "latitude", "longitude"])
df.sort_values(by=["datetime", "unique_id"], inplace=True)

# Save to CSV
df.to_csv("data.csv", index=False)

print("Dataset generated and saved as data.csv")
