from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse
from gmqtt import Client as MQTTClient
from fastapi_mqtt import FastMQTT, MQTTConfig
from starlette.websockets import WebSocketDisconnect, WebSocketState
import datetime
import csv
import base64

# Cryptographic primitive for authenticated encryption
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305 

import pandas as pd
import numpy as np
import time
import asyncio
import aiofiles
import os

import ntplib
from time import ctime

#####################################################################################
############################## Handling Data from MQTT ##############################
#####################################################################################

# Configure MQTT broker connection
mqtt_config = MQTTConfig(
    host="mqtt-dashboard.com",
    port=1883
)
fast_mqtt = FastMQTT(config=mqtt_config)
# Dictionary to store device keys: {unique_id: key_bytes}
device_keys = {}

# Global set to hold active WebSocket connections for broadcasting alerts
active_websockets: set[WebSocket] = set() 

# Async locks to safely manage concurrent access to shared resources
file_lock = asyncio.Lock()  # Protects access to data.csv
ws_lock = asyncio.Lock()    # Protects access to the active_websockets set

# Application lifespan context manager for startup/shutdown tasks
@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Start the MQTT client connection
    await fast_mqtt.mqtt_startup()
    yield
    # Gracefully shut down the MQTT client
    await fast_mqtt.mqtt_shutdown()

# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=_lifespan)


# Create NTP client (used in commented-out latency computation)
ntpclient = ntplib.NTPClient()

# Handler for successful MQTT connection
@fast_mqtt.on_connect()
def connect(client: MQTTClient, flags: int, rc: int, properties: Any):
    # Subscribe to the topic where devices publish encrypted data
    client.subscribe("hazard-monitoring/server")
    print("Connected: ", client, flags, rc, properties)

    # ------------------ LOAD DEVICE KEYS FROM device_log.csv ------------------
    # Load all unique device IDs and their secret keys into the device_keys dictionary
    with open("device_log.csv", mode="r") as f:
        # Skip the header line
        next(f) 
        # Read the rest of the file using specified fieldnames
        reader = csv.DictReader(f, fieldnames=['unique_id', 'username', 'key'])
        for row in reader:
            # Convert the hex string key from the CSV into bytes
            device_keys[row["unique_id"]] = bytes.fromhex(row["key"])
    # ---------------------------------------------------------------------

# Handler for incoming MQTT messages on the subscribed topic
@fast_mqtt.on_message()
async def message(client: MQTTClient, topic: str, payload: bytes, qos: int, properties: Any):
    try:
        # Message format: device_id,nonce,ciphertext,tag (all base64 except id)
        message = payload.decode().strip()
        parts = message.split(",")
        if len(parts) != 4:
            print("Invalid message format:", message)
            return

        device_id, nonce_b64, ciphertext_b64, tag_b64 = parts

        # Check if the device is registered and has a key
        if device_id not in device_keys:
            print(f"Unknown device ID: {device_id}")
            return

        # --- Decode Base64 ---
        nonce = base64.b64decode(nonce_b64)
        ciphertext = base64.b64decode(ciphertext_b64)
        tag = base64.b64decode(tag_b64)

        # --- Decrypt using ChaCha20Poly1305 ---
        key = device_keys[device_id]
        chacha = ChaCha20Poly1305(key)
        # Combine ciphertext and tag for decryption with authentication
        plaintext = chacha.decrypt(nonce, ciphertext + tag, None) 
        
        # Expected decrypted payload structure: 
        # client_id,avg_t,avg_h,avg_spl,lat,lng,fix_age,status,timestamp
        decrypted_msg = plaintext.decode().split(",")

        # Validate the number of fields in the decrypted message
        if len(decrypted_msg) != 9:
            print("Decryption successful, but invalid decrypted field count:", len(decrypted_msg), "Expected 9.")
            return

        # Ensure the device ID in the payload matches the one from the MQTT message
        if decrypted_msg[0] != device_id:
            print("Device ID mismatch", device_id, decrypted_msg[0])
            return
        
        # Extract the status and timestamp fields
        status = decrypted_msg[7] 
        # Parse the timestamp and format it by removing microseconds (for cleaner logging)
        timestamp = str(datetime.datetime.strptime(decrypted_msg[8], "%Y-%m-%d %H:%M:%S.%f"))[:-7]

        # Compute latency
        # ntp_time = datetime.datetime.fromtimestamp(ntpclient.request('pool.ntp.org').tx_time)
        # latency_sec = max(0,(ntp_time - datetime.datetime.strptime(decrypted_msg[8], "%Y-%m-%d %H:%M:%S.%f")).total_seconds())
        # print(f"Message Received from {device_id}; Latency: {latency_sec:.3f} seconds")


        # --- EMERGENCY HANDLING & FORWARDING TO WEBSOCKETS ---
        if status == "EMERGENCY":
            print(f"!!! EMERGENCY from {device_id} !!!")
            
            # Extract necessary fields for the alert
            last_fix_time_sec = decrypted_msg[6]
            latitude = decrypted_msg[4]
            longitude = decrypted_msg[5]
            
            # Format the alert data as a JSON object
            alert_data = {
                "type": "emergency_alert",
                "device_id": device_id,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "last_fix_sec": float(last_fix_time_sec),
                "timestamp": timestamp
            }
                
            # Acquire the lock to safely access the set of active WebSockets
            async with ws_lock:
                websockets_copy = set(active_websockets)

            # Broadcast the emergency alert to all connected WebSockets concurrently
            coros = [ws.send_json(alert_data) for ws in websockets_copy]
            results = await asyncio.gather(*coros, return_exceptions=True)

            # After broadcasting, check for any connection errors (e.g., disconnection)
            async with ws_lock:
                for ws, res in zip(websockets_copy, results):
                    if isinstance(res, Exception):
                        # If sending failed, assume the WebSocket is disconnected and remove it
                        active_websockets.discard(ws)
                        print(f"Cleaned up disconnected WebSocket: {ws}")


        # --- Log decrypted message ---
        async with file_lock:
            log_fields = decrypted_msg[:8] 
            record = f"{timestamp},{','.join(log_fields)}\n"
            print("Decrypted Record:", record)
            # Asynchronously append the record to the data CSV file
            async with aiofiles.open("data.csv", mode="a") as f:
                await f.write(record)

    except Exception as e:
        print("Decryption failed:", e)

# Handler for MQTT disconnection
@fast_mqtt.on_disconnect()
def disconnect(client: MQTTClient, packet, exc=None):
    print("Disconnected")

# Handler for MQTT subscription
@fast_mqtt.on_subscribe()
def subscribe(client: MQTTClient, mid: int, qos: int, properties: Any):
    print("subscribed", client, mid, qos, properties)

#####################################################################################
############################## FrontEnd Dashboard APIs ##############################
#####################################################################################

# --- Configuration ---
# Set the stream delay for the continuous "live" feed.
LIVE_STREAM_DELAY = 1.0

# --- Helper Functions ---

def compute_heat_index(temp, humidity):
    """
    Compute heat index using NOAA formula (simplified).
    T is temperature in Fahrenheit, R is relative humidity (percentage).
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

def load_data_raw(path: str, last_timestamp: datetime.datetime | None = None) -> pd.DataFrame:
    """
    Load only new sensor data from CSV (since last_timestamp), merge with user info,
    compute heat index, and return a clean DataFrame. (Used in old/initial WS logic)
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    
    # Load CSV data
    df = pd.read_csv(path, parse_dates=['datetime'])
    
    # Filter only rows with a timestamp newer than the last streamed record
    if last_timestamp:
        df = df[df['datetime'] > last_timestamp]
    
    if df.empty:
        return df

    # Assign column names based on the expected CSV structure
    expected_payload_cols = ['unique_id', 'temp', 'humidity', 'decibels', 
                             'latitude', 'longitude', 'last_fix', 'status']
    if df.shape[1] == len(expected_payload_cols) + 1:
        current_cols = ['datetime'] + expected_payload_cols
        df.columns = current_cols

    # Load device user mapping from device_log.csv
    user_df = None
    if os.path.exists("device_log.csv"):
        try:
            # Skip header line and provide column names
            user_df = pd.read_csv("device_log.csv", skiprows=1, names=['unique_id', 'username', 'key'])
        except Exception:
            user_df = None
    
    # Merge sensor data with user info based on unique_id
    if user_df is not None and not user_df.empty:
        df = df.merge(user_df[['unique_id', 'username']], on='unique_id', how='left')
    
    # Use unique_id as a fallback for missing usernames
    if 'username' not in df.columns or df['username'].isnull().any():
        df['username'] = df['unique_id'].astype(str)

    # Compute heat index for all new rows
    df['heat_index'] = df.apply(lambda r: compute_heat_index(r['temp'], r['humidity']), axis=1)

    # Sort the data by timestamp
    df = df.sort_values(by='datetime').reset_index(drop=True)
    
    return df

# --- FUNCTION FOR HISTORICAL DATA LOADING (No hardcoded time limit) ---
def load_historical_data(
    path: str, 
    device_ids: List[str] = None, 
    start_date: datetime.datetime = None, 
    end_date: datetime.datetime = None
) -> pd.DataFrame:
    """
    Loads ALL data and filters it based on custom time and device filters for historical view.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
        
    # Load all data from the CSV file
    df = pd.read_csv(path)
    
    # Re-map columns if reading raw CSV without headers
    expected_payload_cols = ['unique_id', 'temp', 'humidity', 'decibels', 'latitude', 'longitude', 'last_fix', 'status']
    if df.shape[1] == len(expected_payload_cols) + 1:
        current_cols = ['datetime'] + expected_payload_cols
        # Only rename if the expected names are not already present (a heuristic)
        if not all(c in df.columns for c in expected_payload_cols):
             df.columns = current_cols

    # Ensure the datetime column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # --- Apply Filters (These are the user-defined filters) ---
    if start_date:
        df = df[df['datetime'] >= start_date]
    if end_date:
        df = df[df['datetime'] <= end_date]
    if device_ids:
        df = df[df['unique_id'].isin(device_ids)]

    if df.empty:
        return df

    # 1. Load User Log (for merging)
    user_df = None
    if os.path.exists("device_log.csv"):
        try:
            # Skip the header
            user_df = pd.read_csv("device_log.csv", skiprows=1, names=['unique_id', 'username', 'key'])
        except Exception:
            user_df = None
        
    # 2. Merge dataframes
    if user_df is not None and not user_df.empty:
        df = df.merge(user_df[['unique_id', 'username']], on='unique_id', how='left')
    
    # 3. Handle missing usernames
    if 'username' not in df.columns or df['username'].isnull().any():
        if 'username' not in df.columns:
             df['username'] = df['unique_id'].astype(str)
        else:
             # Fill NaN usernames with their unique_id
             df['username'] = df['username'].fillna(df['unique_id'].astype(str))
        
    # 4. Final processing and sorting
    df['heat_index'] = compute_heat_index(df['temp'], df['humidity'])
    df = df.sort_values(by='datetime').dropna(subset=['username']).reset_index(drop=True)
    
    return df

def parse_line_to_record(line: str):
    """
    Parse a single CSV line (from data.csv) into the exact JSON-friendly dictionary 
    the frontend expects, including calculated fields (heat_index) and username lookup.
    """
    line = (line or "").strip()
    if not line:
        return None

    parts = line.split(",")
    # Check for the expected number of fields: timestamp + 8 data fields
    if len(parts) < 9:
        return None

    ts = parts[0]
    try:
        # Attempt to parse timestamp with microseconds
        dt = datetime.datetime.fromisoformat(ts)
        dt_iso = dt.isoformat()
    except Exception:
        # fallback: try parsing without microseconds
        try:
            dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            dt_iso = dt.isoformat()
        except Exception:
            # Cannot parse the datetime, discard the line
            return None

    try:
        # Convert data fields to their correct types (float/str)
        unique_id = parts[1]
        temp = float(parts[2])
        humidity = float(parts[3])
        decibels = float(parts[4])
        latitude = float(parts[5])
        longitude = float(parts[6])
        last_fix = float(parts[7])
        status = str(parts[8])
    except Exception:
        # Data conversion failed, discard the line
        return None

    # compute heat index
    heat_index = compute_heat_index(temp, humidity)

    # lookup username from cache (fall back to unique_id)
    username = device_user_map.get(str(unique_id), str(unique_id))

    # Construct the final record dictionary for JSON serialization
    rec = {
        "datetime": dt_iso,
        "unique_id": unique_id,
        "username": username,
        "temp": temp,
        "humidity": humidity,
        "heat_index": float(heat_index),
        "decibels": decibels,
        "latitude": latitude,
        "longitude": longitude,
        "last_fix": last_fix,
        "status": status,
    }
    return rec

# Username cache: maps device unique_id (str) to username (str)
device_user_map: Dict[str, str] = {}

def load_device_user_map():
    """Load all device unique_id and username mappings from device_log.csv into a cache."""
    global device_user_map
    device_user_map = {}
    if os.path.exists("device_log.csv"):
        try:
            # Load and populate the map
            df_users = pd.read_csv("device_log.csv", skiprows=1, names=['unique_id','username','key'])
            for _, r in df_users.iterrows():
                # Use username if present, otherwise use unique_id
                device_user_map[str(r['unique_id'])] = str(r['username']) if not pd.isna(r['username']) else str(r['unique_id'])
        except Exception as e:
            print("Failed to load device_log.csv for username mapping:", e)

# Initialize the username cache on startup
load_device_user_map()

# Simple HTML endpoint for testing (not used by the main dashboard)
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI WebSocket</title>
    </head>
    <body>
        <h1>WebSocket Real-Time Data Stream</h1>
        <p id="data_output">Waiting for data...</p>
        <a href="/dashboard/monitoring">Monitoring Dashboard</a><br>
        <a href="/dashboard/historical">Historical Dashboard</a>
        <script>
            // Determine the correct WebSocket protocol (ws: for http:, wss: for https:)
            const ws_protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            // Use the current host
            const ws_host = window.location.host;
            // Construct the full WebSocket URL
            var ws = new WebSocket(ws_protocol + '//' + ws_host + '/ws/data');

            ws.onmessage = function(event) {
                var messages = document.getElementById('data_output')
                try {
                    var data = JSON.parse(event.data);
                    
                    if (data.type === "history" || data.type === "live") {
                        // For the simple display, we only care about the latest single data point
                        let latest_data = data.type === "history" ? data.data[data.data.length - 1] : data.data;

                        messages.innerHTML = "Last Update: " + new Date().toLocaleTimeString() + "<br>" +
                                             "Device: " + latest_data.username + " (" + latest_data.unique_id + ")<br>" +
                                             "Decibels (dB): " + latest_data.decibels.toFixed(2) + "<br>" +
                                             "Heat Index: " + latest_data.heat_index.toFixed(2);
                    } else if (data.type === "status") {
                        messages.innerHTML = "Status: " + data.message + "... Waiting for next update.";
                    } else if (data.type === "emergency_alert") {
                         messages.innerHTML = "!!! EMERGENCY ALERT from " + data.device_id + " !!!";
                    }
                } catch (e) {
                    console.error("Error parsing message or processing data:", e, event.data);
                    messages.innerHTML = "Error receiving data.";
                }
            };
            
            ws.onclose = function() {
                document.getElementById('data_output').innerHTML = "Connection closed. Waiting for server restart...";
            };
            
            ws.onerror = function(error) {
                console.error("WebSocket Error:", error);
                document.getElementById('data_output').innerHTML = "WebSocket error. Check server logs.";
            };

        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

# Endpoint to serve the historical dashboard HTML file
@app.get("/dashboard/historical")
async def get_historical_page():
    try:
        # Explicitly specify encoding="utf-8"
        with open("dashboard_historical.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: dashboard_historical.html not found.</h1>", status_code=404)

# Endpoint to serve the monitoring dashboard HTML file
@app.get("/dashboard/monitoring")
async def get_monitoring_page():
    try:
        # Explicitly specify encoding="utf-8"
        with open("dashboard_monitoring.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: dashboard_monitoring.html not found.</h1>", status_code=404)

# Endpoint to serve the device list dashboard HTML file
@app.get("/dashboard/list")
async def get_device_list_page():
    # This assumes the new file is named dashboard_list.html
    try:
        with open("dashboard_list.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: dashboard_list.html not found.</h1>", status_code=404)

# --- NEW API ENDPOINT: Get available devices ---
@app.get("/api/devices")
async def get_available_devices():
    """Returns a list of available devices (ID and Username) from device_log.csv."""
    user_df = None
    if os.path.exists("device_log.csv"):
        try:
            # Load user data, skipping header row
            user_df = pd.read_csv("device_log.csv", skiprows=1, names=['unique_id', 'username', 'key'])
            # Convert to a list of dictionaries with only unique_id and username
            return user_df[['unique_id', 'username']].to_dict(orient='records')
        except Exception as e:
            print(f"Error loading device_log.csv: {e}")
            return []
    return []

# --- NEW API ENDPOINT: Get historical data with filters ---
@app.get("/api/history")
async def get_historical_data(
    # Query parameter for comma-separated device IDs
    device_ids: str | None = None,
    # Query parameter for start datetime (ISO format)
    start: str | None = None,
    # Query parameter for end datetime (ISO format)
    end: str | None = None
):
    """Retrieves filtered sensor data for the historical dashboard."""
    path = "data.csv"
    if not os.path.exists(path):
        return []

    # Parse and validate inputs
    parsed_ids = device_ids.split(',') if device_ids else None
    
    try:
        # Convert ISO 8601 string inputs to datetime objects
        parsed_start = datetime.datetime.fromisoformat(start) if start else None
        parsed_end = datetime.datetime.fromisoformat(end) if end else None
    except ValueError as e:
        print(f"Date parsing error: {e}")
        return []

    # Load and filter the data using the helper function
    df = load_historical_data(path, parsed_ids, parsed_start, parsed_end)
    
    # Prepare data for JSON response (ensure correct types for frontend)
    history_list: List[Dict[str, Any]] = df.to_dict(orient='records')
    for record in history_list:
        # Convert datetime object back to ISO string for JSON
        record['datetime'] = record['datetime'].isoformat()
        # Explicit type conversion for safe JSON serialization
        record['username'] = str(record['username']) 
        record['decibels'] = float(record['decibels'])
        record['heat_index'] = float(record['heat_index'])
        record['temp'] = float(record['temp'])
        record['humidity'] = float(record['humidity'])
        record['latitude'] = float(record['latitude'])
        record['longitude'] = float(record['longitude'])
        record['last_fix'] = float(record['last_fix'])
        record['status'] = str(record['status']) 
        
    return history_list

# WebSocket endpoint for real-time data streaming and emergency alerts
@app.websocket("/ws/data")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Register the new WebSocket for receiving emergency alerts
    async with ws_lock:
        active_websockets.add(ws)

    file_path = "data.csv"
    position = 0  # File pointer position for tailing
    initialized = False # Flag to track if initial history has been sent

    try:
        while True:
            # ---------- FIRST LOAD: send only last 1 hour of history ----------
            if not initialized:
                if not os.path.exists(file_path):
                    await asyncio.sleep(1)
                    continue

                try:
                    # Read the entire CSV file for initial history
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print("Failed to read data.csv for initial history:", e)
                    await asyncio.sleep(1)
                    continue

                # Ensure column names are set correctly before processing
                if "datetime" not in df.columns:
                    expected_payload_cols = ['unique_id','temp','humidity','decibels','latitude','longitude','last_fix','status']
                    if df.shape[1] == len(expected_payload_cols) + 1:
                         df.columns = ['datetime'] + expected_payload_cols

                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                except Exception:
                    df = pd.DataFrame() # Clear dataframe if datetime parsing fails

                if not df.empty:
                    # Filter data to the last hour
                    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
                    df = df[df['datetime'] >= one_hour_ago]
                    df = df.sort_values(by='datetime')

                    history_records = []
                    # Process the historical data to match the expected record structure
                    for _, row in df.iterrows():
                        rec = {
                            "datetime": row['datetime'].isoformat(),
                            "unique_id": str(row['unique_id']),
                            "username": device_user_map.get(str(row['unique_id']), str(row['unique_id'])),
                            "temp": float(row['temp']),
                            "humidity": float(row['humidity']),
                            "heat_index": float(compute_heat_index(row['temp'], row['humidity'])),
                            "decibels": float(row['decibels']),
                            "latitude": float(row['latitude']),
                            "longitude": float(row['longitude']),
                            "last_fix": float(row['last_fix']),
                            "status": str(row['status'])
                        }
                        history_records.append(rec)
                else:
                    history_records = []

                # Move file pointer to the end of the file to start tailing fresh lines
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        f.seek(0, os.SEEK_END)
                        position = f.tell()
                except Exception:
                    position = 0

                # Send the initial one-hour history to the client
                await ws.send_json({"type": "history", "data": history_records})
                print(f"[WS] Sent history: {len(history_records)} records (last 1 hour)")
                initialized = True
                # Short pause after initialization
                await asyncio.sleep(0.2)
                continue

            # ---------- LIVE: tail file for appended lines ----------
            try:
                # Open the file and seek to the last read position
                with open(file_path, "r", encoding="utf-8") as f:
                    f.seek(position)
                    new_lines = f.readlines()
                    # Update the position to the new end of the file
                    position = f.tell()
            except FileNotFoundError:
                await asyncio.sleep(1)
                continue
            except Exception as e:
                print("Error tailing file:", e)
                await asyncio.sleep(1)
                continue

            # Process and stream each new line
            for line in new_lines:
                # Parse line, compute heat index, and lookup username
                rec = parse_line_to_record(line)
                if not rec:
                    continue
                # Stream the new live record
                await ws.send_json({"type": "live", "data": rec})

            # Wait for the next stream interval
            await asyncio.sleep(max(0.1, LIVE_STREAM_DELAY))

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket stream error: {e}")

    finally:
        # Clean up: remove the WebSocket from the active set
        async with ws_lock:
            active_websockets.discard(ws)
        try:
            # Ensure the connection is properly closed if it hasn't been already
            if ws.client_state not in {WebSocketState.DISCONNECTED}:
                await ws.close()
        except RuntimeError:
            pass


    
# Endpoint to serve the favicon.ico file
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("favicon.ico")