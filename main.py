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

mqtt_config = MQTTConfig(
    host="mqtt-dashboard.com",
    port=1883
)
fast_mqtt = FastMQTT(config=mqtt_config)
device_keys = {}

# Global set to hold active WebSocket connections for broadcasting alerts
active_websockets: set[WebSocket] = set() 

file_lock = asyncio.Lock()
ws_lock = asyncio.Lock()

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await fast_mqtt.mqtt_startup()
    yield
    await fast_mqtt.mqtt_shutdown()


app = FastAPI(lifespan=_lifespan)


# Create NTP client
ntpclient = ntplib.NTPClient()

@fast_mqtt.on_connect()
def connect(client: MQTTClient, flags: int, rc: int, properties: Any):
    client.subscribe("hazard-monitoring/server")
    print("Connected: ", client, flags, rc, properties)
    
    # --- FIX: Skip the first artifact line when reading device_log.csv ---
    with open("device_log.csv", mode="r") as f:
        # Skip the first line containing the artifact/comment
        next(f) 
        # Read the rest of the file
        reader = csv.DictReader(f, fieldnames=['unique_id', 'username', 'key'])
        for row in reader:
            device_keys[row["unique_id"]] = bytes.fromhex(row["key"])
    # ---------------------------------------------------------------------

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

        if device_id not in device_keys:
            print(f"Unknown device ID: {device_id}")
            return

        # --- Decode Base64 ---
        nonce = base64.b64decode(nonce_b64)
        ciphertext = base64.b64decode(ciphertext_b64)
        tag = base64.b64decode(tag_b64)

        # --- Decrypt ---
        key = device_keys[device_id]
        chacha = ChaCha20Poly1305(key)
        # Combine ciphertext and tag for decryption
        plaintext = chacha.decrypt(nonce, ciphertext + tag, None) 
        
        ## Expected payload: client_id,avg_t,avg_h,avg_spl,lat,lng,fix_age,status,timestamp
        decrypted_msg = plaintext.decode().split(",")

        # The device now sends 9 fields 
        if len(decrypted_msg) != 9:
            print("Decryption successful, but invalid decrypted field count:", len(decrypted_msg), "Expected 9.")
            return

        if decrypted_msg[0] != device_id:
            print("Device ID mismatch", device_id, decrypted_msg[0])
            return
        
        status = decrypted_msg[7] # The new 8th field is the status
        timestamp = str(datetime.datetime.strptime(decrypted_msg[8], "%Y-%m-%d %H:%M:%S.%f"))[:-7]  # new timestamp field

        # Compute latency in seconds (float)
        # ntp_time = datetime.datetime.fromtimestamp(ntpclient.request('pool.ntp.org').tx_time)
        # latency_sec = max(0,(ntp_time - datetime.datetime.strptime(decrypted_msg[8], "%Y-%m-%d %H:%M:%S.%f")).total_seconds())
        # print(f"Message Received from {device_id}; Latency: {latency_sec:.3f} seconds")


        # --- EMERGENCY HANDLING & FORWARDING TO WEBSOCKETS ---
        if status == "EMERGENCY":
            print(f"!!! EMERGENCY from {device_id} !!!")
            
            # Extract info for the alert
            last_fix_time_sec = decrypted_msg[6]
            latitude = decrypted_msg[4]
            longitude = decrypted_msg[5]
            
            # Format the alert for the alert
            alert_data = {
                "type": "emergency_alert",
                "device_id": device_id,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "last_fix_sec": float(last_fix_time_sec),
                "timestamp": timestamp
            }
                   
            async with ws_lock:
                websockets_copy = set(active_websockets)

            coros = [ws.send_json(alert_data) for ws in websockets_copy]
            results = await asyncio.gather(*coros, return_exceptions=True)

            async with ws_lock:
                for ws, res in zip(websockets_copy, results):
                    if isinstance(res, Exception):
                        active_websockets.discard(ws)
                        print(f"Cleaned up disconnected WebSocket: {ws}")



        # --- Log decrypted message ---
        # The log now contains 8 fields: client_id,avg_t,avg_h,avg_spl,payloadLat,payloadLng,payloadSecondsSinceFix,status
        async with file_lock:
            decrypted_msg = decrypted_msg[:8]  # Exclude the timestamp for logging
            record = f"{timestamp},{','.join(decrypted_msg)}\n"
            print("Decrypted Record:", record)
            async with aiofiles.open("data.csv", mode="a") as f:
                await f.write(record)

    except Exception as e:
        print("Decryption failed:", e)

@fast_mqtt.on_disconnect()
def disconnect(client: MQTTClient, packet, exc=None):
    print("Disconnected")

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
    """Compute heat index using NOAA formula (simplified)."""
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
    compute heat index, and return a clean DataFrame.
    
    Args:
        path: Path to data.csv
        last_timestamp: Datetime of the last streamed record (optional)
    
    Returns:
        pd.DataFrame with columns:
        ['datetime', 'unique_id', 'username', 'temp', 'humidity', 'decibels', 
         'latitude', 'longitude', 'last_fix', 'status', 'heat_index']
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    
    # Load CSV incrementally
    df = pd.read_csv(path, parse_dates=['datetime'])
    
    # If last_timestamp is provided, filter only newer rows
    if last_timestamp:
        df = df[df['datetime'] > last_timestamp]
    
    if df.empty:
        return df

    # ASSUMING COLUMN NAMES based on device payload
    expected_payload_cols = ['unique_id', 'temp', 'humidity', 'decibels', 
                             'latitude', 'longitude', 'last_fix', 'status']
    if df.shape[1] == len(expected_payload_cols) + 1:
        current_cols = ['datetime'] + expected_payload_cols
        df.columns = current_cols

    # Load user log once
    user_df = None
    if os.path.exists("device_log.csv"):
        try:
            user_df = pd.read_csv("device_log.csv", skiprows=1, names=['unique_id', 'username', 'key'])
        except Exception:
            user_df = None
    
    # Merge with user info
    if user_df is not None and not user_df.empty:
        df = df.merge(user_df[['unique_id', 'username']], on='unique_id', how='left')
    
    # Fill missing usernames
    if 'username' not in df.columns or df['username'].isnull().any():
        df['username'] = df['unique_id'].astype(str)

    # Compute heat index only for new rows
    df['heat_index'] = df.apply(lambda r: compute_heat_index(r['temp'], r['humidity']), axis=1)

    # Sort by datetime
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
    
    expected_payload_cols = ['unique_id', 'temp', 'humidity', 'decibels', 'latitude', 'longitude', 'last_fix', 'status']
    if df.shape[1] == len(expected_payload_cols) + 1:
        current_cols = ['datetime'] + expected_payload_cols
        if not all(c in df.columns for c in expected_payload_cols):
             df.columns = current_cols

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

    # 1. Load User Log
    user_df = None
    if os.path.exists("device_log.csv"):
        try:
            # --- FIX: Skip the artifact line
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
             df['username'] = df['username'].fillna(df['unique_id'].astype(str))
        
    # 4. Final processing and sorting
    df['heat_index'] = compute_heat_index(df['temp'], df['humidity'])
    df = df.sort_values(by='datetime').dropna(subset=['username']).reset_index(drop=True)
    
    return df

def parse_line_to_record(line: str):
    """
    Parse a CSV line (from data.csv) into the exact JSON-friendly record
    the frontend expects. Returns None for malformed lines.
    Expected CSV line format:
      timestamp,unique_id,temp,humidity,decibels,latitude,longitude,last_fix,status
    """
    line = (line or "").strip()
    if not line:
        return None

    parts = line.split(",")
    if len(parts) < 9:
        return None

    ts = parts[0]
    try:
        dt = datetime.datetime.fromisoformat(ts)
        dt_iso = dt.isoformat()
    except Exception:
        # fallback: try parsing without microseconds
        try:
            dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            dt_iso = dt.isoformat()
        except Exception:
            return None

    try:
        unique_id = parts[1]
        temp = float(parts[2])
        humidity = float(parts[3])
        decibels = float(parts[4])
        latitude = float(parts[5])
        longitude = float(parts[6])
        last_fix = float(parts[7])
        status = str(parts[8])
    except Exception:
        return None

    # compute heat index
    heat_index = compute_heat_index(temp, humidity)

    # lookup username from cache (fall back to unique_id)
    username = device_user_map.get(str(unique_id), str(unique_id))

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

# Username cache (load once)
device_user_map: Dict[str, str] = {}

def load_device_user_map():
    global device_user_map
    device_user_map = {}
    if os.path.exists("device_log.csv"):
        try:
            df_users = pd.read_csv("device_log.csv", skiprows=1, names=['unique_id','username','key'])
            for _, r in df_users.iterrows():
                device_user_map[str(r['unique_id'])] = str(r['username']) if not pd.isna(r['username']) else str(r['unique_id'])
        except Exception as e:
            print("Failed to load device_log.csv for username mapping:", e)

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

@app.get("/dashboard/historical")
async def get_historical_page():
    # FIX: Explicitly specify encoding="utf-8"
    try:
        with open("dashboard_historical.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: dashboard_historical.html not found.</h1>", status_code=404)

@app.get("/dashboard/monitoring")
async def get_monitoring_page():
    # FIX: Explicitly specify encoding="utf-8"
    try:
        with open("dashboard_monitoring.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: dashboard_monitoring.html not found.</h1>", status_code=404)

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
    """Returns a list of available devices from device_log.csv."""
    user_df = None
    if os.path.exists("device_log.csv"):
        try:
            # --- FIX APPLIED HERE: skiprows=1 and explicit column names
            user_df = pd.read_csv("device_log.csv", skiprows=1, names=['unique_id', 'username', 'key'])
            # Convert to list of dicts for JSON serialization
            return user_df[['unique_id', 'username']].to_dict(orient='records')
        except Exception as e:
            print(f"Error loading device_log.csv: {e}")
            return []
    return []

# --- NEW API ENDPOINT: Get historical data with filters ---
@app.get("/api/history")
async def get_historical_data(
    device_ids: str | None = None,
    start: str | None = None,
    end: str | None = None
):
    path = "data.csv"
    if not os.path.exists(path):
        return []

    # Parse and validate inputs
    parsed_ids = device_ids.split(',') if device_ids else None
    
    try:
        # FastAPI handles ISO 8601 string to datetime conversion
        parsed_start = datetime.datetime.fromisoformat(start) if start else None
        parsed_end = datetime.datetime.fromisoformat(end) if end else None
    except ValueError as e:
        print(f"Date parsing error: {e}")
        return []

    # load_historical_data loads ALL data and applies the query filters
    df = load_historical_data(path, parsed_ids, parsed_start, parsed_end)
    
    # Prepare data for JSON response
    history_list: List[Dict[str, Any]] = df.to_dict(orient='records')
    for record in history_list:
        record['datetime'] = record['datetime'].isoformat()
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

@app.websocket("/ws/data")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Register for emergency alerts
    async with ws_lock:
        active_websockets.add(ws)

    file_path = "data.csv"
    position = 0
    initialized = False

    try:
        while True:
            # ---------- FIRST LOAD: send only last 1 hour ----------
            if not initialized:
                if not os.path.exists(file_path):
                    await asyncio.sleep(1)
                    continue

                # Use pandas to filter last 1 hour (one-time)
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print("Failed to read data.csv for initial history:", e)
                    await asyncio.sleep(1)
                    continue

                # ensure column names if missing
                if "datetime" not in df.columns:
                    # best-effort rename if columns are raw
                    expected_payload_cols = ['unique_id','temp','humidity','decibels','latitude','longitude','last_fix','status']
                    if df.shape[1] == len(expected_payload_cols) + 1:
                        df.columns = ['datetime'] + expected_payload_cols

                # parse datetimes and filter last hour
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                except Exception:
                    # if parsing fails, skip sending history
                    df = pd.DataFrame()

                if not df.empty:
                    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
                    df = df[df['datetime'] >= one_hour_ago]
                    df = df.sort_values(by='datetime')

                    history_records = []
                    for _, row in df.iterrows():
                        # build record matching frontend expectations
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

                # move file pointer to end to start tailing fresh lines
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        f.seek(0, os.SEEK_END)
                        position = f.tell()
                except Exception:
                    position = 0

                # send the history (only last 1 hour)
                await ws.send_json({"type": "history", "data": history_records})
                print(f"[WS] Sent history: {len(history_records)} records (last 1 hour)")
                initialized = True
                # small pause to let frontend finish processing
                await asyncio.sleep(0.2)
                continue

            # ---------- LIVE: tail file for appended lines ----------
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.seek(position)
                    new_lines = f.readlines()
                    position = f.tell()
            except FileNotFoundError:
                await asyncio.sleep(1)
                continue
            except Exception as e:
                print("Error tailing file:", e)
                await asyncio.sleep(1)
                continue

            # send each new parsed line (parse_line_to_record returns full structure)
            for line in new_lines:
                rec = parse_line_to_record(line)
                if not rec:
                    continue
                await ws.send_json({"type": "live", "data": rec})

            # small sleep to avoid tight loop
            await asyncio.sleep(max(0.1, LIVE_STREAM_DELAY))

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket stream error: {e}")

    finally:
        async with ws_lock:
            active_websockets.discard(ws)
        try:
            if ws.client_state not in {WebSocketState.DISCONNECTED}:
                await ws.close()
        except RuntimeError:
            pass


    
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("favicon.ico")