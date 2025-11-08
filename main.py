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
import os

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
        
        # Expected payload: client_id,avg_t,avg_h,avg_spl,payloadLat,payloadLng,payloadSecondsSinceFix,status
        decrypted_msg = plaintext.decode().split(",")

        # The device now sends 8 fields (including the new status field)
        if len(decrypted_msg) != 8:
            print("Decryption successful, but invalid decrypted field count:", len(decrypted_msg), "Expected 8.")
            return

        if decrypted_msg[0] != device_id:
            print("Device ID mismatch", device_id, decrypted_msg[0])
            return
        
        status = decrypted_msg[7] # The new 8th field is the status

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
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Broadcast to all connected clients
            disconnected_websockets = set()
            
            async with ws_lock:
                for ws in active_websockets:
                    try:
                        # Send the immediate alert message
                        await ws.send_json(alert_data) 
                    except Exception:
                        # Collect disconnected websockets for removal
                        disconnected_websockets.add(ws)
                
                # Clean up disconnected websockets
                for ws in disconnected_websockets:
                    active_websockets.remove(ws)
                    print(f"Cleaned up disconnected WebSocket: {ws}")


        # --- Log decrypted message ---
        # The log now contains 8 fields: client_id,avg_t,avg_h,avg_spl,payloadLat,payloadLng,payloadSecondsSinceFix,status
        async with file_lock:
            record = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{','.join(decrypted_msg)}\n"
            with open("data.csv", mode="a") as f:
                f.write(record)
        
        print(f"[{device_id}] {",".join(decrypted_msg)}")

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

def load_data_raw(path: str) -> pd.DataFrame:
    """
    Load ALL raw sensor data, merge with user info, and then filter 
    to ONLY the past 1 hour for the live WebSocket dashboard.
    """
    # Load all data from the CSV file
    df = pd.read_csv(path)
    
    # ASSUMING COLUMN NAMES based on device payload and common practice
    expected_payload_cols = ['unique_id', 'temp', 'humidity', 'decibels', 'latitude', 'longitude', 'last_fix', 'status']
    if df.shape[1] == len(expected_payload_cols) + 1:
        current_cols = ['datetime'] + expected_payload_cols
        if not all(c in df.columns for c in expected_payload_cols):
             df.columns = current_cols
    
    # --- Filter data to the past 1 hour (Live Dashboard requirement) ---
    df['datetime'] = pd.to_datetime(df['datetime'])
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    df = df[df['datetime'] >= one_hour_ago]
        
    if df.empty:
        print(f"WARNING: No sensor data found since ({one_hour_ago}). Returning empty DataFrame.")
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
        # Ensure all necessary fields are converted to native Python types for JSON serialization
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
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Add to the active set
    async with ws_lock:
        active_websockets.add(websocket)
    
    file_path = "data.csv"
    
    # State variable to track the last successfully streamed timestamp across reloads
    last_streamed_time: datetime.datetime | None = None 
    
    try:
        # Outer loop to handle persistent streaming and file reloads
        while True:
            # Check if the data file exists before trying to load
            if not os.path.exists(file_path):
                print(f"ERROR: Data file not found at {file_path}. Waiting 5 seconds...")
                await asyncio.sleep(5)
                continue
                
            # Get modification time BEFORE load
            last_mtime = os.path.getmtime(file_path)
            
            try:
                # load_data_raw loads ALL data, but returns only the latest 1 hour
                df_new = load_data_raw(file_path)
                
                # --- Determine which data to send (History or Delta) ---
                df_to_stream = pd.DataFrame()
                
                if df_new.empty:
                    # FIX: Send status message and wait for a long period (60s) before re-checking
                    print("No recent data found (last 1 hour). Sending status to client and pausing file check.")
                    await websocket.send_json({"type": "status", "message": "No data found in the last 1 hour. Checking again in 60s."})
                    await asyncio.sleep(60) 
                    continue # Restart the file check loop
                
                elif last_streamed_time is None:
                    # FIRST LOAD: Send all filtered data (max 1 hour of data) as history
                    df_to_stream = df_new
                    
                    # Format and send history
                    history_list: List[Dict[str, Any]] = df_to_stream.to_dict(orient='records')
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
                    await websocket.send_json({"type": "history", "data": history_list})
                    print(f"FIRST LOAD: Sent {len(history_list)} points as initial history (last 1 hour).")
                    
                else:
                    # RELOAD/FILE CHANGE: Send only new data points (delta) as live updates
                    # Filter df_new to get records strictly after the last streamed time
                    # Note: df_new is already limited to the last 1 hour
                    df_to_stream = df_new[df_new['datetime'] > last_streamed_time]
                    
                    # Stream delta points individually
                    for _, row in df_to_stream.iterrows():
                        data_to_send: Dict[str, Any] = {
                            "type": "live", 
                            "data": {
                                "datetime": row["datetime"].isoformat(), 
                                "unique_id": row["unique_id"],
                                "username": str(row["username"]),
                                "decibels": float(row["decibels"]),
                                "heat_index": float(row["heat_index"]),
                                "temp": float(row["temp"]),
                                "humidity": float(row["humidity"]),
                                "latitude": float(row["latitude"]),
                                "longitude": float(row["longitude"]),
                                "last_fix": float(row["last_fix"]),
                                "status": str(row["status"]),
                            }
                        }
                        await websocket.send_json(data_to_send)
                        await asyncio.sleep(LIVE_STREAM_DELAY) 
                    
                    print(f"RELOAD DETECTED: Sent {len(df_to_stream)} new points as live updates.")

                # Update last_streamed_time *after* sending, using the maximum datetime from the entire newly loaded dataset
                if not df_new.empty:
                     last_streamed_time = df_new['datetime'].max()
                
                # --- STEP 2: Start/Continue Simulation ---
                
                # Get the last row from the complete, current file load
                last_row = df_new.iloc[-1].copy() if not df_new.empty else None
                
                if last_row is not None:
                    current_datetime = pd.to_datetime(last_row["datetime"])
                    static_hi = compute_heat_index(last_row["temp"], last_row["humidity"])
                    
                    while True:
                        # Check for file updates
                        current_mtime = os.path.getmtime(file_path)
                        if current_mtime > last_mtime:
                            print(f"Detected file change in {file_path}. Restarting stream and reloading data.")
                            await websocket.send_json({"type": "status", "message": "reloading"})
                            break # Break the inner simulation loop, causing the outer loop to restart
                        
                        live_data: Dict[str, Any] = {
                            "type": "live",
                            "data": {
                                "datetime": last_row["datetime"].isoformat(), 
                                "unique_id": last_row["unique_id"],
                                "username": str(last_row["username"]),
                                "decibels": float(last_row["decibels"]),
                                "heat_index": float(static_hi),
                                "temp": float(last_row["temp"]),
                                "humidity": float(last_row["humidity"]),
                                "latitude": float(last_row["latitude"]),
                                "longitude": float(last_row["longitude"]),
                                "last_fix": float(last_row["last_fix"]),
                                "status": str(row["status"]),
                            }
                        }
                        
                        await websocket.send_json(live_data)
                        
                        await asyncio.sleep(LIVE_STREAM_DELAY)
                    
            except WebSocketDisconnect:
                raise # Re-raise to be caught by the outer block
            except Exception as e:
                # Handle connection loss or other genuine errors
                print(f"Error during stream or file load: {e}")
                await asyncio.sleep(5) # Wait before attempting file reload
                
    except WebSocketDisconnect:
        print(f"WebSocket {websocket} disconnected gracefully.")
    except Exception as e:
        print(f"WebSocket connection closed or error: {e}")
    finally:
        if websocket in active_websockets:
            async with ws_lock:
                active_websockets.remove(websocket)

        # ✅ Only close if not already closed
        if websocket.client_state not in {WebSocketState.DISCONNECTED}:
            try:
                await websocket.close()
            except RuntimeError:
                pass
    
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("favicon.ico")