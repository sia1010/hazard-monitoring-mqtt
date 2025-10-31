from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from gmqtt import Client as MQTTClient
from fastapi_mqtt import FastMQTT, MQTTConfig

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


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await fast_mqtt.mqtt_startup()
    yield
    await fast_mqtt.mqtt_shutdown()


app = FastAPI(lifespan=_lifespan)


@fast_mqtt.on_connect()
def connect(client: MQTTClient, flags: int, rc: int, properties: Any):
    client.subscribe("hazard-monitoring/server")  # subscribing mqtt topic
    print("Connected: ", client, flags, rc, properties)
    with open("device_log.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            device_keys[row["unique_id"]] = bytes.fromhex(row["key"])

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
        plaintext = chacha.decrypt(nonce, ciphertext + tag, None)
        decrypted_msg = plaintext.decode().split(",")

        if decrypted_msg[0] != device_id:
            print("Device ID mismatch", device_id, decrypted_msg[0])
            return
        
        if decrypted_msg[1] == "EMERGENCY":
            print(f"!!! EMERGENCY from {device_id} !!!")
            if decrypted_msg[6] != "-1":
                print(f"Last Known Location {decrypted_msg[6]} seconds ago at: https://www.google.com/maps?q={decrypted_msg[4]},{decrypted_msg[5]}")

        # --- Log decrypted message ---
        record = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{",".join(decrypted_msg)}\n"
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
    Load and preprocess the raw sensor data, merging with user info.
    Filters the loaded data to only include records from the current day.
    """
    df = pd.read_csv(path)

    # --- Filter data for the past 1 hour ---
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
            user_df = pd.read_csv("device_log.csv")
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

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    file_path = "data.csv"
    
    # State variable to track the last successfully streamed timestamp across reloads
    last_streamed_time: datetime.datetime | None = None 
    
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
            df_new = load_data_raw(file_path)
            
            # --- Determine which data to send (History or Delta) ---
            df_to_stream = pd.DataFrame()
            
            if df_new.empty:
                print("Reloaded file is empty for today. Waiting for data...")
                await asyncio.sleep(5)
                continue
            
            elif last_streamed_time is None:
                # FIRST LOAD: Send all data as history
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
                await websocket.send_json({"type": "history", "data": history_list})
                print(f"FIRST LOAD: Sent {len(history_list)} points as initial history.")
                
            else:
                # RELOAD/FILE CHANGE: Send only new data points (delta) as live updates
                # Filter df_new to get records strictly after the last streamed time
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
                        }
                    }
                    
                    await websocket.send_json(live_data)
                    
                    # Ensure the last_row Series is updated with the new timestamp for the next iteration
                    await asyncio.sleep(LIVE_STREAM_DELAY)
                
        except Exception as e:
            # Handle connection loss or other genuine errors
            print(f"WebSocket connection closed or error: {e}")
            break # Exit the outer loop on connection loss/error
            
    await websocket.close()