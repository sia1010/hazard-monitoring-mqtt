import csv
import os
import time
import base64
import random # Added for data simulation
import sys # Added for non-blocking input handling
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import constant_time
import paho.mqtt.client as mqtt
import datetime

# --- Configuration ---
MQTT_BROKER = "mqtt-dashboard.com"
MQTT_PORT = 1883
MQTT_TOPIC = "hazard-monitoring/server"
DEVICE_LOG_PATH = "device_log.csv"
PUBLISH_INTERVAL_SECONDS = 5 # Auto-publish every 5 seconds

# --- Data Structure Reference ---
# The plaintext payload MUST contain 8 fields, comma-separated:
# 1. client_id (unique_id from device_log)
# 2. avg_t (Average Temperature in °C)
# 3. avg_h (Average Humidity in %)
# 4. avg_spl (Average Sound Pressure Level in dB)
# 5. payloadLat (Latitude)
# 6. payloadLng (Longitude)
# 7. payloadSecondsSinceFix (Time since last GPS fix in seconds)
# 8. status (e.g., "OK", "WARNING", "EMERGENCY")

# --- Simulation Class ---

class DeviceSimulator:
    """Manages the state and generates the next set of simulated sensor readings."""
    def __init__(self, start_lat: float, start_lng: float, start_status: str, start_temp: str, start_humidity: str):
        # 1. Initialize environmental starting points (random within typical range)
        self.temp = start_temp
        self.humidity = start_humidity
        
        # 2. Set static/semi-static GPS data and status
        self.latitude = start_lat
        self.longitude = start_lng
        self.status = start_status
        
        # 3. Movement simulation parameters (small random drift)
        self.lat_drift = 0.0001
        self.lng_drift = 0.0001
        
        # 4. GPS Fix is assumed to be instant
        self.seconds_since_fix = 0.0
        
        print(f"Simulator Initialized: Lat={start_lat:.4f}, Lng={start_lng:.4f}, Status={start_status}")


    def generate_next_data_point(self, new_status: str = None) -> list:
        """Calculates the next data point with a small random drift."""
        
        if new_status:
            self.status = new_status
            
        # 1. Simulate environmental drift (small change based on last value)
        
        # Temp: fluctuation ± 0.5°C, bounds (20, 45)
        self.temp += random.uniform(-0.5, 0.5)
        self.temp = max(20.0, min(45.0, self.temp))
        
        # Humidity: fluctuation ± 2.0%, bounds (30, 99)
        self.humidity += random.uniform(-2.0, 2.0)
        self.humidity = max(30.0, min(99.0, self.humidity))
        
        self.decibels = np.random.normal(loc=55, scale=4) if np.random.random() <= 0.975 else np.random.normal(loc=65, scale=4) if np.random.random() <= 0.975 else np.random.uniform(70, 90)
        
        # 2. Simulate movement (random walk)
        self.latitude += random.uniform(-self.lat_drift, self.lat_drift)
        self.longitude += random.uniform(-self.lng_drift, self.lng_drift)
        
        # 3. Prepare data fields
        data_fields = [
            round(self.temp, 2),
            round(self.humidity, 2),
            round(self.decibels, 2),
            round(self.latitude, 6),
            round(self.longitude, 6),
            round(self.seconds_since_fix, 1),
            self.status,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        ]
        
        return data_fields

# --- Core Functions ---

def load_device_keys(file_path):
    """Loads device IDs and their encryption keys from device_log.csv."""
    device_keys = {}
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Cannot load keys.")
        return device_keys

    try:
        with open(file_path, mode="r", newline='') as f:
            # Skip the first artifact line
            next(f)
            reader = csv.DictReader(f, fieldnames=['unique_id', 'username', 'key'])
            for row in reader:
                device_keys[row["unique_id"]] = bytes.fromhex(row["key"].strip())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
        
    print(f"Successfully loaded keys for {len(device_keys)} devices.")
    return device_keys

def create_encrypted_payload(device_id: str, key: bytes, data_fields: list) -> bytes | None:
    """
    Encrypts the sensor data using the device's key and returns the 
    MQTT payload string: device_id,nonce_b64,ciphertext_b64,tag_b64
    """
    
    # 1. Construct the plaintext message
    plaintext_msg = f"{device_id},{','.join(map(str, data_fields))}"
    plaintext = plaintext_msg.encode('utf-8')
    
    # 2. Initialize ChaCha20Poly1305
    try:
        chacha = ChaCha20Poly1305(key)
    except Exception as e:
        print(f"Encryption Error: Invalid key format for {device_id}. {e}")
        return None
        
    # 3. Generate a 12-byte unique nonce
    nonce = os.urandom(12)
    
    # 4. Encrypt the data
    ciphertext_with_tag = chacha.encrypt(nonce, plaintext, None)
    
    # 5. Separate ciphertext and tag (ChaCha20Poly1305 tag is 16 bytes)
    ciphertext = ciphertext_with_tag[:-16]
    tag = ciphertext_with_tag[-16:]
    
    # 6. Base64 encode the components
    nonce_b64 = base64.b64encode(nonce).decode('utf-8')
    ciphertext_b64 = base64.b64encode(ciphertext).decode('utf-8')
    tag_b64 = base64.b64encode(tag).decode('utf-8')
    
    # 7. Format the final message payload
    mqtt_message_payload = f"{device_id},{nonce_b64},{ciphertext_b64},{tag_b64}"
    
    return mqtt_message_payload.encode('utf-8')

def on_connect(client, userdata, flags, rc):
    """Callback for when the client receives a CONNACK response from the server."""
    if rc == 0:
        print("Connected successfully to MQTT broker.")
    else:
        print(f"Failed to connect, return code {rc}")

def on_publish(client, userdata, mid):
    """Callback for when a message is successfully published."""
    # print(f"Message ID {mid} published.") # Disabled for cleaner auto-publish output
    pass

# --- Input for Initial Parameters ---

def get_initial_params():
    """Prompts user for initial GPS and status, or uses defaults."""
    print("\n--- Set Initial Simulation Parameters ---")
    
    # Default location in Malaysia (close to where the monitoring dashboard example uses)
    default_lat = 4.388000
    default_lng = 100.966000
    default_temp = random.uniform(25.0, 35.0)
    default_humidity = random.uniform(50.0, 80.0)
    
    while True:
        try:
            print(f"Default Location: {default_lat}, {default_lng}")
            lat_input = input(f"1. Starting Latitude (Leave blank for default): ").strip() or str(default_lat)
            lng_input = input(f"2. Starting Longitude (Leave blank for default): ").strip() or str(default_lng)
            status = input("3. Starting Status (OK/WARNING/EMERGENCY, default=OK): ").strip().upper() or "OK"
            temp_input = input(f"4. Starting Temp (Leave blank for default): ").strip() or str(default_temp)
            humidity_input = input(f"5. Starting Humidity (Leave blank for default): ").strip() or str(default_humidity)


            payloadLat = float(lat_input)
            payloadLng = float(lng_input)
            payloadTemp = float(temp_input)
            payloadHumidity = float(humidity_input)
            
            if status not in ["OK", "WARNING", "EMERGENCY"]:
                print("Invalid status. Must be OK, WARNING, or EMERGENCY.")
                continue

            return payloadLat, payloadLng, status, payloadTemp, payloadHumidity

        except ValueError:
            print("Invalid number format. Please ensure Latitude/Longitude are numbers.")
        except Exception as e:
            print(f"An unexpected error occurred during input: {e}")

# --- Main Execution ---

def main():
    """Main loop for device selection, initialization, and automated publishing."""
    device_keys = load_device_keys(DEVICE_LOG_PATH)
    if not device_keys:
        return

    device_ids = list(device_keys.keys())

    # --- 1. Select Device ---
    print("\nAvailable Device IDs:")
    for i, device_id in enumerate(device_ids):
        print(f"[{i+1}] {device_id}")

    selected_device_id = None
    selected_key = None
    while True:
        try:
            selection = input(f"Select device number (1-{len(device_ids)}) or 'exit': ").strip()
            if selection.lower() == 'exit':
                return
            
            index = int(selection) - 1
            if 0 <= index < len(device_ids):
                selected_device_id = device_ids[index]
                selected_key = device_keys[selected_device_id]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'exit'.")

    print(f"\n--- Selected Device: {selected_device_id} ---")
    
    # --- 2. Get Initial Parameters and Initialize Simulator ---
    start_lat, start_lng, start_status, start_temp, start_hum = get_initial_params()
    simulator = DeviceSimulator(start_lat, start_lng, start_status, start_temp, start_hum)

    # --- 3. MQTT Client Setup ---
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start() 

    # --- 4. Auto-Publish Loop ---
    print("\n--- Auto-Publisher Started ---")
    print(f"Publishing data from {selected_device_id} every {PUBLISH_INTERVAL_SECONDS} seconds.")
    print("Action commands: 'STATUS' (change status), 'EXIT' (quit).")

    need_input = True

    while True:
        # Check for user input (non-blocking)
        try:
            # We use a short sleep/wait inside the loop to ensure we don't spam the console/CPU
            time.sleep(PUBLISH_INTERVAL_SECONDS) 
            
            # Non-blocking input check is complex in standard Python console, 
            # so we use a simple input prompt to allow interruption.
            if need_input:
                user_input = input(f"[{time.strftime('%H:%M:%S')}] Publishing... (Type command or press Enter to continue) ").strip().upper()
        except EOFError:
            # Handle script run environments that don't support interactive input
            user_input = "" 
        except Exception:
            user_input = "" 
            
        
        if user_input == 'EXIT':
            break
        elif user_input == 'REP':
            need_input = False
        elif user_input == 'STATUS':
            new_status = input("Enter NEW status (OK/EMERGENCY): ").strip().upper()
            if new_status in ["OK", "EMERGENCY"]:
                simulator.status = new_status
                print(f"Simulator status updated to {new_status}. Data will reflect this.")
            else:
                print("Invalid status. Continuing with current status.")
        
        # 5. Generate next data point
        data_fields = simulator.generate_next_data_point()

        # 6. Encrypt and Publish
        encrypted_payload = create_encrypted_payload(
            selected_device_id, 
            selected_key, 
            data_fields
        )

        if encrypted_payload:
            client.publish(MQTT_TOPIC, encrypted_payload, qos=1)
            print(f"[{time.strftime('%H:%M:%S')}] Published: Temp={data_fields[0]}°C, Noise={data_fields[2]}dB, Lat={data_fields[3]}, Status={data_fields[-1]}")
        else:
            print("Failed to create encrypted payload. Check device key.")


    client.loop_stop()
    client.disconnect()
    print("Publisher script finished.")


if __name__ == "__main__":
    # Initialize random seed for consistent simulation behavior
    random.seed(time.time()) 
    main()