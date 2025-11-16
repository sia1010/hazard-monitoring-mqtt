import tkinter as tk
from tkinter import ttk, messagebox
import serial
import threading
import time
import secrets
import csv
import os

# ===================== SERIAL CONFIG =====================
DEFAULT_PORT = "COM3"
DEFAULT_NETWORK = "Potato"
DEFAULT_NETWORK_PASS = "kfcyummy"
DEFAULT_MQTT_SERVER = "mqtt-dashboard.com"
BAUD = 115200

ser = None


# ===================== SERIAL THREAD =====================
def listen_serial(text_widget):
    # Lines that belong to config handshake
    CONFIG_PREFIXES = (
        "ACK:CONFIG",
        "READY:DATA",
        "OK:",
        "ERR:",
        "CONFIG:",
    )

    while ser and ser.is_open:
        try:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode(errors="ignore").strip()
            if not line:
                continue

            # --- FILTER HERE ---
            if not line.startswith(CONFIG_PREFIXES):
                continue  # ignore unrelated logs

            # Show only the relevant ones
            text_widget.insert(tk.END, f"ESP32: {line}\n")
            text_widget.see(tk.END)

        except:
            break


# ===================== SEND CONFIG =====================
def send_config(port_entry, ssid_entry, pass_entry, mqtt_entry, client_entry, key_entry, log_box):
    global ser

    port = port_entry.get().strip()
    ssid = ssid_entry.get().strip()
    passwd = pass_entry.get().strip()
    mqtt = mqtt_entry.get().strip()
    client_id = client_entry.get().strip()
    key_hex = key_entry.get().strip()

    if not all([ssid, passwd, mqtt, client_id, key_hex]):
        messagebox.showerror("Error", "Please fill in all fields.")
        return

    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        time.sleep(2)  # allow ESP32 to reboot
    except Exception as e:
        messagebox.showerror("Serial Error", str(e))
        return

    log_box.insert(tk.END, "Connecting to ESP32...\n")

    # Start serial listener thread
    threading.Thread(target=listen_serial, args=(log_box,), daemon=True).start()

    # Send config command
    ser.write(b"CMD:CONFIG\n")
    log_box.insert(tk.END, "Sent: CMD:CONFIG\n")

    # Wait for READY inside the background listener, then send DATA:
    def send_data():
        time.sleep(3.0)  # let ESP32 reach "READY:DATA"

        payload = f"{ssid},{passwd},{mqtt},{client_id},{key_hex}"
        ser.write(f"DATA:{payload}\n".encode())
        log_box.insert(tk.END, f"Sent: DATA:{payload}\n")
        log_box.see(tk.END)

    threading.Thread(target=send_data, daemon=True).start()

# ===================== KEY GENERATOR =====================
def generate_key_hex():
    return secrets.token_hex(32)

# ===================== KEY GENERATOR =====================
CSV_PATH = "device_log.csv"

def sync_device(client_entry, username_entry, key_entry):
    """
    Syncs a device entry in device_log.csv:
    - If device_id is new → append it.
    - If device exists → update fields.
    - If nothing changed → do nothing.
    """

    unique_id = client_entry.get().strip()
    username = username_entry.get().strip()
    key_hex = key_entry.get().strip()

    devices = {}

    # --- Load existing CSV if it exists ---
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                devices[row["unique_id"]] = {
                    "username": row["username"],
                    "key": row["key"]
                }

    # --- Check if device exists or changed ---
    if unique_id not in devices:
        print(f"[+] New device detected: {unique_id}")
        devices[unique_id] = {"username": username, "key": key_hex}

    else:
        old = devices[unique_id]
        if old["username"] != username or old["key"] != key_hex:
            print(f"[*] Updating existing device: {unique_id}")
            devices[unique_id] = {"username": username, "key": key_hex}
        else:
            print(f"[=] No change for {unique_id}")
            return  # nothing to write

    # --- Write back to CSV (sorted by unique_id) ---
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["unique_id", "username", "key"])
        writer.writeheader()

        for uid in sorted(devices.keys()):
            writer.writerow({
                "unique_id": uid,
                "username": devices[uid]["username"],
                "key": devices[uid]["key"]
            })

    print("CSV updated.")

# ===================== GUI =====================
root = tk.Tk()
root.title("ESP32 Config Manager")
root.geometry("1200x700")

frm = ttk.Frame(root, padding=20)
frm.pack(fill="both", expand=True)

# COM Port
ttk.Label(frm, text="Serial Port (COM)").grid(row=0, column=0, sticky="w", pady=(0,10))
port_entry = ttk.Entry(frm)
port_entry.insert(0, DEFAULT_PORT)
port_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0,10))

# SSID
ttk.Label(frm, text="WiFi SSID").grid(row=1, column=0, sticky="w", pady=(0,10))
ssid_entry = ttk.Entry(frm)
ssid_entry.insert(0, DEFAULT_NETWORK)
ssid_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0,10))

# Password
ttk.Label(frm, text="WiFi Password").grid(row=2, column=0, sticky="w", pady=(0,10))
pass_entry = ttk.Entry(frm, show="*")
pass_entry.insert(0, DEFAULT_NETWORK_PASS)
pass_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(0,10))

# MQTT
ttk.Label(frm, text="MQTT Server").grid(row=3, column=0, sticky="w", pady=(0,10))
mqtt_entry = ttk.Entry(frm)
mqtt_entry.insert(0, DEFAULT_MQTT_SERVER)
mqtt_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(0,10))

# Client ID
ttk.Label(frm, text="Client ID").grid(row=4, column=0, sticky="w", pady=(0,10))
client_entry = ttk.Entry(frm)
client_entry.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(0,10))

# Username
ttk.Label(frm, text="Username").grid(row=5, column=0, sticky="w", pady=(0,10))
username_entry = ttk.Entry(frm)
username_entry.grid(row=5, column=1, columnspan=2, sticky="ew", pady=(0,10))

# Key
ttk.Label(frm, text="Key (Hex 64 chars)").grid(row=6, column=0, sticky="w", pady=(0,10))
key_entry = ttk.Entry(frm)
key_entry.insert(0, generate_key_hex())
key_entry.grid(row=6, column=1, columnspan=1, sticky="ew", pady=(0,10), padx=(0,10))

# New Key Button
newkey_btn = ttk.Button(
    frm, 
    text="Regenerate Key",
    command=lambda: (key_entry.delete(0, tk.END), key_entry.insert(0, generate_key_hex()))
)
newkey_btn.grid(row=6, column=2, columnspan=1, pady=(0,10))

# Send Button
send_btn = ttk.Button(
    frm, 
    text="Send Config",
    command=lambda: (
        send_config(port_entry, ssid_entry, pass_entry, mqtt_entry, client_entry, key_entry, log_box),
        sync_device(client_entry, username_entry, key_entry)
        )
    )

send_btn.grid(row=7, column=0, columnspan=2, pady=10)

# Log Textbox
ttk.Label(frm, text="Log Output:").grid(row=7, column=0, sticky="w", pady=(10,0))
log_box = tk.Text(frm, height=12)
log_box.grid(row=8, column=0, columnspan=2, sticky="nsew")

# Grid expansion rules
frm.columnconfigure(1, weight=1)
frm.rowconfigure(8, weight=1)

root.mainloop()
