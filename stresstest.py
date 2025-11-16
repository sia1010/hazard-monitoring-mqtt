import subprocess
import time
import os

# --- Configuration ---
NUM_SIMULATORS = 10  # Number of simultaneous device simulators
PUBLISH_INTERVAL = 5  # Seconds (must match your main script default)
DEVICE_LOG_PATH = "device_log.csv"  # Path to device log used by each simulator
SCRIPT_PATH = "stream_dummy_data.py"  # Replace with your main script filename

# --- Main Stress Test ---
processes = []

try:
    for i in range(1, NUM_SIMULATORS+1):
        print(f"Starting simulator instance {i}...")

        # Launch a new subprocess running the simulator script
        # Use Python executable to run the script
        proc = subprocess.Popen(
            ["python", SCRIPT_PATH],
            stdin=subprocess.PIPE,  # Allow simulated input
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        inputs = f"{i+1}\n\n\n\n\n\n\n\nREP\n"
        proc.stdin.write(inputs)
        proc.stdin.flush()
        processes.append(proc)
        time.sleep(0.5)  # Slight stagger to avoid simultaneous starts

    print(f"\n{NUM_SIMULATORS} simulator instances launched. Press Ctrl+C to stop.\n")

    # Keep running until interrupted
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping all simulator instances...")

    for proc in processes:
        try:
            proc.terminate()
        except Exception:
            pass

    print("All simulators stopped.")

finally:
    # Optional: capture output for logging
    for i, proc in enumerate(processes):
        stdout, stderr = proc.communicate(timeout=1)
        if stdout:
            print(f"\n--- Simulator {i+1} STDOUT ---\n{stdout}")
        if stderr:
            print(f"\n--- Simulator {i+1} STDERR ---\n{stderr}")
