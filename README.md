# hazard-monitoring-mqtt

Infrastructure/Tech Overview:

<img width="1920" height="1080" alt="Copy of FYP1 Proposal Defence(2)" src="https://github.com/user-attachments/assets/36f78784-a5be-4156-baca-84ac8998ce72" />


Instructions:

1. Build IoT device according to:

Image:
<img width="1920" height="1080" alt="Copy of FYP1 Proposal Defence(1)" src="https://github.com/user-attachments/assets/1f0942b5-18b5-4789-8726-9aaecb893e89" />

Schematic Diagram:
<img width="846" height="602" alt="SCH_ESP32-monitoring-device_1-P1_2025-11-04" src="https://github.com/user-attachments/assets/bf98aa16-b52d-4b03-a64e-43baf4b356d4" />

2. Upload code in device/device.ino into device (using Arduino IDE).

3. Run device_config.py with device connected via USB and configure device networking credentials and MQTT server. The device should appear within device_log.csv as a new device (delete existing dummy devices if needed).

4. Run main.py to start API servers. Ensure your MQTT server is online as well.

5. Data should start streaming into data.csv (delete existing dummy data if needed). Data is also viewable through dashboard hosted in localhost.
