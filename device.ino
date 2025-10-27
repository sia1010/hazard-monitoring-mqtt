#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "DHT.h"
#include <Arduino.h> 
#include <TinyGPS++.h>
#include <SoftwareSerial.h>

// ============================================================
// ============ CONFIGURATION SECTION =========================
// ============================================================

// --- WiFi Configuration ---
const char* ssid = "Potato";             // WiFi SSID
const char* password = "kfcyummy";       // WiFi Password

// --- MQTT Configuration ---
const char* mqtt_server = "mqtt-dashboard.com"; // MQTT broker server
const char* client_id = "ESP32-unique485934";   // Unique MQTT client ID
WiFiClient espClient;                            // WiFi client
PubSubClient client(espClient);                  // MQTT client

// --- Timing Configuration ---
#define TIME_PER_CYCLE 5000   // Publish data every 5 seconds
#define TIME_PER_READING 1000 // Take sensor reading every 1 second
unsigned long lastReading = 0; // Timestamp for last reading
unsigned long lastPublish = 0; // Timestamp for last MQTT publish

// --- Pin Definitions ---
const int ledPin = 2;  // Built-in LED pin
#define DHTPIN 23      // DHT11 data pin
#define DHTTYPE DHT11  // Type of DHT sensor
#define SOUNDPIN 35    // Sound sensor analog input pin

DHT dht(DHTPIN, DHTTYPE); // Create DHT sensor object

// --- ADC and Sampling Configuration ---
const float ADC_MAX = 4095.0f;  // ESP32 ADC max value (12-bit)
const float VREF = 3.0f;        // ADC reference voltage
const int sampleWindow = 50;    // Sampling window in milliseconds

// --- Buffers for Averaging Sensor Data ---
float sum_temp = 0, sum_humidity = 0, sum_spl = 0; // Sums for averaging
int sample_count = 0;                               // Counter for averaging

// --- GPS Configuration ---
#define RX 16                // GPS RX pin (to TX of module)
#define TX 17                // GPS TX pin (to RX of module)
#define GPS_BAUD 9600        // GPS communication baud rate
TinyGPSPlus gps;             // TinyGPS++ object
HardwareSerial gpsSerial(2); // Use hardware serial port 2 for GPS
double lastValidLat = -1.0;  // Last valid latitude
double lastValidLng = -1.0;  // Last valid longitude
unsigned long lastValidFixTime = 0; // Timestamp of last valid GPS fix

// ============================================================
// ================== FUNCTION DEFINITIONS ====================
// ============================================================

// --- Connect to WiFi Network ---
void setup_wifi() {
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print("."); // Print dots while connecting
  }

  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP()); // Print assigned IP address
}

// --- Handle Incoming MQTT Messages ---
void callback(char* topic, byte* message, unsigned int length) {
  String msg;
  for (int i = 0; i < length; i++) msg += (char)message[i];
  Serial.printf("Message [%s]: %s\n", topic, msg.c_str());

  // Control LED from MQTT command
  if (String(topic) == "hazard-monitoring/client") {
    if (msg == "on") digitalWrite(ledPin, HIGH);
    else if (msg == "off") digitalWrite(ledPin, LOW);
  }
}

// --- Reconnect to MQTT Broker if Disconnected ---
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect(client_id)) {
      Serial.println("connected");
      client.subscribe("hazard-monitoring/client"); // Subscribe to control topic
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying...");
      delay(500);
    }
  }
}

// --- Measure Maximum Peak-to-Peak Sound Value ---
int measureMaxPeakToPeak(int numSamples = 5) {
  int maxPeakToPeak = 0;

  for (int i = 0; i < numSamples; i++) {
    unsigned int signalMax = 0;
    unsigned int signalMin = 4095;
    unsigned long startMillis = millis();

    // Measure sound amplitude in a short time window
    while (millis() - startMillis < sampleWindow) {
      unsigned int sample = analogRead(SOUNDPIN);
      if (sample < 4095) {
        if (sample > signalMax) signalMax = sample;
        if (sample < signalMin) signalMin = sample;
      }
      yield(); // Allow WiFi/MQTT background tasks
    }

    int peakToPeak = signalMax - signalMin; // Amplitude range
    if (peakToPeak > maxPeakToPeak) {
      maxPeakToPeak = peakToPeak; // Keep highest value among samples
    }

    yield(); // Yield between sampling iterations
  }

  return maxPeakToPeak; // Return maximum peak-to-peak amplitude
}

// --- Read and Update GPS Data ---
void updateGPS() {
  // Feed incoming serial data to TinyGPS++ parser
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Update stored location only if new valid fix is available
  if (gps.location.isValid() && gps.location.isUpdated()) {
    lastValidLat = gps.location.lat();
    lastValidLng = gps.location.lng();
    lastValidFixTime = millis();  // Record time of this fix
  }
}

// ============================================================
// ==================== MAIN SETUP ============================
// ============================================================

void setup() {
  Serial.begin(115200);      // Start serial monitor
  pinMode(ledPin, OUTPUT);   // Configure LED pin
  dht.begin();               // Initialize DHT sensor

  setup_wifi();              // Connect to WiFi
  client.setServer(mqtt_server, 1883); // Set MQTT broker
  client.setCallback(callback);        // Set MQTT callback

  Serial.println("System Ready!");

  // Initialize GPS serial communication
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RX, TX);
  Serial.println("GPS Serial started at 9600 baud rate");
}

// ============================================================
// ==================== MAIN LOOP =============================
// ============================================================

void loop() {
  // Reconnect WiFi or MQTT if disconnected
  if (WiFi.status() != WL_CONNECTED) setup_wifi();
  if (!client.connected()) reconnect();
  client.loop(); // Maintain MQTT connection

  unsigned long now = millis();

  // --- Take a reading every 1 second ---
  if (now - lastReading >= TIME_PER_READING) {
    lastReading = now;

    // Read temperature and humidity from DHT11
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) {
      Serial.println("Failed to read from DHT!");
      return;
    }

    // Measure sound peak-to-peak value and convert to dB SPL
    int peakToPeak = measureMaxPeakToPeak();
    float SPL = -56.47 + 41.93 * log10((float)peakToPeak);

    // Update GPS information
    updateGPS();
    float secondsSinceFix = -1.0;

    // Compute time elapsed since last valid GPS fix
    if (lastValidFixTime > 0) {
      secondsSinceFix = (millis() - lastValidFixTime) / 1000.0;
    }

    // Prepare GPS values for display
    double displayLat = lastValidLat;
    double displayLng = lastValidLng;

    // If GPS never had a valid fix, set coordinates to -1
    if (lastValidFixTime == 0) {
      displayLat = -1.0;
      displayLng = -1.0;
    }

    // Print current readings to serial monitor
    Serial.printf(
      "Reading #%d → T: %.2f°C | H: %.2f%% | SPL: %.2f dB | GPS: %.8f, %.8f | Last Fix: %.2f s ago\n",
      sample_count + 1, t, h, SPL, displayLat, displayLng, secondsSinceFix
    );

    // Add current readings to averages
    sum_temp += t;
    sum_humidity += h;
    sum_spl += SPL;
    sample_count++;
  }

  // --- Publish averaged data every 5 seconds ---
  if (now - lastPublish >= TIME_PER_CYCLE && sample_count > 0) {
    lastPublish = now;

    // Compute average values
    float avg_t = sum_temp / sample_count;
    float avg_h = sum_humidity / sample_count;
    float avg_spl = sum_spl / sample_count;

    // Prepare GPS data for payload
    double payloadLat = -1.0;
    double payloadLng = -1.0;
    float payloadSecondsSinceFix = -1.0;

    // Use last known GPS fix if valid
    if (lastValidFixTime > 0) {
      payloadLat = lastValidLat;
      payloadLng = lastValidLng;
      payloadSecondsSinceFix = (millis() - lastValidFixTime) / 1000.0;
    }

    // Create payload buffer (CSV formatted)
    char payload[150];
    snprintf(payload, sizeof(payload),
            "%s,%.2f,%.2f,%.2f,%.8f,%.8f,%.2f",
            client_id, avg_t, avg_h, avg_spl,
            payloadLat, payloadLng, payloadSecondsSinceFix);

    // Print payload to serial and publish to MQTT topic
    Serial.println(payload);
    client.publish("hazard-monitoring/server", payload);

    // Blink LED briefly to indicate publish
    digitalWrite(ledPin, HIGH);
    delay(50);
    digitalWrite(ledPin, LOW);

    // Reset averaging variables
    sum_temp = sum_humidity = sum_spl = 0;
    sample_count = 0;
  }
}
