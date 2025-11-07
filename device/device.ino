#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "DHT.h"
#include <Arduino.h>
#include <TinyGPS++.h>
#include <Preferences.h>
#include <Crypto.h>
#include <ChaChaPoly.h>
#include <string>
#include <cstdint>
#include <base64.h>
#include <vector>

// ============ CONFIGURATION ============

// --- Preferences (NVS Storage) ---
Preferences prefs;
// --- Key Management ---
uint8_t key[32];
uint8_t nonce[12] = { 0, 0, 0, 0, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0, 1 };
uint32_t nonce_counter = 0;  // counter used in nonce[0..3]
ChaChaPoly chacha;

// --- WiFi / MQTT ---
char ssid[64];
char password[64];
char mqtt_server[64];
char client_id[64];

WiFiClient espClient;
PubSubClient client(espClient);

// --- Timing ---
#define TIME_PER_CYCLE 5000    // Publish every 5s
#define TIME_PER_READING 1000  // Take reading every 1s
unsigned long lastReading = 0;
unsigned long lastPublish = 0;

// --- Pins ---
const int ledPin = 2;
#define DHTPIN 23
#define DHTTYPE DHT11
#define SOUNDPIN 35
#define EMERGENCY_BUTTON 19  // New pin for emergency status toggle

DHT dht(DHTPIN, DHTTYPE);

// --- ADC and Sampling ---
const float ADC_MAX = 4095.0f;
const float VREF = 3.0f;
const int sampleWindow = 50;

// --- Buffers for averaging ---
std::vector<double> spl_measurements;
float sum_temp = 0, sum_humidity = 0;
int sample_count = 0;

// --- GPS ---
#define RX 16
#define TX 17
#define GPS_BAUD 9600
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);
double lastValidLat = -1.0;
double lastValidLng = -1.0;
unsigned long lastValidFixTime = 0;

// --- Device Status ---
const char* STATUS_OK = "OK";
const char* STATUS_EMERGENCY = "EMERGENCY";
const char* deviceStatus = STATUS_OK;  // Default status

// --- Variables to hold the latest single reading data ---
// These are needed so the checkEmergencyButton can publish an immediate alert with fresh data
static float current_t = 0;
static float current_h = 0;
static double current_spl = 0;
static double current_lat = -1.0;
static double current_lng = -1.0;
static float current_fix_time = -1.0;


// =======================================
// ========== PREFERENCES HANDLING =======
// =======================================

void handleSerialCommands() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  // ---- ENTER CONFIG MODE ----
  if (line == "CMD:CONFIG") {
    Serial.println("ACK:CONFIG");  // handshake
    enterConfigMode();
  }
}

void enterConfigMode() {
  Serial.println("READY:DATA");  // tell Python we're ready for config payload

  // Wait for data line starting with DATA:
  while (!Serial.available()) delay(20);

  String dataLine = Serial.readStringUntil('\n');
  dataLine.trim();

  if (!dataLine.startsWith("DATA:")) {
    Serial.println("ERR:BAD-DATA");
    return;
  }

  String input = dataLine.substring(5);  // remove "DATA:"
  input.trim();

  // Parse CSV input
  int idx1 = input.indexOf(',');
  int idx2 = input.indexOf(',', idx1 + 1);
  int idx3 = input.indexOf(',', idx2 + 1);
  int idx4 = input.indexOf(',', idx3 + 1);

  if (idx1 < 0 || idx2 < 0 || idx3 < 0 || idx4 < 0) {
    Serial.println("ERR:FORMAT");
    return;
  }

  String ssid     = input.substring(0, idx1);
  String pass     = input.substring(idx1 + 1, idx2);
  String mqtt     = input.substring(idx2 + 1, idx3);
  String client   = input.substring(idx3 + 1, idx4);
  String keyHex   = input.substring(idx4 + 1);

  // Convert hex key → bytes
  uint8_t key[32] = {0};
  for (int i = 0; i < 32 && (i * 2 + 1) < keyHex.length(); i++) {
    key[i] = strtoul(keyHex.substring(i * 2, i * 2 + 2).c_str(), NULL, 16);
  }

  // Save config
  saveConfig(ssid.c_str(), pass.c_str(), mqtt.c_str(), client.c_str(), key);

  Serial.println("OK:SAVED");
  delay(400);
  ESP.restart();
}

void saveConfig(const char* ssid, const char* password, const char* mqtt, const char* client, const uint8_t* key) {
  prefs.begin("config", false);
  prefs.putString("ssid", ssid);
  prefs.putString("pass", password);
  prefs.putString("mqtt", mqtt);
  prefs.putString("client", client);
  prefs.putBytes("key", key, 32);
  prefs.end();
  Serial.println("Configuration saved to NVS!");
}

bool loadConfig() {
  prefs.begin("config", true);

  String ssidStr = prefs.getString("ssid", "");
  String passStr = prefs.getString("pass", "");
  String mqttStr = prefs.getString("mqtt", "");
  String clientStr = prefs.getString("client", "");
  size_t keyLen = prefs.getBytesLength("key");
  if (ssidStr == "" || passStr == "" || mqttStr == "" || clientStr == "" || keyLen != 32) {
    prefs.end();
    Serial.println("No valid configuration found in NVS");
    return false;
  }

  ssidStr.toCharArray(ssid, sizeof(ssid));
  passStr.toCharArray(password, sizeof(password));
  mqttStr.toCharArray(mqtt_server, sizeof(mqtt_server));
  clientStr.toCharArray(client_id, sizeof(client_id));
  prefs.getBytes("key", key, 32);

  prefs.end();
  Serial.println("Configuration loaded from NVS!");
  return true;
}

void generateNonce(uint8_t* nonce, size_t len) {
  for (size_t i = 0; i < len; i++) {
    nonce[i] = random(0, 256);
  }
}

size_t encryptPayload(const char* plaintext, uint8_t* ciphertext, uint8_t* tag, uint8_t* nonce) {
  size_t len = strlen(plaintext);

  chacha.clear();
  chacha.setKey(key, sizeof(key));
  chacha.setIV(nonce, 12);
  chacha.encrypt(ciphertext, (const uint8_t*)plaintext, len);
  chacha.computeTag(tag, 16);

  return len;
}

String base64Encode(const uint8_t* data, size_t length) {
  return base64::encode(data, length);
}
// =======================================
// ========== CORE FUNCTIONS =============
// =======================================

void setup_wifi() {
  Serial.println();
  Serial.printf("Connecting to %s...\n", ssid);

  WiFi.begin(ssid, password);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 30) {
    delay(500);
    Serial.print(".");
    retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi; Entering Config Mode");
    while (true){
      handleSerialCommands();
    }
  }
}

void callback(char* topic, byte* message, unsigned int length) {
  String msg;
  for (int i = 0; i < length; i++) msg += (char)message[i];
  Serial.printf("Message [%s]: %s\n", topic, msg.c_str());
  if (String(topic) == "hazard-monitoring/client") {
    if (msg == "on") digitalWrite(ledPin, HIGH);
    else if (msg == "off") digitalWrite(ledPin, LOW);
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect(client_id)) {
      Serial.println("connected");
      client.subscribe("hazard-monitoring/client");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying...");
      delay(500);
    }
  }
}

int measureAvgPeakToPeak(int numSamples = 5) {
  int sum = 0;
  for (int i = 0; i < numSamples; i++) {
    unsigned int signalMax = 0;
    unsigned int signalMin = 4095;
    unsigned long startMillis = millis();
    while (millis() - startMillis < sampleWindow) {
      unsigned int sample = analogRead(SOUNDPIN);
      if (sample < 4095) {
        if (sample > signalMax) signalMax = sample;
        if (sample < signalMin) signalMin = sample;
      }
    }

    int peakToPeak = signalMax - signalMin;
    sum += peakToPeak;
    yield();
  }

  return sum / numSamples;
}

double logarithmicAverageSPL(const std::vector<double>& spl_values) {
  if (spl_values.empty()) {
    Serial.print("Error");
    return 0.0;
  }

  // Step 1 & 2: Convert each dB to its linear intensity ratio and sum them up.
  double sum_linear_intensity = 0.0;
  for (double spl_i : spl_values) {
    // Calculate the linear intensity ratio: 10^(Lp_i / 10)
    sum_linear_intensity += pow(10.0, spl_i / 10.0);
  }

  // Calculate the average linear intensity ratio: I_avg = sum(Ii) / N
  double average_linear_intensity = sum_linear_intensity / spl_values.size();
  // Step 3: Convert the average linear intensity ratio back to dB.
  // Lp_avg = 10 * log10(I_avg)
  double average_spl = 10.0 * log10(average_linear_intensity);

  return average_spl;
}

void updateGPS() {
  while (gpsSerial.available() > 0) gps.encode(gpsSerial.read());

  if (gps.location.isValid() && gps.location.isUpdated()) {
    lastValidLat = gps.location.lat();
    lastValidLng = gps.location.lng();
    lastValidFixTime = millis();
  }
}

/**
 * Publishes sensor data to MQTT.
 * @param isImmediate If true, does not clear averaging buffers.
 */
void publishSensorData(
  float avg_t,
  float avg_h,
  double avg_spl,
  double payloadLat,
  double payloadLng,
  float payloadSecondsSinceFix,
  bool isImmediate) {
  if (!client.connected()) {
    Serial.println("MQTT not connected, skipping publish.");
    return;
  }

  char payload[150];
  // Updated snprintf to include deviceStatus as the 8th field
  snprintf(payload, sizeof(payload),
           "%s,%.2f,%.2f,%.2f,%.8f,%.8f,%.2f,%s",
           client_id, avg_t, avg_h, avg_spl,
           payloadLat, payloadLng, payloadSecondsSinceFix, deviceStatus);

  Serial.print(isImmediate ? "Immediate Payload: " : "Periodic Payload: ");
  Serial.println(payload);

  // Buffers
  uint8_t ciphertext[200];
  uint8_t tag[16];
  uint8_t nonce[12];

  generateNonce(nonce, sizeof(nonce));

  // Encrypt
  size_t cipherLen = encryptPayload(payload, ciphertext, tag, nonce);

  String encryptedMsg = String(client_id) + "," + base64Encode(nonce, 12) + "," + base64Encode(ciphertext, cipherLen) + "," + base64Encode(tag, 16);

  client.publish("hazard-monitoring/server", encryptedMsg.c_str());

  digitalWrite(ledPin, HIGH);
  delay(50);
  digitalWrite(ledPin, LOW);

  // Clear buffers only if this is the periodic publish
  if (!isImmediate) {
    sum_temp = sum_humidity = 0;
    spl_measurements.clear();
    sample_count = 0;
  }
}


/**
 * Checks the emergency button state and toggles deviceStatus.
 * Publishes an immediate alert when status changes to EMERGENCY.
 */
void checkEmergencyButton(float t, float h, double SPL, double lat, double lng, float fixTime) {
  static unsigned long buttonPressStart = 0;
  const unsigned long requiredPressTime = 5000;  // 5 seconds

  // Pin is pulled HIGH, button press is LOW
  if (digitalRead(EMERGENCY_BUTTON) == LOW) {
    if (buttonPressStart == 0) {
      buttonPressStart = millis();  // Start timer on first press detection
      Serial.println("Emergency button pressed. Holding for 5 seconds to toggle status...");
    }

    if (millis() - buttonPressStart >= requiredPressTime) {

      // Toggle status
      if (deviceStatus == STATUS_OK) {
        deviceStatus = STATUS_EMERGENCY;
        Serial.println("!!! EMERGENCY STATUS ACTIVATED !!!");
      } else {
        deviceStatus = STATUS_OK;
        Serial.println("Emergency status reset to OK.");
      }

      // Send immediate message ONLY if status is EMERGENCY
      if (deviceStatus == STATUS_EMERGENCY) {
        // Publish the current single reading as an immediate alert (no averaging)
        publishSensorData(t, h, SPL, lat, lng, fixTime, true);
      }

      // Wait for release OR debounce after toggle
      unsigned long blockTime = millis();
      // Block for max 1 second or until released to prevent double-triggering
      while (digitalRead(EMERGENCY_BUTTON) == LOW && (millis() - blockTime) < 1000) {
        delay(10);
      }

      buttonPressStart = 0;  // Reset timer
    }
  } else {
    // Button is released/not pressed
    if (buttonPressStart != 0) {
      Serial.print("Button released after ");
      Serial.print(millis() - buttonPressStart);
      Serial.println("ms. No toggle.");
    }
    buttonPressStart = 0;
  }
}

// =======================================
// ========== SETUP & LOOP ===============
// =======================================

void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
  dht.begin();
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RX, TX);
  pinMode(EMERGENCY_BUTTON, INPUT_PULLUP);  // Initialize new button pin

  Serial.println("\n=== ESP32 Hazard Monitoring ===");
  // Load or save config
  if (!loadConfig()) {
    Serial.print("No config detected, please setup config");
    while (true) {
      handleSerialCommands();
      Serial.print(".");
      delay(100);
    }
  }

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);

  Serial.println("System Ready!");
}

void loop() {
  handleSerialCommands();

  if (WiFi.status() != WL_CONNECTED) setup_wifi();

  if (!client.connected()) reconnect();

  client.loop();

  unsigned long now = millis();
  if (now - lastReading >= TIME_PER_READING) {
    lastReading = now;

    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) {
      Serial.println("Failed to read from DHT!");
      return;
    }

    int peakToPeak = measureAvgPeakToPeak();
    double SPL = 20.0 * log10(pow((float)peakToPeak, 2.0) * 0.00000006) + 94.0;

    updateGPS();
    float secondsSinceFix = -1.0;

    if (lastValidFixTime > 0)
      secondsSinceFix = (millis() - lastValidFixTime) / 1000.0;
    double displayLat = (lastValidFixTime > 0) ? lastValidLat : -1.0;
    double displayLng = (lastValidFixTime > 0) ? lastValidLng : -1.0;

    // Store latest readings in global statics for the emergency function
    current_t = t;
    current_h = h;
    current_spl = SPL;
    current_lat = displayLat;
    current_lng = displayLng;
    current_fix_time = secondsSinceFix;

    Serial.printf(
      "Reading #%d → T: %.2f°C | H: %.2f%% | SPL: %.2f dB (p2p: %d) | GPS: %.8f, %.8f | Fix: %.2f s | Status: %s\n",
      sample_count + 1, current_t, current_h, current_spl, peakToPeak, current_lat, current_lng, current_fix_time, deviceStatus);
    spl_measurements.push_back(current_spl);

    sum_temp += current_t;
    sum_humidity += current_h;
    sample_count++;
  }

  // Check the emergency button based on the latest single reading
  checkEmergencyButton(current_t, current_h, current_spl, current_lat, current_lng, current_fix_time);


  if (now - lastPublish >= TIME_PER_CYCLE && sample_count > 0) {
    lastPublish = now;
    float avg_t = sum_temp / sample_count;
    float avg_h = sum_humidity / sample_count;
    double avg_spl = logarithmicAverageSPL(spl_measurements);
    double payloadLat = (lastValidFixTime > 0) ? lastValidLat : -1.0;
    double payloadLng = (lastValidFixTime > 0) ? lastValidLng : -1.0;
    float payloadSecondsSinceFix = (lastValidFixTime > 0)
                                     ? (millis() - lastValidFixTime) / 1000.0
                                     : -1.0;

    // Use the refactored function for periodic publish (isImmediate = false)
    publishSensorData(avg_t, avg_h, avg_spl, payloadLat, payloadLng, payloadSecondsSinceFix, false);

    // Note: sum_temp/humidity and spl_measurements are cleared inside publishSensorData(..., false)
  }
}