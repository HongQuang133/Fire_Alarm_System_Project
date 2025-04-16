#include "esp_camera.h"
#include <Arduino.h>
#include <WiFi.h>
#include <Adafruit_AMG88xx.h>
#include <Wire.h>
#include <DHT.h>
#include <ESP32Servo.h>
#include <Buzzer.h>
#include "esp_system.h"
#include "esp_log.h"
#include <SPI.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <UniversalTelegramBot.h>
#include <WiFiClientSecure.h>
#include <FreeRTOS.h>
#include <task.h>
#include "camera_server.h"

#include <TensorFlowLite_ESP32.h>
#include <model.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
  tflite:: ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite:: MicroInterpreter * interpreter = nullptr;
  TfLiteTensor* model_input = nullptr; 
  TfLiteTensor* model_output = nullptr;
  const int kTensorArenaSize =  120000;
  uint8_t *tensor_arena;
}
#define MQTT_MAX_PACKET_SIZE 20000
const char* ssid = "DEC";
const char* password = "13030103";
const char* mqtt_server = "broker.emqx.io";
const char* mqtt_clientid ="mqttx_b5926362";
const char* mqtt_username = "mine13";
const char* mqtt_password = "13313";
const char* camera_topic = "home/sensor/camera";
const char* thermal_topic = "home/sensor/thermal";
const char* dht_topic = "home/sensor/dht";
const char* smoke_topic = "home/sensor/smoke";
const char* bat_topic = "home/bat";
const int MAX_PAYLOAD = 60000;
WiFiClient espClient;
PubSubClient client(espClient);

// Telegram BOT Token (Get from Botfather)
String BOT_TOKEN = "7054569607:AAGwHflWAZhSRBAhoMk6a0omd9XDL0-SITY";
String chat_id = "6504093876";
WiFiClientSecure secured_client;
UniversalTelegramBot bot(BOT_TOKEN, secured_client);
bool isMoreDataAvailable();
bool dataAvailable = false;

bool reconnect() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi không kết nối. Không thể kết nối đến MQTT broker.");
    return false;
  }

  int attempts = 0;
  while (!client.connected() && attempts < 5) {
    Serial.print("Đang thử kết nối MQTT...");
    // Thử kết nối
    if (client.connect(mqtt_clientid, mqtt_username, mqtt_password)) {
      Serial.println("đã kết nối");
      // Subscribe Topic nếu cần
      // client.subscribe(mqtt_subscribe_topic);
      return true;
    } else {
      Serial.print("thất bại, rc=");
      Serial.print(client.state());
      Serial.println(" thử lại sau 5 giây");
      // Chờ 5 giây trước khi thử lại
      delay(5000);
      attempts++;
    }
  }
  return false;
}

void callback(char *topic, byte *message, unsigned int length)
{
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  String messageTemp;

  for (int i = 0; i < length; i++)
  {
    Serial.print((char)message[i]);
    messageTemp += (char)message[i];
  }
  Serial.println();

  // Feel free to add more if statements to control more GPIOs with MQTT
}

#define DHTPIN 35
#define DHTTYPE DHT11
#define MP2PIN 3
unsigned long previousMillisMP2 = 0;
unsigned long previousMillisMP2MQTT = 0;
const long intervalMP2 = 1000;
const long intervalMP2MQTT = 5000;
//Buzzer
#define BUZZER_PIN 47
Buzzer buzzer(BUZZER_PIN);

//pinLipo
#define LipoPIN 2
float calibration = 0.36; // Check Battery voltage using multimeter & add/subtract the value
unsigned long previousMillisBAT = 0;
const long intervalBAT = 10000;
unsigned long previousMillisBATMQTT = 0;
const long intervalBATMQTT = 10000;
//amg8833
//DHT11
DHT dht(DHTPIN, DHTTYPE);
int t = 0;
int h = 0;
unsigned long previousMillisDHT = 0;
unsigned long previousMillisDHTMQTT = 0;
const long intervalDHT = 2000;
const long intervalDHTMQTT = 5000;
//amg8833
TwoWire I2CAMG = 0;
Adafruit_AMG88xx amg;
float pixels[64];
float interpolated_pixels[784];
float get_point(float *p, uint8_t rows, uint8_t cols, int8_t x, int8_t y);
void set_point(float *p, uint8_t rows, uint8_t cols, int8_t x, int8_t y, float f);
void get_adjacents_1d(float *src, float *dest, uint8_t rows, uint8_t cols, int8_t x, int8_t y);
void get_adjacents_2d(float *src, float *dest, uint8_t rows, uint8_t cols, int8_t x, int8_t y);
float cubicInterpolate(float p[], float x);
float bicubicInterpolate(float p[], float x, float y);
void interpolate_image(float *src, uint8_t src_rows, uint8_t src_cols, 
                       float *dest, uint8_t dest_rows, uint8_t dest_cols);
unsigned long previousMillisAMG = 0;
unsigned long previousMillisAMGMQTT = 0;
const long intervalAMG = 1000;
const long intervalAMGMQTT = 1000;
int alarmStartTime = 0;
int alarmDuration = 0;
unsigned long previousMillisCamera = 0;
const long intervalCamera = 1000;

void startCameraServer();

void initAMG(){
  I2CAMG.begin(20,21,10000);

  if(!amg.begin(0x68, &I2CAMG)){
    Serial.println("AMG8833 not connect");
  }
    Serial.println("AMG8833 connect");
}

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 8;
  config.pin_d1 = 9;
  config.pin_d2 = 10;
  config.pin_d3 = 11;
  config.pin_d4 = 12;
  config.pin_d5 = 18;
  config.pin_d6 = 17;
  config.pin_d7 = 16;
  config.pin_xclk = -1;
  config.pin_pclk = 15;
  config.pin_vsync = 6;
  config.pin_href = 7;
  config.pin_sscb_sda = 4;
  config.pin_sscb_scl = 5;
  config.pin_pwdn = -1;
  config.pin_reset = 13;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  // Khởi động camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    Serial.println();
    return;
  }
  else{
    Serial.println("Camera connected");
  }
}
void streamCameraServer(){
    if (WiFi.status() == WL_CONNECTED) {
    Serial.println(WiFi.localIP());
    startCameraServer();
    }
    else
    Serial.println("Stream failed!");
}
String sendPhotoTelegram(){
  const char* myDomain = "api.telegram.org";
  String getAll = "";
  String getBody = "";

  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();  
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    ESP.restart();
    return "Camera capture failed";
  }  
  
  Serial.println("Connect to " + String(myDomain));

  if (secured_client.connect(myDomain, 443)) {
    Serial.println("Connection successful");
    
    String head = "--RandomNerdTutorials\r\nContent-Disposition: form-data; name=\"chat_id\"; \r\n\r\n" + chat_id + "\r\n--RandomNerdTutorials\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"esp32-cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--RandomNerdTutorials--\r\n";

    uint16_t imageLen = fb->len;
    uint16_t extraLen = head.length() + tail.length();
    uint16_t totalLen = imageLen + extraLen;
  
    secured_client.println("POST /bot"+BOT_TOKEN+"/sendPhoto HTTP/1.1");
    secured_client.println("Host: " + String(myDomain));
    secured_client.println("Content-Length: " + String(totalLen));
    secured_client.println("Content-Type: multipart/form-data; boundary=RandomNerdTutorials");
    secured_client.println();
    secured_client.print(head);
  
    uint8_t *fbBuf = fb->buf;
    size_t fbLen = fb->len;
    for (size_t n=0;n<fbLen;n=n+1024) {
      if (n+1024<fbLen) {
        secured_client.write(fbBuf, 1024);
        fbBuf += 1024;
      }
      else if (fbLen%1024>0) {
        size_t remainder = fbLen%1024;
        secured_client.write(fbBuf, remainder);
      }
    }  
    
   secured_client.print(tail);
    
    esp_camera_fb_return(fb);
    
    int waitTime = 10000;   // timeout 10 seconds
    long startTimer = millis();
    boolean state = false;
    
    while ((startTimer + waitTime) > millis()){
      Serial.print(".");
      delay(100);      
      while (secured_client.available()) {
        char c = secured_client.read();
        if (state==true) getBody += String(c);        
        if (c == '\n') {
          if (getAll.length()==0) state=true; 
          getAll = "";
        } 
        else if (c != '\r')
          getAll += String(c);
        startTimer = millis();
      }
      if (getBody.length()>0) break;
    }
    secured_client.stop();
    Serial.println(getBody);
  }
  else {
    getBody="Connected to api.telegram.org failed.";
    Serial.println("Connected to api.telegram.org failed.");
  }
  return getBody;
}

void initDHT(){
  dht.begin();
  Serial.println("DHT connected");
}
void readDHT() {
    float h = dht.readHumidity();
    float t = dht.readTemperature();

    if (isnan(h) || isnan(t)) {
      Serial.println("Failed to read from DHT sensor!");
    } else {
      Serial.print("Temperature: ");
      Serial.print(t);
      Serial.print(" C, Humidity: ");
      Serial.print(h);
      Serial.println(" %");
    }
}

int readMP2Data(){
  int smokeDetected = digitalRead(MP2PIN);
  return smokeDetected;
}
void  readAMGData(void *parameter){
  float maxTemp = 0.0;
  amg.readPixels(pixels);
  for (int i=0;i<64;i++){
    if(pixels[i]>maxTemp){
      maxTemp = pixels[i];
    }
  }
}
//send thermal pixel to MQTT
void send_thermal(){
   if (!client.connected()) {
    if (!reconnect()) {
      Serial.println("Can't connect MQTT.");
      return;
    }
  }
  amg.readPixels(pixels);
  String image = "";
  amg.readPixels(pixels);  
  Serial.print("[");
  for(int i=AMG88xx_PIXEL_ARRAY_SIZE; i>0; i--){
    image = image + pixels[i-1] + ",";
    Serial.print(pixels[i-1]);
    Serial.print(",");
    if( i%8 == 0 ) Serial.println();
  }
  image = image.substring(0, image.length() - 1);
  Serial.println("]");
  Serial.println();
  if (client.publish(thermal_topic, image.c_str())) {
    Serial.println("Data AMG sended: ");
    Serial.println(image);
  } else {
    Serial.println("Send data AMG fail.");
  }
}
void send_DHT(){
  if (!client.connected()) {
    if (!reconnect()) {
      Serial.println("Can't connect MQTT.");
      return;
    }
  }
  client.loop();
  int h = dht.readHumidity();
  int t = dht.readTemperature();
  StaticJsonDocument<200> docDHT;
  docDHT["temperature"] = t;
  docDHT["humidity"] = h;
  String jsonBufferDHT;
  serializeJson(docDHT, jsonBufferDHT);
  if (client.publish(dht_topic, jsonBufferDHT.c_str())) {
    Serial.println("Dữ liệu DHT đã được gửi: ");
    Serial.println(jsonBufferDHT);
  } else {
    Serial.println("Gửi dữ liệu DHT thất bại.");
  }
}
float readBatteryVoltage(){
  int pinValue = analogRead(LipoPIN);
  float voltage;
  voltage = ((pinValue*3.3)/4095.0);
  float bat_per = (voltage-1.5)*100/(2.1-1.5);
  if (bat_per >= 100)
  {
    bat_per = 100;
  }
  if (bat_per <= 0)
  {
    bat_per= 1;
  }
  return bat_per;
}
void sendBatteryStatus() {
  if (!client.connected()) {
    if (!reconnect()) {
      Serial.println("Can't connect MQTT.");
      return;
    }
  }
    int batteryPercentage = readBatteryVoltage();
    StaticJsonDocument<200> docBat; // Tạo một đối tượng JSON
    docBat["battery"] = batteryPercentage;
    char jsonBuffer[512];
    serializeJson(docBat, jsonBuffer); // Chuyển đổi JSON thành chuỗi ký tự
    if (client.publish(bat_topic, jsonBuffer)) {
    Serial.println("Data battery not send: ");
    Serial.println(jsonBuffer);
  } else {
    Serial.println("Data battery sended.");
  }
}
void sendBatTelegram(){
  float batteryPercentage = readBatteryVoltage();
  if(batteryPercentage<=50){
    if (WiFi.status() == WL_CONNECTED)
    {
       String message = "Low battery alarm needs to be charged";
      bot.sendMessage(chat_id, message, "");
    }
    else {
      Serial.println("WiFi not connected. Unable to send message to Telegram.");
    }
  }
  Serial.println(batteryPercentage);
}
void alarm_sound(void *parameters){
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  buzzer.sound(4000,500);
  buzzer.sound(1000,500);
  vTaskDelete(NULL);
}

void fire_detect(){
  h = dht.readHumidity();
  t = dht.readTemperature();
  amg.readPixels(pixels);
  interpolate_image(pixels,8,8,interpolated_pixels,28,28);
  uint8_t data_in[1][28][28];
  for(int i=0;i<28;i++){
    for(int j=0; j<28; j++){
      data_in[0][i][j]=round(interpolated_pixels[i*8 + j]);
    }
     Serial.println();
  }
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      model_input->data.uint8[i * 28 + j] = data_in[0][i][j]; // Dữ liệu đầu vào ở định dạng float
    } 
  }
  interpreter->Invoke();
  int output = model_output->data.uint8[0];
  Serial.println(output);
  if (output>112){
    Serial.println("FIRE!");
    xTaskCreate(alarm_sound, "AlarmSoundTask", 10000, NULL, 1, NULL);
    if (WiFi.status() == WL_CONNECTED) {
      String message = "Fire detected!!\nEnvironment: Temperature: " + String(t) + " C  Humidity: " + String(h) + " %";
      bot.sendMessage(chat_id, message, "");
      sendPhotoTelegram();
    } else {
      Serial.println("WiFi not connected. Unable to send message to Telegram.");
    }
  }
  else 
    Serial.println("No Fire");
}
void smoke_detect(){
  h = dht.readHumidity();
  t = dht.readTemperature();
  int smokeDetected = readMP2Data();
  if(smokeDetected == LOW){
    Serial.println("Smoke detected");
    xTaskCreate(alarm_sound, "AlarmSoundTask", 10000, NULL, 1, NULL);
    if (WiFi.status() == WL_CONNECTED) {
      String message = "Smoke detected!!\nEnvironment: Temperature: " + String(t) + " C  Humidity: " + String(h) + " %";
      bot.sendMessage(chat_id, message, "");
      sendPhotoTelegram();
    } else {
      Serial.println("WiFi not connected. Unable to send message to Telegram.");
    }
  }
  else{
    Serial.println("No Smoke");
  }
}
void setupWiFi(){
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
// Connect to Wi-Fi network
  WiFi.begin(ssid, password);
  unsigned long startTime = millis();
// Wait until the connection is established or 10 seconds have passed
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < 10000) {
  delay(1000);
  Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
  // Print the IP address assigned to the ESP32
    Serial.println("");
    Serial.println("WiFi connected.");
  } 
  else {
    Serial.println("");
    Serial.println("WiFi connection timed out. Continuing without WiFi.");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_tflite);
  if(model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report("Model version not matched");
  }

  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddFullyConnected();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddLogistic();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  tensor_arena = (uint8_t *)malloc(kTensorArenaSize);
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize,error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("Allocate Tensor failed");
      while (1);
    }
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);  
  secured_client.setInsecure();
  secured_client.setCACert(TELEGRAM_CERTIFICATE_ROOT);
  client.setServer(mqtt_server, 1883);
  client.setBufferSize (MAX_PAYLOAD); //This is the maximum payload length
  client.setCallback(callback);
  setupWiFi();
   if (WiFi.status() == WL_CONNECTED) {
    Serial.println(WiFi.localIP());
   }
  initCamera();
  initAMG();
  initDHT();
}
void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillisDHT >= intervalDHT) {
        previousMillisDHT = currentMillis;
      readDHT();
  }
  if (currentMillis - previousMillisDHTMQTT >= intervalDHTMQTT) {
        previousMillisDHTMQTT= currentMillis;
      send_DHT();
  }
  if (currentMillis - previousMillisAMGMQTT >= intervalAMGMQTT) {
        previousMillisAMGMQTT = currentMillis;
      send_thermal();
  }
  if (currentMillis - previousMillisAMG >= intervalAMG) {
        previousMillisAMG = currentMillis;
      fire_detect();
  }
  if (currentMillis - previousMillisMP2 >= intervalMP2) {
        previousMillisMP2 = currentMillis;
      smoke_detect();
  }
  if (currentMillis - previousMillisBAT >= intervalBAT) {
        previousMillisBAT = currentMillis;
      sendBatTelegram();
  }
  if (currentMillis - previousMillisBATMQTT >= intervalBATMQTT) {
        previousMillisBATMQTT = currentMillis;
      sendBatteryStatus();
  }
  if (currentMillis - previousMillisCamera >= intervalCamera) {
        previousMillisCamera = currentMillis;
      streamCameraServer();
  }
}