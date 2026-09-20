#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "imu_model.h"
#include "imu_scaler.h"
#include <Wire.h>

constexpr int kTensorArenaSize = 50 * 1024;
constexpr int kWindowSamples = 62;
constexpr int kFeaturesPerImu = 6;
constexpr uint8_t kImu0Address = 0x68;
constexpr uint8_t kImu1Address = 0x69;
constexpr uint8_t kImu2Address = 0x68;
constexpr uint8_t kAccelRegister = 0x2D;
constexpr uint8_t kGyroRegister = 0x33;
constexpr uint8_t kPowerManagement1 = 0x06;
constexpr uint8_t kPowerManagement2 = 0x07;
constexpr uint8_t kSoftwareSda = 2;
constexpr uint8_t kSoftwareScl = 3;
constexpr int kExpectedLabel = -1;
constexpr unsigned long kSampleIntervalMs = 10;
constexpr unsigned long kPredictionIntervalMs = 2000;
uint8_t tensor_arena[kTensorArenaSize];

const char* labels[] = {
  "Extension",
  "Flexion",
  "Pronation",
  "Radial_Deviation",
  "Supination",
  "Ulnar_Deviation"
};

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
float latest_imu0[kFeaturesPerImu];
float latest_imu1[kFeaturesPerImu];
float latest_imu2[kFeaturesPerImu];
float window_data[kWindowSamples][N_FEATURES];
unsigned long window_number = 0;
unsigned long correct_windows = 0;
unsigned long labeled_windows = 0;

class BitBangI2C {
 public:
  BitBangI2C(uint8_t sda, uint8_t scl) : sda_(sda), scl_(scl) {}

  void begin() {
    release(sda_);
    release(scl_);
  }

  bool readRegister(uint8_t address, uint8_t reg, uint8_t* buffer, uint8_t length) {
    start();
    if (!writeByte(static_cast<uint8_t>(address << 1)) || !writeByte(reg)) {
      stop();
      return false;
    }
    start();
    if (!writeByte(static_cast<uint8_t>((address << 1) | 1))) {
      stop();
      return false;
    }
    for (uint8_t index = 0; index < length; ++index) {
      buffer[index] = readByte(index + 1 < length);
    }
    stop();
    return true;
  }

  bool writeRegister(uint8_t address, uint8_t reg, uint8_t value) {
    start();
    bool acknowledged = writeByte(static_cast<uint8_t>(address << 1)) &&
                        writeByte(reg) && writeByte(value);
    stop();
    return acknowledged;
  }

 private:
  uint8_t sda_;
  uint8_t scl_;

  void release(uint8_t pin) {
    pinMode(pin, INPUT_PULLUP);
  }

  void pullLow(uint8_t pin) {
    digitalWrite(pin, LOW);
    pinMode(pin, OUTPUT);
  }

  void clockHigh() {
    release(scl_);
    delayMicroseconds(5);
  }

  void clockLow() {
    pullLow(scl_);
    delayMicroseconds(5);
  }

  void start() {
    release(sda_);
    clockHigh();
    pullLow(sda_);
    clockLow();
  }

  void stop() {
    pullLow(sda_);
    clockHigh();
    release(sda_);
  }

  bool writeByte(uint8_t value) {
    for (uint8_t bit = 0; bit < 8; ++bit) {
      if (value & 0x80) release(sda_);
      else pullLow(sda_);
      clockHigh();
      clockLow();
      value <<= 1;
    }
    release(sda_);
    clockHigh();
    bool acknowledged = digitalRead(sda_) == LOW;
    clockLow();
    return acknowledged;
  }

  uint8_t readByte(bool acknowledge) {
    uint8_t value = 0;
    release(sda_);
    for (uint8_t bit = 0; bit < 8; ++bit) {
      value <<= 1;
      clockHigh();
      if (digitalRead(sda_) == HIGH) value |= 1;
      clockLow();
    }
    if (acknowledge) pullLow(sda_);
    else release(sda_);
    clockHigh();
    clockLow();
    return value;
  }
};

BitBangI2C software_i2c(kSoftwareSda, kSoftwareScl);

bool readBytes(uint8_t address, uint8_t reg, uint8_t* buffer, uint8_t length) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(address, length) != length) {
    return false;
  }

  for (uint8_t index = 0; index < length; ++index) {
    buffer[index] = Wire.read();
  }
  return true;
}

bool readBytes(BitBangI2C& bus, uint8_t address, uint8_t reg, uint8_t* buffer, uint8_t length) {
  return bus.readRegister(address, reg, buffer, length);
}

bool readImu(uint8_t address, float* values) {
  uint8_t accel_bytes[6];
  uint8_t gyro_bytes[6];
  if (!readBytes(address, kAccelRegister, accel_bytes, 6) ||
      !readBytes(address, kGyroRegister, gyro_bytes, 6)) {
    return false;
  }

  int16_t raw_accel[3];
  int16_t raw_gyro[3];
  for (int axis = 0; axis < 3; ++axis) {
    raw_accel[axis] = static_cast<int16_t>(
      (static_cast<uint16_t>(accel_bytes[axis * 2]) << 8) |
      accel_bytes[axis * 2 + 1]);
    raw_gyro[axis] = static_cast<int16_t>(
      (static_cast<uint16_t>(gyro_bytes[axis * 2]) << 8) |
      gyro_bytes[axis * 2 + 1]);
    values[axis] = raw_accel[axis] / 16384.0f;
    values[axis + 3] = raw_gyro[axis] / 131.0f;
  }
  return true;
}

bool readImu(BitBangI2C& bus, uint8_t address, float* values) {
  uint8_t accel_bytes[6];
  uint8_t gyro_bytes[6];
  if (!readBytes(bus, address, kAccelRegister, accel_bytes, 6) ||
      !readBytes(bus, address, kGyroRegister, gyro_bytes, 6)) {
    return false;
  }

  for (int axis = 0; axis < 3; ++axis) {
    int16_t accel = static_cast<int16_t>(
      (static_cast<uint16_t>(accel_bytes[axis * 2]) << 8) |
      accel_bytes[axis * 2 + 1]);
    int16_t gyro = static_cast<int16_t>(
      (static_cast<uint16_t>(gyro_bytes[axis * 2]) << 8) |
      gyro_bytes[axis * 2 + 1]);
    values[axis] = accel / 16384.0f;
    values[axis + 3] = gyro / 131.0f;
  }
  return true;
}

bool initializeImu(uint8_t address) {
  Wire.beginTransmission(address);
  Wire.write(kPowerManagement1);
  Wire.write(0x01);
  if (Wire.endTransmission() != 0) return false;
  delay(100);

  Wire.beginTransmission(address);
  Wire.write(kPowerManagement2);
  Wire.write(0x00);
  return Wire.endTransmission() == 0;
}

bool initializeImu(BitBangI2C& bus, uint8_t address) {
  if (!bus.writeRegister(address, kPowerManagement1, 0x01)) return false;
  delay(100);
  return bus.writeRegister(address, kPowerManagement2, 0x00);
}

void setFeature(int sample, int feature, float raw_value) {
  float normalized = (raw_value - feature_means[feature]) / feature_scales[feature];
  int quantized = static_cast<int>(roundf(normalized / input->params.scale)) +
                  input->params.zero_point;
  quantized = constrain(quantized, -128, 127);
  input->data.int8[sample * N_FEATURES + feature] =
    static_cast<int8_t>(quantized);
}

bool fillInputWindow() {
  float imu0[kFeaturesPerImu];
  float imu1[kFeaturesPerImu];
  float imu2[kFeaturesPerImu];

  for (int sample = 0; sample < kWindowSamples; ++sample) {
    if (!readImu(kImu0Address, imu0) || !readImu(kImu1Address, imu1) ||
        !readImu(software_i2c, kImu2Address, imu2)) {
      return false;
    }

    for (int feature = 0; feature < kFeaturesPerImu; ++feature) {
      latest_imu0[feature] = imu0[feature];
      latest_imu1[feature] = imu1[feature];
      latest_imu2[feature] = imu2[feature];
    }

    for (int feature = 0; feature < kFeaturesPerImu; ++feature) {
      setFeature(sample, feature, imu0[feature]);
      setFeature(sample, feature + 6, imu1[feature]);
      setFeature(sample, feature + 12, imu2[feature]);
      window_data[sample][feature] = imu0[feature];
      window_data[sample][feature + 6] = imu1[feature];
      window_data[sample][feature + 12] = imu2[feature];
    }
    delay(kSampleIntervalMs);
  }
  return true;
}

void printCsvWindow(int prediction) {
  Serial.println("CSV_BEGIN");
  Serial.println("window,sample,imu0_acc_x,imu0_acc_y,imu0_acc_z,imu0_gyro_x,imu0_gyro_y,imu0_gyro_z,imu1_acc_x,imu1_acc_y,imu1_acc_z,imu1_gyro_x,imu1_gyro_y,imu1_gyro_z,imu2_acc_x,imu2_acc_y,imu2_acc_z,imu2_gyro_x,imu2_gyro_y,imu2_gyro_z,prediction_label");
  for (int sample = 0; sample < kWindowSamples; ++sample) {
    Serial.print(window_number);
    Serial.print(",");
    Serial.print(sample);
    for (int feature = 0; feature < N_FEATURES; ++feature) {
      Serial.print(",");
      Serial.print(window_data[sample][feature], 6);
    }
    Serial.print(",");
    Serial.println(labels[prediction]);
  }
  Serial.println("CSV_END");
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000) {
  }

  model = tflite::GetModel(imu_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: model schema version mismatch");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input type: ");
  Serial.println(input->type == kTfLiteInt8 ? "INT8" : "unexpected");
  Serial.print("Input bytes: ");
  Serial.println(input->bytes);
  Serial.print("Input scale: ");
  Serial.println(input->params.scale, 8);
  Serial.print("Input zero point: ");
  Serial.println(input->params.zero_point);
  Wire.begin();
  Wire.setClock(100000);
  software_i2c.begin();
  Serial.print("IMU0 init: ");
  Serial.println(initializeImu(kImu0Address) ? "OK" : "FAILED");
  Serial.print("IMU1 init: ");
  Serial.println(initializeImu(kImu1Address) ? "OK" : "FAILED");
  Serial.print("IMU2 init: ");
  Serial.println(initializeImu(software_i2c, kImu2Address) ? "OK" : "FAILED");
  Serial.println("Model ready");
}

void loop() {
  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    delay(1000);
    return;
  }

  if (!fillInputWindow()) {
    Serial.println("ERROR: could not read both hardware I2C IMUs");
    delay(1000);
    return;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: inference failed");
    delay(1000);
    return;
  }

  int best_index = 0;
  for (int i = 1; i < 6; ++i) {
    if (output->data.int8[i] > output->data.int8[best_index]) {
      best_index = i;
    }

    ++window_number;
    if (kExpectedLabel >= 0 && kExpectedLabel < 6) {
      ++labeled_windows;
      if (best_index == kExpectedLabel) ++correct_windows;
    }
  }

  Serial.print("Prediction: ");
  Serial.println(labels[best_index]);
  if (labeled_windows > 0) {
    Serial.print("Accuracy: ");
    Serial.print(100.0f * correct_windows / labeled_windows, 2);
    Serial.println(" percent");
  } else {
    Serial.println("Accuracy: unavailable; set kExpectedLabel to 0..5");
  }

  // Delay between inferences to allow user to return hand to start position
  delay(kPredictionIntervalMs);
}
