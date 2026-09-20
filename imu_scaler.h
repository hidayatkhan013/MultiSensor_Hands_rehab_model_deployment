// Normalization parameters for on-device inference
#ifndef IMU_SCALER_H
#define IMU_SCALER_H

#define N_FEATURES 18

const float feature_means[N_FEATURES] = {
  -0.160874f, 0.311689f, 0.513255f, -1.396849f, -1.213665f, -2.142970f, -0.156189f, 0.273914f, 0.592776f, -0.749593f, 0.481544f, 1.393550f, -0.151199f, 0.339919f, 0.571637f, -0.957621f, 0.788829f, 0.603229f
};

const float feature_scales[N_FEATURES] = {
  0.402216f, 0.336016f, 0.582882f, 37.270354f, 33.494968f, 24.220823f, 0.340725f, 0.410694f, 0.549296f, 37.321557f, 30.580528f, 23.812148f, 0.363268f, 0.352714f, 0.568984f, 36.959308f, 32.243000f, 23.984998f
};

// Usage: normalized = (raw - mean) / scale
#endif
