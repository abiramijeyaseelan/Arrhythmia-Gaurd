# ArrhythmiaGuard — ECG Arrhythmia Detection on Edge

> Real-time ECG acquisition and arrhythmia pre-screening on ESP32-S3. Dual-model pipeline (Random Forest + 1D CNN) validated on AAMI EC57 inter-patient protocol. Fully on-device — no cloud, no server.

**AUC 0.908 · AAMI EC57 validated · INT8 TFLite · ESP32-S3 N16R8 · ₹5,000 vs ₹3,00,000 Holter monitor**

---

## What it does

Acquires ECG signal from the AD8232 analog front-end, preprocesses it in real-time, detects R-peaks, extracts 17 features per beat, and runs a 1D CNN on-chip to classify each beat as normal or arrhythmic. Results display on an SSD1306 OLED with LED indicators.

A Random Forest with SHAP explainability runs on a Streamlit dashboard for desktop-side analysis.

---

## Hardware

| Component | Purpose |
|---|---|
| ESP32-S3 N16R8 | MCU — inference + signal processing |
| AD8232 AFE | ECG analog front-end, signal conditioning |
| SSD1306 OLED | Beat classification display |
| LED Indicators | Normal / arrhythmia visual alert |

**Total BOM: ₹5,000** vs ₹3,00,000 (Holter Monitor) / ₹75,000 (Apple Watch)

---

## Signal Processing Pipeline

```
AD8232 ECG Signal (360 Hz)
        ↓
Bandpass filter 0.5–40 Hz (4th order Butterworth)
        ↓
50 Hz notch filter (India power line)
        ↓
Max normalisation
        ↓
R-peak detection → beat segmentation
(258 samples: 100 before, 158 after R-peak)
        ↓
Feature extraction (17 features/beat)
        ↓
1D CNN inference (TFLite INT8, float32 I/O)
        ↓
OLED + LED output
```

---

## Feature Extraction — 17 features per beat

| Feature | Description |
|---|---|
| RR interval | Time between consecutive R-peaks |
| RR ratio | Current / previous RR interval |
| Morphology (×5) | Beat shape samples around R-peak |
| QRS width | Width of QRS complex (ms) |
| QRS energy | Energy in QRS region |
| beat_std | Standard deviation of beat amplitude |
| beat_skew | Skewness of beat waveform |
| beat_kurt | Kurtosis of beat waveform |
| beat_entropy | Entropy of absolute beat values |

---

## ML Pipeline

### Dual-model architecture

**Random Forest** — explainability + desktop dashboard
- 17-feature input
- Patient-level cross-validation (GroupKFold)
- SMOTE balancing on training set only
- SHAP explainability — top features driving each prediction
- Role: interpretable dashboard, clinical use

**1D CNN** — edge deployment on ESP32-S3
- Raw beat input: 258 samples × 1 channel
- Architecture fits <150 KB before quantisation
- INT8 weight quantisation, float32 I/O
- Role: real-time on-device inference

### Validation protocol

**AAMI EC57 inter-patient split (De Chazal 2004)**
- DS1 (training): 21 patients
- DS2 (testing): 20 patients
- Paced records excluded: 102, 104, 107, 217
- SMOTE applied to training set only — test set never touched
- Classes: Normal (N) vs Abnormal (S/V/F)

---

## Results

| Model | Accuracy | AUC-ROC | Role |
|---|---|---|---|

| 1D CNN (edge) | — | **0.908** | ESP32-S3 deployment |

*Exact accuracy and sensitivity values print at end of Cell 14 / Cell 17 — varies slightly by run due to SMOTE randomisation.*

**Model size:** TFLite INT8 weights, float32 I/O — fits on ESP32-S3 N16R8 (16 MB flash, 8 MB PSRAM, 200 KB arena)

---

## The TFLite INT8 Bug — and the fix

Running full INT8 quantisation (INT8 weights + INT8 I/O) on ESP32-S3 with TFLite 2.19 causes **zero output** — the model runs without error but returns 0.0 for every inference.

**Root cause:** Quantisation parameter mismatch at the inference layer. The input/output tensors are quantised to INT8 but the dequantisation scale factors are not correctly propagated on ESP32-S3 with this TFLite version.

**Fix:** INT8 weights + float32 I/O — do NOT set `inference_input_type` or `inference_output_type` to `tf.int8`.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# NOTE: Do NOT add inference_input/output_type = int8
# Keeps I/O as float32 — prevents zero-output bug on ESP32-S3
tflite_model = converter.convert()
```

This is undocumented in TFLite 2.19 release notes.



## How to run the training pipeline

1. Open `training/ArrhythmiaGuard_Complete.py` in Google Colab
2. Set runtime to **T4 GPU**
3. Run Cell 1 → **Restart runtime** → Run Cell 2 onwards
4. MIT-BIH dataset downloads automatically via `wfdb`
5. Cell 17 downloads all outputs: `cnn_model.h`, `rf_model.pkl`, results plots

## How to flash firmware

```bash
# Arduino IDE settings
# Board    : ESP32S3 Dev Module
# Flash    : 16MB (128Mb)
# PSRAM    : OPI PSRAM
# Open firmware/main.ino and upload
```

---

## Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet. 48 half-hour ECG recordings at 360 Hz. Inter-patient split follows AAMI EC57 standard (De Chazal et al., 2004).

No manual download needed — `wfdb.dl_database('mitdb', ...)` handles it automatically.

---

## Tech stack

`ESP32-S3` `AD8232` `TFLite Micro` `TinyML` `INT8 quantisation` `1D CNN` `Random Forest` `SHAP` `SMOTE` `AAMI EC57` `MIT-BIH` `wfdb` `librosa` `TensorFlow/Keras` `Streamlit` `Python` `Embedded C`
