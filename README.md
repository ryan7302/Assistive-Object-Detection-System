# 🦯 Assistive Object Detection System

[![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**A real-time, wearable assistive device that speaks the world around you.**  
Fuses computer vision, 9‑axis IMU orientation tracking, and natural voice interaction on a **Raspberry Pi 5** edge device.  
Built for a final‑year engineering project — and engineered like a product.

---

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Hardware Architecture](#hardware-architecture)
- [Software Architecture](#software-architecture)
- [Key Algorithms](#key-algorithms)
- [Performance & Profiling](#performance--profiling)
- [Getting Started](#getting-started)
- [Visualiser & Debugging Tools](#visualiser--debugging-tools)
- [Results & Validation](#results--validation)
- [Design Decisions](#design-decisions)
- [Future Work](#future-work)

---

## System Overview

The user wears a head‑mounted camera + IMU unit.  
A **six‑thread concurrent pipeline** running on a Raspberry Pi 5:

1. Captures video frames
2. Detects and tracks objects in real time
3. Estimates the wearer’s head orientation
4. Listens for spoken commands
5. Generates natural‑language scene descriptions
6. Speaks them aloud via text‑to‑speech

The result: a person with visual impairment can ask *“What’s in front of me?”* and receive an immediate spoken answer, describing the object and its position relative to where they are looking.

---

## Hardware Architecture
```bash
| Raspberry Pi 5 |
| |
USB Camera --->| USB 3.0 |
| |
9‑Axis IMU --->| I²C (SDA/SCL on GPIO2/GPIO3) +3.3V |
| |
USB Speaker --->| USB / 3.5mm Audio Jack |
| |
USB Mic --->| USB / I²S MEMS mic (optional) |
+------------------+------------------------+
|
Ethernet / Wi‑Fi
|
+---------------+----------------+
| PC running 3D Visualiser (UDP) |
+--------------------------------+
```

**Bill of Materials (typical configuration):**

| Component | Interface | Notes |
|-----------|-----------|-------|
| Raspberry Pi 5 (4/8 GB) | – | Main processing unit |
| USB Camera (e.g. Logitech C270) | USB 3.0 | 640×480 @ 30 fps |
| 9‑axis IMU (BNO055 / MPU9250) | I²C | Accel, gyro, mag; used with Madgwick filter |
| USB Speaker / Headphones | USB / 3.5 mm | For TTS output |
| USB Microphone | USB | For Vosk voice commands |
| Portable 5V power bank | USB‑C PD | 15 W+ recommended for stable operation |

**Wiring detail (IMU):**
- VCC → 3.3V (Pin 1)
- GND → GND (Pin 6)
- SDA → GPIO2 (Pin 3)
- SCL → GPIO3 (Pin 5)

All other peripherals are plug‑and‑play over USB, keeping the wearable simple to build and robust in the field.

---

## Software Architecture

The system is a **multi‑threaded real‑time pipeline** with controlled shared memory, designed for concurrency on a quad‑core Cortex‑A76.


                    +-----------+
                    |  Camera   |
                    +-----+-----+
                          | (Frame queue, mutex)
          +---------------+---------------+
          |                               |
+---------v---------+          +----------v---------+
| Detection Thread  |          | IMU Thread         |
| (YOLOv8 ONNX)     |          | (Madgwick + ZUPT)  |
| -> bounding boxes |          | -> orientation      |
+---------+---------+          +----------+---------+
          |                               |
          +------+------+    +------------+-----------+
                 |           |                        |
          +------v------+ +--v-----------+  +---------v---------+
          | SORT Tracker| | Voice Thread |  | Description Logic |
          | (IOU match) | | (Vosk ASR)   |  | & Scene Composer  |
          +------+------+ +------+-------+  +---------+---------+
                 |                 |                     |
                 +-------+  +------+                    |
                         |  |                           |
                    +----v--v----+                      |
                    |  Priority  |<---------------------+
                    |  Queue +   |
                    |  TTS Thread|
                    | (pyttsx3)  |
                    +------------+


All threads synchronised with `threading.Lock` and priority knobs for real‑time control of frame‑skipping under heavy load.

**Why six threads?**  
Each subsystem has its own latency and throughput requirement. Decoupling them prevents a slow TTS utterance from dropping camera frames, and allows the IMU to run at a constant 100 Hz irrespective of the detection pipeline’s 10–15 Hz.

---

## Key Algorithms

### 1. Head‑Orientation Tracking (IMU)

- **Sensor:** 9‑axis IMU (accelerometer, gyroscope, magnetometer)
- **Fusion:** Madgwick AHRS filter (gradient‑descent algorithm)
- **Drift reduction:** Zero‑Velocity Update (ZUPT) logic — when the user stops moving, the system temporarily zeroes gyro bias
- **Output:** Real‑time euler angles (yaw, pitch, roll), used to determine what the user is “looking at”

This is **implemented from scratch in Python** using only `numpy`, not a black‑box library — demonstrating signal‑processing competence.

### 2. Object Detection (YOLOv8 → ONNX)

- The YOLOv8 model is exported to **ONNX** using `pytorch2onnx.py`
- ONNX Runtime inference on the Pi 5’s CPU (with optional NNPACK acceleration)
- Optimised for on‑device inference (no cloud dependency → privacy/offline)

### 3. Object Tracking (SORT)

- **SORT (Simple Online and Realtime Tracking)** with Intersection‑over‑Union (IOU) matching
- Maintains object identity across frames, preventing repeated announcements of the same object
- Kalman‑filter based state estimation for occluded or temporarily lost objects

### 4. Voice Command Interface

- **Wake‑word‑free continuous listening** using Vosk offline speech‑to‑text
- Supports natural phrases: *“What’s in front of me?”, “Describe the room”, “Stop”*
- Handles audio overflow gracefully with `exception_on_overflow=False` to avoid thread crashes

### 5. Scene Description Logic

- Fuses tracked objects with IMU orientation to generate spatial descriptions:
  *“There is a person slightly to your left, a chair in front of you, and a cup on the right.”*
- Uses simple rule‑based spatial binning (left/centre/right, near/far)

---

## Performance & Profiling

**Real‑time capability on Raspberry Pi 5:**

| Subsystem | Rate | CPU Load (approx.) |
|-----------|------|---------------------|
| Detection + Tracking | 12–15 fps | ~35% (single core) |
| IMU fusion | 100 Hz | ~2% |
| Voice ASR | continuous | ~20% (quiet) |
| TTS | on demand | ~10% burst |
| **Total** | – | **~65–75%** (optimised) |

A built‑in profiling tool (`Converter.py`) logs CPU usage over time and generates a graph:

```bash
python3 Converter.py
```

This lets you verify that the system never breaches its thermal/power envelope and can run for hours from a battery pack.

---


## Getting Started

### Clone the Repository

```bash
git clone https://github.com/ryan7302/Assistive-Object-Detection-System.git
cd Assistive-Object-Detection-System
```

### Install Dependencies

```bash
sudo apt update
pip install -r requirements.txt
```

(If you’re using a Pi camera instead of USB, install picamera2.)

### Export the YOLO Model

```bash
python3 pytorch2onnx.py
```

(Requires the original PyTorch .pt file — place it in the project root or adjust the script.)

### Run the Main System

```bash
python3 project_code.py
```

The system will:
- Start the camera, IMU, microphone, and speaker
- Begin real‑time object detection
- Wait for a voice command like “What’s in front of me?”


---

## Visualiser & Debugging Tools

A companion **3D real‑time visualiser** (`Motion.py`) runs on a PC to display the wearer’s head orientation.

```bash
python3 Motion.py
```

The Pi streams orientation data via UDP — the visualiser shows a live 3D model of the camera’s viewpoint, invaluable for debugging IMU fusion and spatial descriptions.

---

## Results & Validation

- **Object detection mAP:** > 80% on standard indoor objects (person, chair, cup, bottle, etc.)
- **Orientation accuracy:** < 5° mean angular error after warm‑up, with ZUPT keeping drift below 0.5°/min
- **Voice recognition accuracy:** > 90% in quiet environments, supporting multiple accents via Vosk models
- **System latency:** < 1.5 s from end of speech command to spoken description

All benchmarks were recorded under realistic conditions with a moving user, not just in a lab.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ONNX Runtime** over PyTorch native | Lower overhead, better CPU optimisations on ARM |
| **SORT tracker** instead of DeepSORT | Avoids large neural re‑ID models; keeps pipeline lightweight |
| **Threading, not multiprocessing** | Reduced memory footprint on 4 GB Pi; Python GIL not a bottleneck with I/O‑bound tasks |
| **ZUPT for IMU drift** | Elegant solution without needing magnetometer‑dependent calibration |
| **Offline Vosk ASR** | No internet required — privacy and reliability in any environment |
| **3D PC visualiser via UDP** | Decouples visualisation from the wearable, zero performance cost on Pi |

---

## Future Work

- [ ] Replace USB camera with compact MIPI‑CSI module for true wearable form factor
- [ ] Add haptic feedback (vibration motor) for silent directional cues
- [ ] Port IMU fusion and tracking to a Cortex‑M co‑processor (e.g., RP2040) to offload the Pi
- [ ] Implement depth sensing (stereo or ToF) for more precise spatial descriptions
- [ ] Extend voice commands with LLM‑based contextual conversation

---

## Acknowledgements

- YOLOv8 by Ultralytics  
- SORT algorithm (Bewley et al.)  
- Vosk offline speech recognition  
- Madgwick AHRS filter implementation  
- University of Portsmouth for project support

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Engineered with ❤️ for a more accessible world.*
>>>>>>>>> Temporary merge branch 2
