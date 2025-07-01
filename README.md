# Assistive-Object-Detection-System
real-time assistive system for Raspberry Pi with object detection, IMU tracking, and voice control, plus PC visualization and CPU monitoring.

This project is a real-time assistive system built for the Raspberry Pi with PC-based visualization of IMU. It integrates:

- Object Detection using YOLOv8 ONNX model
- IMU-based camera motion tracking
- Voice control using Vosk
- Real-time 3D orientation visualization on PC
- CPU monitoring and post-run usage analysis

---

## Overview

### Raspberry Pi:
- Runs `project_code_17.py`
- Performs object detection, motion tracking, voice recognition, and TTS
- Runs `Converter.py` for analyzing CPU logs for montiorization

### PC/Laptop:
- Runs `Motion.py` for visualizing real-time orientation

---

## Raspberry Pi Setup

### Hardware Requirements:
- Raspberry Pi 5 (64-bit OS) + at least 16GB sd card
- USB camera
- ICM20948 IMU (via I2C)
- Bluetooth Earbuds (Microphone & headphones or speaker)


### Software Requirements:
```bash
sudo apt update
sudo apt install python3-venv python3-pip python3-smbus i2c-tools ffmpeg espeak portaudio19-dev libatlas-base-dev -y
```

## Earbud Connection Setup

1. Open Bluetooth from the Raspberry Pi GUI and pair your Bluetooth Earbuds.
2. Set the earbud as the default speaker and microphone:
   - Right-click on the respective **sound** and **microphone** icons in the GUI and choose your earbud.


### Environment Setup:
```bash
python3 -m venv --system-site-packages myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Enable I2C via `sudo raspi-config` → Interface Options → I2C → Enable

### Required Files (Manual Steps):

#### 1. YOLO ONNX Model
Run the exporter script provided:
```bash
python3 pytorch2onnx.py
```
This generates `yolo11n.onnx`, which should be placed in the same directory as `project_code_17.py`.

#### 2. Vosk Speech Model
- Download from: https://alphacephei.com/vosk/models
- Use `vosk-model-small-en-us-0.15`
- Extract the folder and place it in the same directory as the main code

#### 3. SORT Tracker
```bash
wget https://raw.githubusercontent.com/abewley/sort/master/sort.py
```


## 4. Running `project_code_17.py`
-NOTE: The IMU should be laying flat during the calibration processing in order to get correct locational feedback

```bash
source myenv/bin/activate
python3 project_code_17.py
```

**Voice Commands:**
- `start`: Begin continuous detection
- `stop`: Stop detection
- `detect`: Single scan
- `exit`: Shut down the system
- Say an object name: It will announce its position

**Keyboard Commands(to be used by selecting Opencv window):**
- `s`-->`start`: Begin continuous detection
- `t`-->`stop`: Stop detection
- `d`-->`detect`: Single scan
- `e`-->`exit`: Shut down the system

CPU usage will be logged automatically every 5 seconds.

---------------------------------------------------------------------
## Additional Features

## PC Visualization: `Motion.py`

### Purpose: Live visual feedback of camera orientation using IMU quaternions sent via UDP.

### Setup on PC:
```bash
pip install vpython numpy
```

### Run:
```bash
python3 Motion.py
```
> Make sure both devices are on the same Wi-Fi network and the pc ip is replaced in `project_code_17.py` under class initization of IMU-Tracker.

---

## Post process CPU Analysis: `Converter.py`

### Setup:
```bash
pip install pandas matplotlib
```

### Run:
```bash
python3 Converter.py <cpu_log_file.csv>
```
This will generate a graph `cpu_log_file_plot.png`.

------------------------------------------------------------------------------------

## File Descriptions

| File                | Description |
|---------------------|-------------|
| `project_code_17.py`| Main Raspberry Pi program for object detection, IMU tracking, TTS, and voice commands |
| `Motion.py`         | PC visualizer that plots camera orientation in 3D using VPython |
| `Converter.py`      | Post-run analysis tool that plots per-core CPU usage over time from log file |
| `pytorch2onnx.py`   | Script to export YOLOv8 `.pt` model into ONNX format |
| `requirements.txt`  | All required Python packages with pinned versions |
| `sort.py`           | Object tracker used for persistence across frames |
| `yolo11n.onnx`      | Exported YOLO object detection model |
| `vosk-model-small-en-us-0.15/` | Folder containing pre-trained speech recognition model |

---

## Troubleshooting

- **No camera feed?**
  - Run `libcamera-hello` or `ls /dev/video*` to verify connection

- **Voice not working?**
  - Make sure `espeak` and `pyttsx3` are working and volume is up

- **Real-time 3D not showing?**
  - Ensure IP address in `project_code_17.py` (under `self.PC_IP`) matches your PC IP

- **YOLO export fails?**
  - Run `pip install onnx onnxruntime` before exporting

---

## 📅 Author
**Paing Htet Kyaw** (up2301555) – Final Year BEng Electronic Engineering, University of Portsmouth

---

## 🎓 Acknowledgements
- Ultralytics (YOLOv8)
- Vosk Speech Recognition
- VPython
- SORT Tracker by abewley
- ICM20948 IMU support
