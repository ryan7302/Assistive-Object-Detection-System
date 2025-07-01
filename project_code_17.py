'''
Written by Paing Htet Kyaw -up2301555
'''
import sys
import socket
import cv2
import os
os.environ['ORT_INTRA_OP_NUM_THREADS'] = '2'  # Set 2 cores for Ultralytics ONNX before importing 
from ultralytics import YOLO
from icm20948 import ICM20948
from ahrs.filters import Madgwick
from sort import Sort
from scipy.optimize import linear_sum_assignment
import pyaudio
import json
from vosk import Model, KaldiRecognizer
import pyttsx3
import threading
import queue
import numpy as np
import time
import psutil
import csv

# cpu core monitoring
def monitor_cpu(interval=5):
    log_filename = f"cpu_usage_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    print(f"Starting CPU monitoring, logging to {log_filename}")
    with open(log_filename, 'w', newline='') as csvfile:
        num_cores = psutil.cpu_count()
        fieldnames = ['timestamp'] + [f'core{i}' for i in range(num_cores)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            usage = psutil.cpu_percent(interval=interval, percpu=True)
            row = {'timestamp': time.time()}
            for i, u in enumerate(usage):
                row[f'core{i}'] = u
            writer.writerow(row)
            csvfile.flush()

class TTSEngine:
    def __init__(self):
        # Voice Settings
        self.engine = pyttsx3.init(driverName='espeak')
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 1.0) 
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[12].id)  # A specific voice

       # Queues and Flags
        self.queue = queue.PriorityQueue()
        self.running = True

    def say(self, message, priority=2):
        self.queue.put((priority, message))

    def worker(self):
        while self.running:
            try:
                _, msg = self.queue.get(timeout=0.1)
                print(f"TTS: {msg}")
                self.engine.say(f"{msg} .")
                self.engine.runAndWait()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.engine.stop()

class IMUTracker:
    def __init__(self):
        # IMU Initilization 
        self.imu = ICM20948()
        self.madgwick = Madgwick(frequency=100, beta=0.1)
        
        # Calibration parameters
        self.accel_offsets = np.zeros(3)
        self.gyro_offsets = np.zeros(3)
        
        # Tracking states
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.rotation_matrix = np.eye(3)
        
        # Sensor fusion parameters
        self.ZUPT_THRESHOLD = 0.05  # m/s^2
        self.ZUPT_HYSTERESIS = 0.02  # m/s^2
        self.Zupt_active = False
        self.DT = 0.01
        self.GRAVITY = 9.81
        
        # Message to PC
        self.PC_IP = '192.168.33.178' 
        self.PORT = 9999 
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Threadlock and Flags 
        self.running = False
        self.lock = threading.Lock()

    def start_tracking(self):
        """Start continuous tracking thread"""
        self.running = True
        self.calibrate()
        threading.Thread(target=self.calculation, daemon=True).start()
        threading.Thread(target=self.send_data, daemon=True).start()

    def read_raw_data(self):
        """Read raw sensor data from IMU"""
        ax, ay, az, gx, gy, gz = self.imu.read_accelerometer_gyro_data()
        accel_m_s2 = np.array([ax, ay, az]) * 9.81 # Convert g to m/s^2
        return accel_m_s2, np.array([gx, gy, gz])

    def calibrate(self, samples=500):
        print("Calibrating IMU - keep camera stationary...")
        accel_data, gyro_data = [], []
        
        for _ in range(samples):
            a, g = self.read_raw_data()
            accel_data.append(a)
            gyro_data.append(g)
            time.sleep(0.002)
            
        # Gravity is removed from Z-axis assuming IMU is lying flat
        self.accel_offsets = np.mean(accel_data, axis=0) - np.array([0, 0, self.GRAVITY])
        self.gyro_offsets = np.mean(gyro_data, axis=0)
        
        self.accel_noise_std = np.std(accel_data, axis=0).mean()  # Measure noise
        self.ZUPT_THRESHOLD = 2.5 * self.accel_noise_std # Dynamic threshold (~0.05 m/s^2)
        print(f"Calibration complete. \n Gyro offsets (deg/s): {self.gyro_offsets} \n Accel Offsets (m/s²): {self.accel_offsets}")

    def calculation(self):
        """Main sensor fusion loop running in a background thread"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            if dt < self.DT:
                time.sleep(0.001)
                continue
                
            last_time = current_time
            
            # Read and calibrate sensor data
            raw_accel, raw_gyro = self.read_raw_data()
            accel_calibrated = raw_accel - self.accel_offsets
            gyro_calibrated = raw_gyro - self.gyro_offsets
            gyro_rad = np.radians(gyro_calibrated) # deg/s to rad/s

            with self.lock:
                # Update orientation filter
                self.orientation = self.madgwick.updateIMU(self.orientation, gyr=gyro_rad, acc=accel_calibrated)
                
                # Get rotation matrix
                self.rotation_matrix = self.quaternion_to_rotation_matrix(self.orientation)
                
                # Transform acceleration to world frame
                world_accel = self.rotation_matrix @ accel_calibrated
                
                # Subtract gravity in world frame
                motion_accel = world_accel - np.array([0, 0, self.GRAVITY])

                # Apply Zero Velocity Update (ZUPT)
                accel_norm = np.linalg.norm(motion_accel)
                
                if not self.Zupt_active and accel_norm < self.ZUPT_THRESHOLD:
                    self.velocity[:] = 0
                    self.Zupt_active = True
                elif self.Zupt_active and accel_norm > (self.ZUPT_THRESHOLD + self.ZUPT_HYSTERESIS):
                    self.Zupt_active = False 
                else:
                    self.velocity += motion_accel * dt 
                
                self.velocity = np.clip(self.velocity * 0.995, -5.0, 5.0) # 0.5% decay per update
                self.velocity[np.abs(self.velocity) < 0.01] = 0

                #print(f"Raw Accel: {raw_accel}, World Accel: {motion_accel}, Velocity: {self.velocity}")
                #camera_speed = np.linalg.norm(self.velocity)
                #print(f"Camera Speed: {camera_speed} m/s")  

                self.position += self.velocity * dt  # Update position

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to 3x3 rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
        w, x, y, z = q
        roll = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2)))
        pitch = np.degrees(np.arcsin(2*(w*y - z*x)))
        yaw = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2)))
        return np.array([roll, pitch, yaw])

    def get_pose(self):
        """Get current camera pose with thread-safe access"""
        with self.lock:
            return {
                'rotation': self.rotation_matrix.copy(),
                'velocity': self.velocity.copy(),
                'position': self.position.copy(),
                'euler': self.quaternion_to_euler(self.orientation)
            }

    def send_data(self):
        while self.running:
            # Convert to comma-separated string
            message = ','.join([f"{val:.5f}" for val in self.orientation])
            
            # Send to PC
            self.sock.sendto(message.encode(), (self.PC_IP, self.PORT))

            time.sleep(0.01)  # 100 Hz

    def detect_camera_orientation(self):
        """Get human-readable orientation description"""
        roll, pitch, yaw = self.quaternion_to_euler(self.orientation)
        
        orientation = []
        if abs(roll) > 170:
            orientation.append("upside down")
        elif roll > 15:
            orientation.append("tilted left")
        elif roll < -15:
            orientation.append("tilted right")
            
        if pitch > 15:
            orientation.append("pointing up")
        elif pitch < -15:
            orientation.append("pointing down")
            
        if abs(yaw) > 45:
            orientation.append(f"rotated {abs(yaw):.0f}Â° {'left' if yaw > 0 else 'right'}")
            
        return "Camera is " + ("level" not in orientation and " ".join(orientation) or "level")
    
    def stop_tracking(self):
        self.running = False

class CameraFeed:
    def __init__(self):        
        self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)  # USB camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)        
        
        self.detect_cont_mode = False
        self.detected_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.msg = None
        self.running = True
        self.frame_lock = threading.Lock() 

    def draw_grid(self, frame):
        """Draw a 3x3 grid on the frame."""
        height, width, _ = frame.shape
        # Vertical lines
        cv2.line(frame, (width // 3, 0), (width // 3, height), (255, 255, 255), 1)
        cv2.line(frame, (2 * width // 3, 0), (2 * width // 3, height), (255, 255, 255), 1)
        # Horizontal lines
        cv2.line(frame, (0, height // 3), (width, height // 3), (255, 255, 255), 1)
        cv2.line(frame, (0, 2 * height // 3), (width, 2 * height // 3), (255, 255, 255), 1)

    def start_feed(self):
        """Continuously capture and display the camera feed with a 3x3 grid."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue 
        
            if self.detect_cont_mode:
                frame_bgr = self.detected_frame 
            else:
                frame_bgr = frame
                
            self.draw_grid(frame_bgr)  # Draw 3x3 grid
            cv2.imshow("Pi Camera", frame_bgr)  # Display the frame

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to exit
                break
            elif key == ord('d'):
                print("Key 'd' pressed: Setting msg to 'detect'")
                self.msg = 'detect'     
            elif key == ord('s'):
                print("Key 's' pressed: Setting msg to 'start'")
                self.msg = 'start'     
            elif key == ord('t'):
                print("Key 't' pressed: Setting msg to 'stop'")
                self.msg = 'stop'         
            elif key == ord('e'):
                print("Key 'e' pressed: Setting msg to 'exit'")
                self.msg = 'exit'
            time.sleep(0.003)  # ~30 FPS
            
        self.cap.release()
        cv2.destroyAllWindows()
    
    def get_frame(self):
        with self.frame_lock:
            ret, frame = self.cap.read()
            if not ret:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop_feed(self):
        self.running = False

class ObjectDetection:
    def __init__(self, tts_engine, imu_tracker, camera_feed):
        #self.model = YOLO("yolo11n.pt")
        #self.model.export(format="onnx")
        self.onnx_model = YOLO("yolo11n.onnx", task='detect')
        self.tts_engine = tts_engine
        self.imu_tracker = imu_tracker # Camera Tracking
        self.img_tracker = Sort() #Object Tracking
        self.camera = camera_feed
        
        self.prev_positions = {}
        self.prev_time = time.time()
        
        # Warning Configuration
        self.last_warning_time = {}
        self.WARNING_DELAY = 2.0
        
        self.detect_mode = False
        self.detect_cont_mode = False
        self.detected_results = []
        self.last_announced_objects = []
        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()

        self.track_id_to_name = {} 
        self.iou_threshold = 0.2 
        self.last_detections = []  # Store raw detections for fallback
         
        self._detection_thread = threading.Thread(target=self._run_detection_loop, daemon=True)
        self._detection_thread.start()

        # Configurations
        self.focal_length = 3.04e-3  # meters
        self.sensor_width = 3.68e-3   # meters
        self.frame_width = 1280       # pixels
        self.track_threshold = 50 # pixels/frame

    def get_location(self, x_center, y_center, frame_width, frame_height):
        """Determine the object's location based on its center coordinates and camera orientation."""
        # Get current camera orientation
        pose = self.imu_tracker.get_pose()
        roll, pitch, yaw = pose['euler']
        
        # Adjust coordinates based on camera rotation
        if abs(roll) > 170:  # Upside down
            y_center = frame_height - y_center  # Flip vertical axis
        
        # Rotate coordinates based on yaw (simple approximation)
        if abs(yaw) > 45:
            x_center, y_center = y_center, frame_width - x_center
        
        horizontal_region = "center"
        vertical_region = "middle"

        if x_center < frame_width / 3:
            horizontal_region = "left"
        elif x_center > 2 * frame_width / 3:
            horizontal_region = "right"

        if y_center < frame_height / 3:
            vertical_region = "top"
        elif y_center > 2 * frame_height / 3:
            vertical_region = "bottom"
        return f"{vertical_region} {horizontal_region}"
    
    def perform_detection(self, frame):
        """Detect objects, compensate for camera motion, and estimate object speed."""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.onnx_model(frame_bgr, conf=0.5)
        frame_height, frame_width, _ = frame_bgr.shape

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        # Get camera motion estimate
        camera_velocity = self.imu_tracker.velocity.copy()
        camera_speed = np.linalg.norm(camera_velocity)
        camera_moving = camera_speed > 0.05
        avg_object_distance = 1.0 
        px_per_m = (self.frame_width * self.focal_length) / (self.sensor_width * avg_object_distance)
        camera_shift_px = camera_velocity * px_per_m * dt
        
        # Collect current frame detections
        current_detections = []
        class_names = []
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                name = result.names[cls]
                detections.append([x1, y1, x2, y2, conf])
                current_detections.append({
                    "box": [x1, y1, x2, y2],
                    "name": name
                })
                class_names.append(name)

        # Store raw detections for fallback
        self.last_detections = current_detections

        # Update tracker and get tracks
        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = self.img_tracker.update(dets_np)

        # Match tracks to detections using IoU
        updated_mapping = {}
        if len(current_detections) > 0:
            iou_matrix = np.zeros((len(tracks), len(current_detections)))
            for t_idx, track in enumerate(tracks):
                for d_idx, det in enumerate(current_detections):
                    iou_matrix[t_idx, d_idx] = self._calculate_iou(track[:4], det["box"])
            
            # Hungarian algorithm for optimal matching
            track_indices, det_indices = linear_sum_assignment(-iou_matrix)
            
            for t_idx, d_idx in zip(track_indices, det_indices):
                if iou_matrix[t_idx, d_idx] > self.iou_threshold:  # IoU threshold
                    track_id = int(tracks[t_idx][4])
                    updated_mapping[track_id] = current_detections[d_idx]["name"]

        # Update persistent tracking information
        for track in tracks:
            track_id = int(track[4])
            if track_id not in updated_mapping:
                # Retain previous name if exists, else mark as unknown
                updated_mapping[track_id] = self.track_id_to_name.get(track_id, "unknown object")
        
        self.track_id_to_name = updated_mapping

        with self.lock:
            self.detected_results.clear()
            if len(tracks) == 0 and len(self.last_detections) > 0:
                # Fallback to raw detections if tracking fails
                print("Falling back to raw detections")
                for det in self.last_detections:
                    x1, y1, x2, y2 = det["box"]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    location = self.get_location(cx, cy, frame_width, frame_height)
                    name = det["name"]
                    self.detected_results.append((name, location))
            else:
                for track in tracks:
                    x1, y1, x2, y2, track_id = map(int, track)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    location = self.get_location(cx, cy, frame_width, frame_height)
                    name = self.track_id_to_name.get(track_id, "unknown object")

                    # Speed calculation
                    speed = 0
                    moving = False
                    if track_id in self.prev_positions:
                        px, py = self.prev_positions[track_id]
                        dx = cx - px
                        dy = cy - py
                    
                        if camera_moving:
                            object_motion = np.linalg.norm([dx, dy]) # Scalar relative motion
                            camera_motion = np.linalg.norm(camera_shift_px[:2])  # Only use X/Y components
                            relative_motion = object_motion - camera_motion
                            speed = max(relative_motion, 0) / dt
                            moving = relative_motion > self.track_threshold
                        else:
                            speed = np.linalg.norm([dx, dy]) / dt
                            moving = speed > self.track_threshold

                     
                    self.prev_positions[track_id] = (cx, cy)
                    self.detected_results.append((name, location))

                    # Draw annotations and provide Warning
                    if self.detect_cont_mode:
                        if name == "car" and moving:                       
                            current_time = time.time()
                            last_time = self.last_warning_time.get(track_id, 0)
                            if current_time - last_time >= self.WARNING_DELAY:
                                self.tts_engine.say(f"Warning: {name} is moving at {location}", priority=1)
                                self.last_warning_time[track_id] = current_time                     
                                
                        color = (0, 255, 0) if not moving else (0, 0, 255)
                        label = f"{name} {'moving' if moving else 'still'} {speed:.1f}px/s"
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_bgr, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        with self.frame_lock:
                            self.camera.detected_frame = frame_bgr.copy()
                            self.camera.detect_cont_mode = True
                        
                # Clean up old track IDs
                current_track_ids = set(int(track[4]) for track in tracks)
                for track_id in list(self.last_warning_time.keys()):
                    if track_id not in current_track_ids:
                        del self.last_warning_time[track_id]
                for track_id in list(self.prev_positions.keys()):
                    if track_id not in current_track_ids:
                        del self.prev_positions[track_id]
 
            if self.detect_cont_mode:
                with self.frame_lock:
                    self.camera.detected_frame = frame_bgr.copy()
                    self.camera.detect_cont_mode = True
            else:
                self.camera.detect_cont_mode = False

            #print (f'Camera Moving at {camera_speed:.2f} m/s')
            #print(f"Detected results: {self.detected_results}")  # Debug log  
            
            # Handle single detection mode
            if self.detect_mode:
                if not self.detected_results:
                    self.tts_engine.say("No objects detected.")
                else:
                    counts = {}
                    for name, _ in self.detected_results:
                        counts[name] = counts.get(name, 0) + 1
                    
                    summary = []
                    for name, count in counts.items():
                        if count == 1:
                            summary.append(f"{count} {name}")
                        else:
                            plural = name+"s" if not name.endswith("s") else name
                            summary.append(f"{count} {plural}")
                    
                    self.tts_engine.say(f"Detected: {', '.join(summary)}.")
                self.detect_mode = False
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _run_detection_loop(self):
        while True:
            if self.detect_cont_mode or self.detect_mode:
                frame = self.camera.get_frame()
                self.perform_detection(frame)
                self.detect_mode = False
            time.sleep(0.01)

class VoiceCommand:
    def __init__(self, camera_feed, object_detection, tts_engine, imu_tracker):
        self.camera = camera_feed
        self.imu_tracker = imu_tracker
        self.object_detection = object_detection            
        self.tts_engine = tts_engine
        self.running = True

        # Vosk setup
        self.model_path = "vosk-model-small-en-us-0.15"
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)  # 16kHz sample rate
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8192
        )
        self.stream.start_stream()

    def listen_and_respond(self):
        """Listen for voice commands using Vosk and respond accordingly."""
        print("Loading Vosk model and starting speech recognition...")
        print("Listening for commands...")

        while self.running:
            try:
                # Read audio data from the stream
                data = self.stream.read(4096, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    # Get the final recognition result
                    result = json.loads(self.recognizer.Result())
                    command = result.get("text", "").lower().strip()
                    if command:
                        print(f"Command received: {command}")
                        self.process_command(command)
                    elif self.camera.msg:
                        print(f"Command received: {self.camera.msg}")
                        self.process_command(self.camera.msg)
                        self.camera.msg = None
                else:
                    # Optionally handle partial results for real-time feedback
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "").lower().strip()
                    if partial_text:
                        print(f"Partial: {partial_text}", end='\r')

            except Exception as e:
                print(f"Error during speech recognition: {e}")
                time.sleep(0.1)  # Prevent tight loop on errors

    def process_command(self, command):
        """Process recognized voice commands."""
        if "start" in command:
            if not self.object_detection.detect_cont_mode:
                self.object_detection.detect_cont_mode = True
                self.tts_engine.say("Started continuous detection.")

        elif "stop" in command:
            if self.object_detection.detect_cont_mode:
                self.object_detection.detect_cont_mode = False
                self.tts_engine.say("Stopped continuous detection.")

        elif "detect" in command:
            if not self.object_detection.detect_cont_mode and not self.object_detection.detect_mode:
                self.object_detection.detect_mode = True

        elif any(obj in command for obj, _ in self.object_detection.detected_results):
            for obj, location in self.object_detection.detected_results:
                if obj in command:
                    self.tts_engine.say(f"{obj} is at {location}.")

        elif "exit" in command:
            print("Exiting system. Cleaning up resources...")
            self.tts_engine.say("Exiting system.")
            self.object_detection.detect_cont_mode = False
            self.object_detection.detect_mode = False
            self.camera.stop_feed()
            self.imu_tracker.stop_tracking()
            self.tts_engine.stop()
            self.stop()
            sys.exit()

    def stop(self):
        """Clean up Vosk and PyAudio resources."""
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        self.recognizer = None
        self.model = None

def main():
    try:
        """Main function to initialize and run the system."""    
        tts_engine = TTSEngine()
        imu_tracker = IMUTracker()
        camera_feed = CameraFeed()
        object_detection = ObjectDetection(tts_engine, imu_tracker, camera_feed) 
        voice_command = VoiceCommand(camera_feed, object_detection, tts_engine,imu_tracker)

        # Start TTS worker thread
        tts_thread = threading.Thread(target=tts_engine.worker, daemon=True)
        tts_thread.start()
        # Start IMU tracking thread
        imu_tracker.start_tracking()
        
        # Start camera feed thread
        camera_thread = threading.Thread(target=camera_feed.start_feed)
        camera_thread.start()

        # Start voice command listener
        voice_thread = threading.Thread(target=voice_command.listen_and_respond, daemon=True)
        voice_thread.start()
        
        # CPU monitoring thread
        monitor_thread = threading.Thread(target=monitor_cpu, args=(5,), daemon=True)
        monitor_thread.start()
            
    except KeyboardInterrupt:
        print("\nSystem Crash")
        print("Exiting system. Cleaning up resources...")
        object_detection.detect_cont_mode = False
        object_detection.detect_mode = False
        camera_feed.stop_feed()
        imu_tracker.stop_tracking()
        tts_engine.say("Exiting the system. Goodbye!")
        tts_engine.stop()  # Signal the TTS engine to stop
        voice_command.stop()
        sys.exit()

if __name__ == "__main__":
    main()
