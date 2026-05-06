'''
Written by Paing Htet Kyaw -up2301555
'''

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.export(format="onnx")
