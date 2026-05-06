'''
Written by Paing Htet Kyaw -up2301555
'''
import os
import socket
import numpy as np
from vpython import canvas, box, vector, rate, arrow, label, color, button



# -------------------- UDP Setup --------------------#
UDP_IP = "0.0.0.0"  # Listen on all interfaces
PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, PORT))
sock.settimeout(2)


# -------------------- VPython Setup --------------------#
scene = canvas(title="IMU Axis Debugging", width=1200, height=800)
scene.range = 3
scene.forward = vector(1, 1, -1)
scene.up = vector (0, 0, 1)

# ----------------- World Frame ----------------- #
# X (Red), Y (Green), Z (Blue = Up)
arrow(pos=vector(-2, 0, 0), axis=vector(4, 0, 0), color=color.red, shaftwidth=0.1)
arrow(pos=vector(0, -2, 0), axis=vector(0, 4, 0), color=color.green, shaftwidth=0.1)
arrow(pos=vector(0, 0, -2), axis=vector(0, 0, 4), color=color.blue, shaftwidth=0.1)
label(pos=vector(2.5, 0, 0), text="WORLD X", color=color.red)
label(pos=vector(0, 2.5, 0), text="WORLD Y", color=color.green)
label(pos=vector(0, 0, 2.5), text="WORLD Z (UP)", color=color.blue)

# ----------------- IMU Object ----------------- #
imu_cube = box(pos=vector(0, 0, 0), length=1, height=0.2, width=1, opacity=0.7)

# Local axes with different colors
local_x = arrow(pos=imu_cube.pos, axis=vector(1,0,0), color=color.magenta, shaftwidth=0.1)
local_y = arrow(pos=imu_cube.pos, axis=vector(0,1,0), color=color.cyan, shaftwidth=0.1)
local_z = arrow(pos=imu_cube.pos, axis=vector(0,0,1), color=color.yellow, shaftwidth=0.1)
label(pos=local_x.axis*1.2, text="IMU X", color=local_x.color)
label(pos=local_y.axis*1.2, text="IMU Y", color=local_y.color)
label(pos=local_z.axis*1.2, text="IMU Z", color=local_z.color)

def stop_program():
    print("Quit button pressed.")
    os._exit(0)

# Create button
button(text='Quit', bind=stop_program)

# ----------------- Axis Transformation ----------------- #
def update_axes(q):
    w, x, y, z = q
    # X-axis
    right = vector(
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ).norm()
    # Y-axis
    forward = vector(
        2*(x*y - w*z),
        1 - 2*(x**2 + z**2),
        2*(y*z + w*x)
    ).norm()
    # Z-axis
    up = vector(
        2*(x*z + w*y),
        2*(y*z - w*x),
        1 - 2*(x**2 + y**2)
    ).norm()
    
    return right, forward, up

# ----------------- Main Loop ----------------- #
while True:
    try:
        data, _ = sock.recvfrom(1024)
        line = data.decode().strip()
        q = np.array([float(val) for val in line.split(',')])
        
        right, forward, up = update_axes(q)
        imu_cube.axis = forward
        imu_cube.up = up
        local_x.axis = right * 1.5
        local_y.axis = forward * 1.5
        local_z.axis = up * 1.5

        rate(100)
        
    except socket.timeout:
        print("Waiting for data...")
    except Exception as e:
        print("Error:", e)
    

