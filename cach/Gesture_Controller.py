import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import tkinter as tk
from PIL import Image, ImageTk

# Define constants for better readability
HAND_LEFT = "Left"
HAND_RIGHT = "Right"

# Define thresholds for gesture recognition
VOL_DIST_THRESHOLD = 100
BRIGHTNESS_DIST_THRESHOLD = 150

def update_controls():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    if Process.multi_hand_landmarks:
        for landmarks in Process.multi_hand_landmarks:
            # Extract hand landmarks
            landmarkList = []
            for _id, lm in enumerate(landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                landmarkList.append([_id, x, y])

            # Determine hand side
            thumb_x = landmarkList[4][1]
            hand_side = HAND_LEFT if thumb_x < w / 2 else HAND_RIGHT

            # Calculate distance between thumb and index finger
            thumb_index_dist = np.linalg.norm(np.array(landmarkList[4][1:]) - np.array(landmarkList[8][1:]))
            
            if hand_side == HAND_RIGHT and thumb_index_dist < VOL_DIST_THRESHOLD:
                # Adjust volume
                vol = np.interp(thumb_index_dist, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)
                volPer = np.interp(thumb_index_dist, [50, 220], [0, 100])
                volume_label.config(text=f'Volume: {int(volPer)}%')

            elif hand_side == HAND_LEFT and thumb_index_dist < BRIGHTNESS_DIST_THRESHOLD:
                # Adjust brightness
                b_level = np.interp(thumb_index_dist, [15, 220], [0, 100])
                sbc.set_brightness(int(b_level))
                brightness_label.config(text=f'Brightness: {int(b_level)}%')

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    video_label.img = img
    video_label.configure(image=img)
    video_label.after(10, update_controls)  # Update every 10 milliseconds

# Create the main window
root = tk.Tk()
root.title("Hand Gesture Control")
root.geometry("800x600")
root.configure(bg="#2C3E50")  # Background color

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Initialize the Hand Tracking Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Labels for volume and brightness
volume_label = tk.Label(root, text="Volume: 0%", font=("Helvetica", 14), bg="#2C3E50", fg="white")
volume_label.pack(pady=10)

brightness_label = tk.Label(root, text="Brightness: 0%", font=("Helvetica", 14), bg="#2C3E50", fg="white")
brightness_label.pack(pady=10)

# Label to display webcam feed
video_label = tk.Label(root, bg="#2C3E50")
video_label.pack()

# Start updating controls
update_controls()

# Run the main loop
root.mainloop()

# Release the webcam when the window is closed
cap.release()
cv2.destroyAllWindows()
