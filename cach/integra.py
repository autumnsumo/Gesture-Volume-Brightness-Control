import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Initializing the Hand Tracking Models
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0

# Main Loop
while True:
    # Read video frame by frame
    _, frame = cap.read()

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    Process = hands.process(frameRGB)

    landmarkList = []
    handSide = []  # To store the side of each detected hand

    # If hands are present in the image(frame)
    if Process.multi_hand_landmarks:
        # Detect hand landmarks
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                # Store height and width of the image
                height, width, color_channels = frame.shape

                # Calculate and append x, y coordinates of hand landmarks from image(frame) to landmarkList
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            # Draw Landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

            # Identify hand side based on the position of the thumb
            thumb_x = landmarkList[4][1]
            if thumb_x < width / 2:
                handSide.append("Left")
            else:
                handSide.append("Right")

    # If landmarks list is not empty
    if landmarkList:
        for hand, landmarks in zip(handSide, Process.multi_hand_landmarks):
            # Store x, y coordinates of (tip of) thumb
            x_1, y_1 = landmarks.landmark[4].x * width, landmarks.landmark[4].y * height

            # Store x, y coordinates of (tip of) index finger
            x_2, y_2 = landmarks.landmark[8].x * width, landmarks.landmark[8].y * height

            # Draw circle on thumb and index finger tip
            cv2.circle(frame, (int(x_1), int(y_1)), 7, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (int(x_2), int(y_2)), 7, (0, 255, 0), cv2.FILLED)

            # Draw line from tip of thumb to tip of index finger
            cv2.line(frame, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 255, 0), 3)

            # Calculate Euclidean distance
            L = hypot(x_2 - x_1, y_2 - y_1)

            if hand == "Right":  # Volume control for the right hand
                vol = np.interp(L, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)
                volBar = np.interp(L, [50, 220], [400, 150])
                volPer = np.interp(L, [50, 220], [0, 100])

                # Draw volume bar
                cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 3)
                cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

            elif hand == "Left":  # Brightness control for the left hand
                b_level = np.interp(L, [15, 220], [0, 100])
                sbc.set_brightness(int(b_level))

    # Display video and when 'q' is entered, destroy the window
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
