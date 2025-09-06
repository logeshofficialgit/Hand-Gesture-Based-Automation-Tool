import cv2
import mediapipe as mp
import os

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Load images from the 'images' folder
image_folder = 'images'
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
current_index = 0
rotate_angle = 0

def count_fingers(hand_landmarks):
    finger_tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[finger_tips_ids[0]].x < hand_landmarks.landmark[finger_tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip_id in finger_tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

def apply_image_effect(image, fingers_count):
    global current_index, rotate_angle

    if fingers_count == 1:
        current_index = (current_index + 1) % len(image_files)
    elif fingers_count == 2:
        current_index = (current_index - 1) % len(image_files)
    elif fingers_count == 3:
        rotate_angle = (rotate_angle + 90) % 360
    elif fingers_count == 4:
        image = cv2.GaussianBlur(image, (15, 15), 0)

    return image

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    image = cv2.imread(os.path.join(image_folder, image_files[current_index]))
    image = cv2.resize(image, (w, h))

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers_up = count_fingers(hand_landmarks)

            if fingers_up in [1, 2, 3, 4]:
                image = apply_image_effect(image, fingers_up)

    if rotate_angle != 0:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

    combined = cv2.hconcat([frame, image])
    cv2.imshow("Webcam Feed + Image Viewer", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
