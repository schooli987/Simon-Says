import cv2
import mediapipe as mp
import random
import time

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

FINGER_TIPS = [8, 12, 16, 20]

# --- gesture detection ---
def get_gesture(hand_landmarks):
    fingers = []
    for tip in FINGER_TIPS:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [0, 0, 0, 0]:
        return "Rock"
    elif fingers == [1, 1, 1, 1]:
        return "Paper"
    elif fingers == [1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"

# --- instructions ---
instructions = ["Rock", "Paper", "Scissors",
                "Simon says Rock", "Simon says Paper", "Simon says Scissors"]

expected_gesture = random.choice(instructions)
last_instruction_time = time.time()
INSTRUCTION_HOLD = 5   # round duration (seconds)

stable_gesture, stable_since = None, None
HOLD_TIME = 1.0
result_label = ""
gesture_text = "Waiting..."

# --- scoring ---
player_score = 0
WIN_SCORE = 5

cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- change instruction every X sec ---
    if time.time() - last_instruction_time > INSTRUCTION_HOLD:
        expected_gesture = random.choice(instructions)
        last_instruction_time = time.time()
        result_label = ""           # clear result before new round
        stable_gesture = None       # reset locked gesture
        gesture_text = "Waiting..." # reset message

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        detected_gesture = get_gesture(hand_landmarks)

        if detected_gesture == stable_gesture:
            if time.time() - stable_since >= HOLD_TIME:
                gesture_text = f"Locked: {detected_gesture}"

                if result_label == "":
                    if instructions.index(expected_gesture) > 2:
                        if detected_gesture == expected_gesture.split("Simon says ")[-1]:
                            result_label = "Correct!"
                            player_score += 1
                        else:
                            result_label = "Uh Oh! Wrong Move"
                            player_score-=1
                    else:
                        if detected_gesture:
                            result_label = "Uh Oh! Simon didn't say!"
                            player_score-=1
            
        else:
            stable_gesture, stable_since = detected_gesture, time.time()

    # --- UI Text ---
    cv2.putText(frame, f"Instruction: {expected_gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, gesture_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if result_label:
        cv2.putText(frame, result_label, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if result_label == "Correct!" else (0, 0, 255), 3)

    # --- Score Display ---
    cv2.putText(frame, f"Score: {player_score}/{WIN_SCORE}", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # --- Game Over ---
    if player_score >= WIN_SCORE:
        cv2.putText(frame, "YOU WIN!", (200, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
    elif player_score < 0:
        cv2.putText(frame, "YOU LOOSE", (200, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
        cv2.imshow("Simon Says RPS", frame)
        cv2.waitKey(5000)   # hold the screen for 5 sec
        break

    # --- Show Frame ---
    cv2.imshow("Simon Says RPS", frame)

    # ESC key to quit early
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
