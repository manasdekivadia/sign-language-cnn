import cv2 as cv
import mediapipe as mp
import csv
import copy

from slr.model.classifier import KeyPointClassifier
from slr.utils.pre_process import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark
)


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # Mediapipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Load classifier
    classifier = KeyPointClassifier()

    # Load labels
    with open("slr/model/label.csv", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        labels = [row[0] for row in reader]

    print("System ready.")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv.flip(frame, 1)
        debug_img = copy.deepcopy(frame)

        # Preprocess
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)

        sign_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Bounding box (for drawing only)
                brect = calc_bounding_rect(debug_img, hand_landmarks)

                # Convert to landmark list
                landmark_list = calc_landmark_list(debug_img, hand_landmarks)

                # Preprocess landmark list
                processed = pre_process_landmark(landmark_list)

                # Predict
                class_id = classifier(processed)
                sign_text = labels[class_id] if class_id < len(labels) else ""

                # Draw bounding box
                cv.rectangle(debug_img, (brect[0], brect[1]), (brect[2], brect[3]),
                             (0, 255, 0), 2)

                # Draw prediction
                cv.putText(debug_img, sign_text, (brect[0], brect[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show window
        cv.imshow("Sign Recognition (Simple)", debug_img)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
