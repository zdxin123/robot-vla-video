import cv2
import mediapipe as mp
import numpy as np
import sys


mp_pose = mp.solutions.pose.Pose(static_image_mode=False)


def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            seq.append(keypoints)

    cap.release()
    return np.array(seq, dtype=np.float32)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_keypoints.py <video_path>")
        sys.exit(1)

    arr = extract_keypoints_from_video(sys.argv[1])
    print("Keypoints shape:", arr.shape)
