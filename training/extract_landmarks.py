import random
import time
from cProfile import label

import numpy as np
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path

DATA_SET = Path('/Users/dhyansuresh/datasets/asl-alphabet/asl_alphabet_train/asl_alphabet_train/')


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.1)


with HandLandmarker.create_from_options(options) as landmarker:
    folder_path = sorted(list(DATA_SET.glob('*')))

    vectors = []
    labels = []

    for folder in folder_path:
        detected = 0
        images = sorted(list(folder.glob('*.jpg')))
        start = time.perf_counter()
        for image in tqdm(images):
            mp_image = mp.Image.create_from_file(str(image))
            hand_landmarks_results = landmarker.detect(mp_image)

            if (hand_landmarks_results.hand_landmarks) :
                detected += 1
                landmarks = hand_landmarks_results.hand_landmarks[0]
                coords = []
                for landmark in landmarks:
                    coords.append(landmark.x)
                    coords.append(landmark.y)
                    coords.append(landmark.z)
                vectors.append(coords)
                labels.append(folder.name)

        end = time.perf_counter() - start
        tqdm.write(f"Folder {folder.name}:")
        tqdm.write(f"Detected:{detected} / {len(images)} files.")
        tqdm.write(f"Image success rate: {(detected / len(images)):.1%}")
        tqdm.write(f"Total time: {end:.1f}s\n")

    vector_arr = np.array(vectors)
    labels_arr = np.array(labels)
    print(vector_arr.shape)
    print(labels_arr.shape)
    np.savez("data/landmarks.npz", vectors=vector_arr, labels=labels_arr)
