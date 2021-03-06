import os
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
path_images = "../../materiale/MediaPipeImages/pose"
count = 0
BG_COLOR = (192, 192, 192)  # gray
with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                  min_detection_confidence=0.5) as pose:  # model complexity = complessita del pose model, da 0 a 2.
    # piú aumenta complessitá, piú aumenta
    # accuratezza punto riferimento e latenza dell'inferenza
    # enable segmentation = se True, oltre punti riferimento posa, genera anche maschera segmentazione

    for file in os.listdir(path_images):
        image = cv2.imread(os.path.join(path_images, file))
        image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print("pose landmarks: ", results.pose_landmarks) # 33 punti riferimento

        if not results.pose_landmarks:
            continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )
        annotated_image = image.copy()

        # Draw segmentation on the image.To improve segmentation around boundaries,
        # consider applying a joint bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('annotated_image' + str(count) + '.png', annotated_image)

        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        # ciascun landmark = (x, y, z, visibility) = (larghezza, lunghezza, profonditá con orgine = profonditá fianchi, punto riferimento visibile o non visibile)
