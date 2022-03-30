import os, cv2, mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# for static images:
path_images = '../../materiale/MediaPipeImages/'
count = 0

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:  # min_detection_confidace = confidenza minima affinché compito sia riuscito
    for file in os.listdir(path_images):
        image = cv2.imread(os.path.join(path_images, file))

        # Convert BGR image to RGB and processs it with mediapipe
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face
        if not results.detections:
            continue
        annotated_image = image.copy()
        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)
        # cv2.imwrite('FD_MediaPipe_'+ str(count)+ '.png', annotated_image)

        pixel_array = np.array(annotated_image)
        plt.imshow(annotated_image)
        plt.show()
        count = count+1

