from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = '../../materiale/img1.jpg'
faces = RetinaFace.detect_faces(img_path)

img = cv2.imread(img_path)
count = 1

for face in faces:
    identity = faces['face_'+ str(count)+'']
    facial_area = identity['facial_area']
    landmarks = identity['landmarks']

    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 3)
    cv2.circle(img, (int(landmarks['left_eye'][0]), int(landmarks['left_eye'][1])), 3, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks['right_eye'][0]), int(landmarks['right_eye'][1])), 3, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks['nose'][0]), int(landmarks['nose'][1])), 3, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks['mouth_left'][0]), int(landmarks['mouth_left'][1])), 3, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks['mouth_right'][0]), int(landmarks['mouth_right'][1])), 3, (0, 0, 255), -1)

    count = count + 1

print(count)

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = np.array(image)
plt.imshow(pixels)
plt.show()