from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "../../materiale/Angelina_Jolie2.jpg"
# faces in a dict

faces = RetinaFace.detect_faces(img_path)

identity = faces['face_1']
facial_area = identity['facial_area']
landmarks = identity['landmarks']

# highlight facial area

img = cv2.imread(img_path)
cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 3)

# highlight the landmarks

cv2.circle(img, (int(landmarks['left_eye'][0]), int(landmarks['left_eye'][1])), 3, (0, 0, 255), -1)
cv2.circle(img, (int(landmarks['right_eye'][0]), int(landmarks['right_eye'][1])), 3, (0, 0, 255), -1)
cv2.circle(img, (int(landmarks['nose'][0]), int(landmarks['nose'][1])), 3, (0, 0, 255), -1)
cv2.circle(img, (int(landmarks['mouth_left'][0]), int(landmarks['mouth_left'][1])), 3, (0, 0, 255), -1)
cv2.circle(img, (int(landmarks['mouth_right'][0]), int(landmarks['mouth_right'][1])), 3, (0, 0, 255), -1)

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = np.array(image)
plt.imshow(pixels)
plt.show()