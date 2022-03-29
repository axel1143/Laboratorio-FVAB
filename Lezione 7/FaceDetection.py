import cv2, numpy as np, dlib

# read image, changed in gray scale
img = cv2.imread("../materiale/woman.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# init fece detector
detector = dlib.get_frontal_face_detector()
faces = detector(img_gray, 1)  # secondo parametro = numero strati priamide dell'immagine da applicare durante upscaling, prima di applicare rilevatore

print(faces)
print("Number of faces detected:" , len(faces))

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imshow("face detection", img)
cv2.waitKey(0)
cv2.destroyWindow()
