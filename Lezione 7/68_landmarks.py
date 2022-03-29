import cv2, numpy as np, dlib

# read image, changed in gray scale
img = cv2.imread("../materiale/woman.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# init fece detector
path_predictor = '../materiale/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_predictor)

faces = detector(img_gray)

# detect landmarks for each faces detected
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x,y))

        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("68 facial landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
