from collections import OrderedDict
import numpy as np, cv2, dlib, imutils

facial_features_cordinates = {}

# mapping string with features coordinates
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])


def shape_to_numpy_array(shape, dtype="int"):
    # initialize list of coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)
    # loop over 68 landmarks and convert them to a 2-tuple of (x,y)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x,y) coordinates associated with the face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if suppose draw jawline
        if name == "Jaw":
            # jawline non-enclosed facial region, draw lines between the (x, y) coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute convex hull of the landmarks coordinates and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply trasperent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    print(facial_features_cordinates)
    return output


path_predictor = '../materiale/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_predictor)

image = cv2.imread("../materiale/woman.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over face detections
for (i, rect) in enumerate(rects):
    # determine facial landmarks for the face region
    # convert landmark (x, y)-coordinates to Numpy array

    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)

    output = visualize_facial_landmarks(image, shape)
    cv2.imshow("facial landmarks", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

