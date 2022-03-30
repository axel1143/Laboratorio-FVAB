from retinaface import RetinaFace # need with tensorflow 2.1.0

im_path = "../../materiale/Angelina_Jolie1.jpg"
faces = RetinaFace.detect_faces(im_path, threshold=0.5) # decrease to detect faces with low resolution
print(faces)