from retinaface import RetinaFace
import matplotlib.pyplot as plt

#align_True
faces = RetinaFace.extract_faces(img_path="../../materiale/Angelina_Jolie2.jpg", align=True)
for face in faces:
  plt.imshow(face)
  plt.show()


#align_False
faces = RetinaFace.extract_faces(img_path="../../materiale/Angelina_Jolie2.jpg", align=False)
for face in faces:
  plt.imshow(face)
  plt.show()