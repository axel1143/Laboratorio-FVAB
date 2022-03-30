from deepface import DeepFace

obj = DeepFace.verify("../../materiale/Angelina_Jolie1.jpg",
                      "../../materiale/Angelina_Jolie2.jpg",
                      model_name='ArcFace',
                      detector_backend='retinaface')

print(obj)