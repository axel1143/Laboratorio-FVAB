import os, cv2, mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# for static images
path_images = '../../materiale/MediaPipeImages/'
count = 0
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, refine_landmarks=True,
                           min_detection_confidence=0.5) as face_mesh:  # static image mode (True = immagini singole,
    # False = frame di un video)
    # max num faces = numero massimo di volti da rilevare
    # refine landmarks = affina cordinate dei punti di riferimento intorno ad occhi e labbra, produce punti riferimento aggiuntivo


    for file in os.listdir(path_images):
        image = cv2.imread(os.path.join(path_images, file))

        # convert BGR to RGB before processing
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # print and draw face mesh landmarks on the image
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            print('face_landmarks:', face_landmarks)  # 468 punti riferimento 3D per ciascuna image
            # (ciascun punto = x larghezza, y altezza, z profonditá)
            mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,  # disegna mash
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,  # disegna contorni
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,  # disegna iridi
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        cv2.imwrite('FM_MediaPipe_' + str(count) + '.png', annotated_image)
        count = count + 1
