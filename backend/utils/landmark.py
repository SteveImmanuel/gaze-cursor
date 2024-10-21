import mediapipe as mp
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_IRISES

def get_landmark_indices():
    CUSTOM_FACEMESH_CONTOURS = frozenset().union(*[
        FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
        FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_IRISES
    ])

    unique_idx = set()
    for item in CUSTOM_FACEMESH_CONTOURS:
        unique_idx.add(item[0])
        unique_idx.add(item[1])

    return list(unique_idx)

LANDMARK_INDICES = get_landmark_indices()

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.
            get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.
            get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.
            get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image

def get_model(model_path:str):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    return FaceLandmarker.create_from_options(options)

def extract_landmark_features(model, image: np.ndarray):
    # model from get_model
    # image in rgb format
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = model.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    if len(detection_result.face_landmarks) == 0:
        raise Exception('No face detected')
    
    face_landmarks = []
    for landmark in detection_result.face_landmarks[0]:
        face_landmarks.append([landmark.x, landmark.y, landmark.z])
    
    face_landmarks = np.array(face_landmarks)
    face_landmarks = face_landmarks[LANDMARK_INDICES]

    return face_landmarks, annotated_image

if __name__ == '__main__':
    model = get_model('../ckpts/face_landmarker.task')
    face_landmarks, annotated_image = extract_landmark_features(model)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('annotated_image.jpg', annotated_image)
    print(face_landmarks)
    print(face_landmarks.shape)