import mediapipe as mp
import PIL
import numpy as np

# Read image and convert to Numpy array
img = PIL.Image.open("../assets/images/test2.png")
img = img.convert("RGB")
img = np.array(img)

# FaceMesh detection module of Mediapipe
## solutions: chứa các model đã đc train sẵn của Mediapipe
## static_image_mode: ảnh tĩnh chứ k phải video
## refine_landmarks: lọc lại giá trị landmark, tăng độ chính xác nhưng nặng performance, vì dùng ảnh tĩnh nên k cần
## max_num_faces: số face tối đa cần detect là 1
## min_detection_confidence: ngưỡng confindence khi detect
with mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
) as face_mesh:
    # Bắt đầu chạy nhận diện
    results = face_mesh.process(img)

print(results.__class__.__name__)
print(len(results.multi_face_landmarks))
