import mediapipe as mp

# Task module: bên cạnh solution là model đc build sẵn, task cho phép sử dụng các model custom có thể chỉnh sửa sâu hơn
## BaseOptions: các option cho model: path đến file model, số thread sử dụng,..
base_options = mp.tasks.BaseOptions

# 1 loại task của MP phục vụ Image Segmentation
image_segmenter = mp.tasks.vision.ImageSegmenter

# 1 class con extend từ BaseOptions, thêm các option riêng cho ImageSegmenter
image_segmenter_options = mp.tasks.vision.ImageSegmenterOptions

# Enum quy chứa các mode khi chạy của Task vision
running_mode = mp.tasks.vision.RunningMode
