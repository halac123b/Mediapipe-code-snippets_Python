import mediapipe as mp

# Task module: bên cạnh solution là model đc build sẵn, task cho phép sử dụng các model custom có thể chỉnh sửa sâu hơn
## BaseOptions: các option cho model: path đến file model, số thread sử dụng,..
model_data = []  # Binary content của model, thường đọc từ file .tflite
# model_asset_buffer: data của model
base_options = mp.tasks.BaseOptions(model_asset_buffer=model_data)

# 1 loại task của MP phục vụ Image Segmentation
image_segmenter = mp.tasks.vision.ImageSegmenter

# Enum quy chứa các mode khi chạy của Task vision
running_mode = mp.tasks.vision.RunningMode

# 1 class con extend từ BaseOptions, thêm các option riêng cho ImageSegmenter
options = mp.tasks.vision.ImageSegmenterOptions(
    # BaseOptions, vì class con nên vẫn cần inherit từ base class
    # model_asset_buffer: binary content of model (for this from .tflite file)
    base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
    # Runnding mode process 1 static image
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    # Trong quá trình segment, từng pixel sẽ đc phân loại thành các class
    ## Option này giúp tích hợp kết quả phân loại vào output để sau đó có thể sử dụng
    output_category_mask=True,
)
