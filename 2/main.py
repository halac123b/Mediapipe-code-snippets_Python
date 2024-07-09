import mediapipe as mp
import PIL
import numpy as np

# Task module: bên cạnh solution là model đc build sẵn, task cho phép sử dụng các model custom có thể chỉnh sửa sâu hơn
## BaseOptions: các option cho model: path đến file model, số thread sử dụng,..
model_data = []  # Binary content của model, thường đọc từ file .tflite
# model_asset_buffer: data của model
base_options = mp.tasks.BaseOptions(model_asset_buffer=model_data)

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

image = PIL.Image.open("../assets/images/test2.png")
image_data = np.array(image)
# Mediapipe Image obj
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

# 1 loại task của MP phục vụ Image Segmentation
## Create from ImageSegmenterOptions tạo trc đó
with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
    # SegmentationResult
    segmented_masks = segmenter.segment(mp_image)
    # Lấy category_mask: 1 array chứa index của label mà từng pixel đc phân loại
    segmented_masks_result = segmented_masks.category_mask
    # Chuyển array trên thành dạng chuẩn Numpy
    masks_result_np_array = segmented_masks_result.numpy_view()
