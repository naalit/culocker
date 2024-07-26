from transformers import AutoModel
from PIL import Image
#from timm.data.transforms_factory import create_transform
import timm
import requests
import time
import torch

#model = AutoModel.from_pretrained("nvidia/MambaVision-L2-1K", trust_remote_code=True)
model = timm.create_model('convnextv2_base.fcmae', pretrained=True)
#model = model.eval()

# eval mode for inference
model.cuda().eval()

for frame_num in range(10):
    start_time = time.time()
    #print("frame", frame_num)
    image = Image.open(f'frames/{frame_num}.jpg')

    # prepare image for the model
    #url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
    #image = Image.open(requests.get(url, stream=True).raw)
    input_resolution = (3, 1920, 1080)  # MambaVision supports any input resolutions

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    inputs = transforms(image).unsqueeze(0).cuda()
    # model inference
    output = model(inputs)
    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    print("got:", top5_probabilities, top5_class_indices)

    print(f"{frame_num},{time.time() - start_time}")

    time.sleep(0.5)


# from transformers import DetrImageProcessor, DetrForObjectDetection
# import torch
# from PIL import Image
# import requests
# import time
# 
# #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# #image = Image.open(requests.get(url, stream=True).raw)
# 
# # you can specify the revision tag if you don't want the timm dependency
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)
# 
# 
# for frame_num in range(5):
#     start_time = time.time()
#     #print("frame", frame_num)
#     image = Image.open(f'frames/{frame_num}.jpg')
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     outputs = model(**inputs)
# 
#     # convert outputs (bounding boxes and class logits) to COCO API
#     # let's only keep detections with score > 0.9
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
# 
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         print(
#                 f"Detected {model.config.id2label[label.item()]} with confidence "
#                 f"{round(score.item(), 3)} at location {box}"
#         )
# 
#     print(f"{frame_num},{time.time() - start_time}")
# 
#     time.sleep(0.5)



# from sahi import AutoDetectionModel
# from sahi.utils.file import download_from_url
# from sahi.predict import get_prediction
# from PIL import Image
# 
# MMDET_YOLOX_TINY_MODEL_URL = "https://huggingface.co/fcakyon/mmdet-yolox-tiny/resolve/main/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
# MMDET_YOLOX_TINY_MODEL_PATH = "yolox.pt"
# MMDET_YOLOX_TINY_CONFIG_URL = "https://huggingface.co/fcakyon/mmdet-yolox-tiny/raw/main/yolox_tiny_8x8_300e_coco.py"
# MMDET_YOLOX_TINY_CONFIG_PATH = "config.py"
# IMAGE_URL = "https://user-images.githubusercontent.com/34196005/142730935-2ace3999-a47b-49bb-83e0-2bdd509f1c90.jpg"
# 
# # download weight and config
# download_from_url(
#   MMDET_YOLOX_TINY_MODEL_URL,
#   MMDET_YOLOX_TINY_MODEL_PATH,
# )
# download_from_url(
#   MMDET_YOLOX_TINY_CONFIG_URL,
#   MMDET_YOLOX_TINY_CONFIG_PATH,
# )
# 
# # create model
# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='mmdet',
#     model_path=MMDET_YOLOX_TINY_MODEL_PATH,
#     config_path=MMDET_YOLOX_TINY_CONFIG_PATH,
#     confidence_threshold=0.5,
#     device="cuda:0", # or 'cpu'
# )
# 
# # prepare input image
# image = Image.open(IMAGE_URL)
# 
# # perform prediction
# prediction_result = get_prediction(
#   image=image,
#   detection_model=detection_model
# )
# 
# # visualize predictions
# prediction_result.export_predictions(export_dir='results/')
# 
# # get predictions
# prediction_result.object_prediction_list
