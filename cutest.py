import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoImageProcessor, AutoModelForDepthEstimation, DetrImageProcessor, DetrForObjectDetection
import time
import sys
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print('Warning: CUDA not available in PyTorch, using CPU instead', file=sys.stderr)

# trait Model { fn process(image: Image);  }
class MaskDepthModel:
    def __init__(self):
        # load Mask2Former fine-tuned on COCO panoptic segmentation
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-panoptic", device=device)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-panoptic").to(device)
        
        # load depth-anything
        self.d_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", device=device)
        self.d_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)

    def process(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        inputs = { 'pixel_values': inputs.pixel_values }
        d_inputs = self.d_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                outputs = self.model(**inputs)
                d_outputs = self.d_model(**d_inputs)
                predicted_depth = d_outputs.predicted_depth

        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        result = self.processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
        predicted_instance_map = result["segmentation"].to(device) # not sure why this goes on the cpu by default. can't the post processing happen on device? idk

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        for i in range(1,6):
            f = float(i)

            # Create a mask where predicted_instance_map equals instance_value
            mask = (predicted_instance_map == f)

            # Use the mask to select values from prediction
            masked_prediction = torch.where(mask, prediction, torch.tensor(float('-inf')))

            # Return the maximum value
            max_value = torch.max(masked_prediction).item()

            #print(str(f) + ':', max_value)

class ConvNetModel:
    def __init__(self):
        self.model = timm.create_model('convnextv2_base.fcmae', pretrained=True)
        self.model.cuda().eval()

    def process(self, image):
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
    
        inputs = transforms(image).unsqueeze(0).cuda()
        # model inference
        output = self.model(inputs)
        top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
        #print("got:", top5_probabilities, top5_class_indices)

class DetrModel:
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)

    def process(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # print(
            #         f"Detected {model.config.id2label[label.item()]} with confidence "
            #         f"{round(score.item(), 3)} at location {box}"
            # )
        

def run_experiment(model, n_frames):
    # per frame
    for frame_num in range(n_frames):
        start_time = time.time()
        #print("frame", frame_num)
        image = Image.open(f'frames/{frame_num}.jpg')

        model.process(image)

        print(f"{frame_num},{time.time() - start_time}")

def create_model():
    if len(sys.argv) > 1:
        if sys.argv[1] == "mask-depth":
            return MaskDepthModel()
        elif sys.argv[1] == "convnet":
            return ConvNetModel()
        elif sys.argv[1] == "detr":
            return DetrModel()
        else:
            print(f"unknown model name {sys.argv[1]}, defaulting to mask-depth")
    return MaskDepthModel()

if __name__ == "__main__":
    model = create_model()
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_experiment(model, n_frames)
