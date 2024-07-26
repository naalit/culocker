import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoImageProcessor, AutoModelForDepthEstimation
import time
import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load Mask2Former fine-tuned on COCO panoptic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-panoptic", device=device)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-panoptic").to(device)

# load depth-anything
d_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", device=device)
d_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)

def run_experiment():
    # per frame
    for frame_num in range(50):
        start_time = time.time()
        #print("frame", frame_num)
        image = Image.open(f'frames/{frame_num}.jpg')
        inputs = processor(images=image, return_tensors="pt").to(device)
        inputs = { 'pixel_values': inputs.pixel_values }
        d_inputs = d_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                outputs = model(**inputs)
                d_outputs = d_model(**d_inputs)
                predicted_depth = d_outputs.predicted_depth

        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
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

        print(f"{frame_num},{time.time() - start_time}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        while True:
            run_experiment()
    else:
        run_experiment()
