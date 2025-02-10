import time
import torch
from PIL import Image
import depth_pro
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

# --------------------------------------------------
# 1. Set Up Device and Print Device Info
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# 2. Load the Model and Preprocessing Transforms; Move to GPU
# --------------------------------------------------
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device)
model.eval()  # Set model to evaluation mode

# --------------------------------------------------
# 3. Load and Preprocess the Images; Ensure Tensors Are on GPU
# --------------------------------------------------
image_path1 = "../../../Datasets/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000000.png"
image_path2 = "../../../Datasets/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000001.png"

# Load images and their focal lengths
image1, _, f_px1 = depth_pro.load_rgb(image_path1)
image2, _, f_px2 = depth_pro.load_rgb(image_path2)

# Apply the model's transforms
image1 = transform(image1)
image2 = transform(image2)

# Move images to GPU if they are PyTorch tensors
if torch.is_tensor(image1):
    image1 = image1.to(device)
if torch.is_tensor(image2):
    image2 = image2.to(device)

# Batch the images together. Assuming images have the same dimensions.
batched_images = image1#torch.stack([image1, image2])  # Shape: (2, C, H, W)

# Batch the focal lengths. Adjust this if your model requires a tensor.
batched_f_px = f_px1#[f_px1, f_px2]

print("Done with setup: model and images are loaded on", device)

# --------------------------------------------------
# 4. Define Batched Inference Function with Timing (Mixed Precision)
# --------------------------------------------------
def run_inference_batch(model, images, f_px, label=""):
    with torch.no_grad(), amp.autocast():
        start_time = time.time()
        # Run batched inference.
        # Ensure your model.infer supports batched inputs.
        batched_prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Batch inference time {label} (mixed precision): {elapsed:.3f} seconds")
    return batched_prediction

# --------------------------------------------------
# 5. Warm-Up Run (Batched)
# --------------------------------------------------
print("Warming up the model with one batched inference call...")
with torch.no_grad():
    _ = model.infer(batched_images, f_px=batched_f_px)
    torch.cuda.synchronize()

# --------------------------------------------------
# 6. Run Batched Inference and Time It
# --------------------------------------------------
print("Running batched inference on both images...")
batched_prediction = run_inference_batch(model, batched_images, batched_f_px, label="(Batch)")
batched_prediction = run_inference_batch(model, batched_images, batched_f_px, label="(Batch)")
batched_prediction = run_inference_batch(model, batched_images, batched_f_px, label="(Batch)")
batched_prediction = run_inference_batch(model, batched_images, batched_f_px, label="(Batch)")
batched_prediction = run_inference_batch(model, batched_images, batched_f_px, label="(Batch)")
batched_prediction = run_inference_batch(model, batched_images, batched_f_px, label="(Batch)")
batched_prediction = run_inference_batch(model, batched_images, batched_f_px, label="(Batch)")

# --------------------------------------------------
# 7. Print GPU Memory Summary
# --------------------------------------------------
print("\nGPU Memory Summary:")
print(torch.cuda.memory_summary(device=device, abbreviated=True))

# --------------------------------------------------
# 8. (Optional) Visualize a Depth Map Using Matplotlib
# --------------------------------------------------
# Assuming the batched output is a list (or dict) of predictions for each image.
# For example, if batched_prediction is a list of dicts:
if isinstance(batched_prediction, (list, tuple)):
    prediction1 = batched_prediction[0]
    prediction2 = batched_prediction[1]
elif isinstance(batched_prediction, dict):
    # If the model returns a dict of lists (e.g., "depth" is a list of depth maps)
    # adjust accordingly:
    prediction1 = {k: v[0] for k, v in batched_prediction.items()}
    prediction2 = {k: v[1] for k, v in batched_prediction.items()}
else:
    # Fallback: assume the batched_prediction is a single dict per image
    prediction1 = batched_prediction
    prediction2 = None

depth_map = prediction1.get("depth")
if depth_map is not None:
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap="plasma")
    plt.title("Depth Map for Image 1")
    plt.colorbar()
    plt.show()
else:
    print("No depth map found in the prediction for Image 1.")
