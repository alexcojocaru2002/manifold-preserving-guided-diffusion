import json

import torch
from PIL import Image
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from transformers.image_transforms import to_pil_image

# -------- CONFIG --------
image_path = "image_0.png"  # <-- Replace with your image path
threshold = 0.7

# -------- LOAD MODEL --------
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# -------- PREPROCESS --------
transform = T.Compose([
    T.ToTensor()
])
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image)

# -------- INFERENCE --------
with torch.no_grad():
    print(img_tensor.min().item(), img_tensor.max().item())
    predictions = model([img_tensor])[0]

# -------- POSTPROCESS --------
labels = predictions["labels"]
scores = predictions["scores"]
boxes = predictions["boxes"]

# Load COCO class names
COCO_CLASSES = weights.meta["categories"]

# Filter by threshold
keep = scores > threshold

# Print predictions
for label, score, box in zip(labels[keep], scores[keep], boxes[keep]):
    class_name = COCO_CLASSES[label]
    print(f"{class_name:<15} score={score:.3f} box={box.tolist()}")

# -------- DRAWING --------
# Prepare label names
label_strs = [COCO_CLASSES[label] for label in labels[keep]]

# Convert image to uint8 for drawing
img_uint8 = (img_tensor * 255).clamp(0, 255).to(torch.uint8)

# Draw boxes
drawn = draw_bounding_boxes(
    img_uint8,
    boxes[keep],
    labels=label_strs,
    colors="red",
    width=3,
    font_size=16
)

# Convert back to PIL for display
result_img = to_pil_image(drawn)

# Show image
plt.figure(figsize=(8, 8))
plt.imshow(result_img)
plt.axis("off")
plt.show()


# -------- DRAWING JSON REFERENCE --------

# Load reference boxes and labels from json
with open('../../references/reference.json', 'r') as f:
    ref = json.load(f)

ref_boxes = torch.tensor(ref['image']['boxes'], dtype=torch.float)
ref_labels = ref['image']['labels']
ref_label_strs = [COCO_CLASSES[label] for label in ref_labels]

# Create black image of same size as input
width, height = image.size
black_img = Image.new("RGB", (width, height), (0, 0, 0))
black_tensor = T.ToTensor()(black_img) * 255  # convert to uint8 tensor

# Draw boxes on black image
ref_drawn = draw_bounding_boxes(
    black_tensor.to(torch.uint8),
    ref_boxes,
    labels=ref_label_strs,
    colors="green",
    width=3,
    font_size=16
)

ref_img = to_pil_image(ref_drawn)

plt.figure(figsize=(8, 8))
plt.imshow(ref_img)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
