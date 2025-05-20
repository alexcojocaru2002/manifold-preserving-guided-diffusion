import clip
import torch
from PIL import Image

def style_loss(ref_img, curr_img, model):
    with torch.no_grad():
        image_features = model.encode_image(curr_img)
        image_ref_features = model.encode_image(ref_img)

    gram1 = torch.mm(image_features.t(), image_features)
    gram2 = torch.mm(image_ref_features.t(), image_ref_features)

    diff = gram1 - gram2
    norm = torch.linalg.norm(diff)

    return norm

model_name = "ViT-B/16"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)

astronaut = preprocess(Image.open("reference_old.png")).unsqueeze(0).to(device)
church1 = preprocess(Image.open("ref1.png")).unsqueeze(0).to(device)
church2 = preprocess(Image.open("reference.png")).unsqueeze(0).to(device)

print("astronaut + church1: ", style_loss(astronaut, church1, model).item())
print("astronaut + church2: ", style_loss(astronaut, church2, model).item())
print("church1 + church2: ", style_loss(church1, church2, model).item())


# model_name = "ViT-B/16"
# prompt = ["dog", "a Big Ben clock towering over the city of London"]

# model, preprocess = clip.load(model_name, device=device)

# text = clip.tokenize(prompt).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     image_ref_features = model.encode_image(image_ref)
#     text_features = model.encode_text(text)
#
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:")
# for i, prob in enumerate(probs[0]):
#     print(f"{prompt[i]}: {prob:.4f}")

