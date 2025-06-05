import torch
from PIL import Image
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

from losses.ss_loss import SSGuidanceLoss
from losses.text_guidance_loss import CLIPTextGuidanceLoss
from pipelines.pipeline import MPGDStableDiffusionGenerator
from losses.loss_mse_image import MSEGuidanceLoss
from PIL import Image
from torchvision import transforms

def visualize_image(image, save_name, save_folder="visualization"):
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    full_save_path = save_path / save_name
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(full_save_path)
    
# def visualize_data(generator, prompt, num_samples):

def setup_image_guidance_generator(reference_path, device, num_samples):
    reference_path = reference_path
    reference = Image.open(reference_path).convert("RGB")
    reference = reference.resize((512, 512))
    # Preprocess: convert to tensor normalized in [-1, 1]
    image_tensor = transforms.ToTensor()(reference).unsqueeze(0).to(device)  # shape [1,3,H,W]
    image_tensor = 2.0 * image_tensor - 1.0  # scale from [0,1] to [-1,1]

    generator = MPGDStableDiffusionGenerator(
        loss=SSGuidanceLoss(image_tensor, device=device)
    )

    # Generate images
    print("Showing image")
    images = generator.generate(
        batch_size=num_samples,
        height=512,
        width=512,
        num_inference_steps=50,
        seed=42,  # TO DO: Make this random later
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")

def setup_text_guidance_generator(prompt, device, num_samples):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    generator = MPGDStableDiffusionGenerator(
        loss=CLIPTextGuidanceLoss(prompt, clip_model, clip_processor, device=device)
    )

    # Generate images
    print("Showing image")
    images = generator.generate(
        batch_size=num_samples,
        height=512,
        width=512,
        num_inference_steps=15,
        seed=42,  # TO DO: Make this random later
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")

# Extra_param can be a path for a file if we use image or audio and str for text like stable diff
def run(reference_path, prompt, num_samples=1):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'Running on {torch.cuda.get_device_name(0)}')

    if reference_path is None:
        setup_text_guidance_generator(prompt, device, num_samples)
    else:
        setup_image_guidance_generator(reference_path, device, num_samples)

