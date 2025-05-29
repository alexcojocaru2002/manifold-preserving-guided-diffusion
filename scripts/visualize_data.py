from PIL import Image
from pathlib import Path


def visualize_image(image, save_name, save_folder="visualization"):
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    full_save_path = save_path / save_name

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(full_save_path)
