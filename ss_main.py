import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms
from losses.ss_loss import SSGuidanceLoss
from model import MPGDLatent as MPGDStableDiffusionGenerator
from scripts.visualize_data import visualize_image
from datetime import datetime

# Import your custom classes (adjust the import paths as needed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 4  # Set the number of images to generate per trial


reference_image_path = "images.jpeg"

reference = Image.open(reference_image_path).convert("RGB")
reference = reference.resize((512, 512))
# Preprocess: convert to tensor normalized in [-1, 1]
image_tensor = (
    transforms.ToTensor()(reference).unsqueeze(0).to(device)
)  # shape [1,3,H,W]
image_tensor = 2.0 * image_tensor - 1.0  # scale from [0,1] to [-1,1]

n = 100  # Set the number of random search trials
search_hyperparameters = True

if search_hyperparameters:
    for trial in range(n):
        # Sample three values from Dirichlet to ensure they sum to 1
        lambdas = np.random.dirichlet(np.ones(3), size=1)[0]
        lpips_lambda, msss_lambda, spatial_lambda = lambdas

        # Sample lr in log space between 1e-3 and 1e3
        # log_lr = np.random.uniform(4, 6)
        # lr = 10**log_lr

        lr = 5e05

        print(
            f"Trial {trial+1}: lpips_lambda={lpips_lambda:.3f}, msss_lambda={msss_lambda:.3f}, spatial_lambda={spatial_lambda:.3f}, lr={lr:.5f}"
        )

        # Create loss with sampled hyperparameters
        loss = SSGuidanceLoss(
            image_tensor,
            device=device,
            lpips_lambda=lpips_lambda,
            msss_lambda=msss_lambda,
            spatial_lambda=spatial_lambda,
            # lr=lr,
        )

        generator = MPGDStableDiffusionGenerator(lr=lr, loss_fn=loss)

        # Generate images (could evaluate quality here)
        images = generator()
        # Save or evaluate images as needed
        trial_folder = Path(f"data/ss_search_trial")
        trial_folder.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"search_trial{trial+1}_lpips{lpips_lambda:.3f}_msss{msss_lambda:.3f}_spatial{spatial_lambda:.3f}_lr{lr:.5f}_img{i}.png"
            save_name = f"{timestamp}_" + save_name

            visualize_image(image.unsqueeze(0), save_name, trial_folder)
else:
    generator = MPGDStableDiffusionGenerator(loss=loss)

    # Generate images
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
