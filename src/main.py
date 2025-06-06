import argparse

from scripts import visualize_data
import huggingface_hub
import configparser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run different scripts based on the provided arguments"
    )
    subparsers = parser.add_subparsers(dest="script_name", required=True, help="Script to run")

    # Visualize data parser
    parser_visualize_data = subparsers.add_parser("visualize_data", help="Visualize data")
    parser_visualize_data.add_argument(
        '--num_samples', type=int, required=True, help='Number of samples to visualize'
    )

    # Optional: either reference path or prompt (at least one required)
    parser_visualize_data.add_argument(
        '--reference_path', type=str, help='Path to reference image'
    )
    parser_visualize_data.add_argument(
        '--prompt', type=str, help='Text prompt for image generation'
    )

    args = parser.parse_args()

    if args.script_name == "visualize_data":
        # Use whichever is set (prioritize prompt if both are present)
        visualize_data(num_samples=args.num_samples, reference_path=args.reference_path, prompt=args.prompt)

#ss_loss =

#original_image = ss_loss.original_image
##low_quality_image = ss_loss.low_quality_image
#reference_image = ss_loss.reference

#visualize_image(original_image, f"ss_original_image.png")
#visualize_image(low_quality_image, f"ss_low_quality_image.png")
#visualize_image(reference_image, f"ss_reference_image.png")

#mpgd = MPGDLatent(ss_loss, num_inference_steps=50)
# image = mpgd()

# visualize_image(image, f"ss_result.png")