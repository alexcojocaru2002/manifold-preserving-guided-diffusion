import argparse

from scripts import visualize_data
import configparser

if __name__ == '__main__':
    import huggingface_hub
    print(dir(huggingface_hub))

    parser = argparse.ArgumentParser(
        description="Run different scripts based on the provided arguments"
    )
    subparsers = parser.add_subparsers(dest="script_name", required=True,
                                       help="Script to run")

    parser_visualize_data = subparsers.add_parser("visualize_data", help="Visualize data")
    parser_visualize_data.add_argument(
        '--num_samples', type=int, required=True, help='Number of samples to visualize'
    )

    args = parser.parse_args()
    if args.script_name == "visualize_data":
        visualize_data(num_samples=args.num_samples)
