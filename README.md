# Manifold Preserving Guided Diffusion

The aim of this project is to reproduce and extend the Manifold Preserving Guided Diffusion \[1]. 


The repository contains Stable Diffusion v1-4 checkpoint. 

## Installation
1. Clone the repository

```sh
git clone https://github.com/alexcojocaru2002/manifold-preserving-guided-diffusion.git
```

2. Install the required libraries (make sure you have Python and pip already installed)
**Optional**: If you have CUDA, first check its version by running 
```sh
nvcc --version
```
and then go to https://pytorch.org/get-started/locally/ and run the command suggested there to ensure the torch versions maches the CUDA version.

Then proceed with installing the rest of the dependencies:

```sh
pip install -r requirements.txt
```


## Data Visualization 

To run an example for stable diffusion you can use the following command 

```sh
python main.py visualize_data --num_samples 1 prompt "a cat playing fotball"
```

1. He, Y., Murata, N., Lai, C.-H., Takida, Y., Uesaka, T., Kim, D., Liao, W.-H., Mitsufuji, Y., Kolter, J. Z., Salakhutdinov, R., & Ermon, S. (2023). Manifold Preserving Guided Diffusion. arXiv preprint arXiv:2311.16424. https://arxiv.org/abs/2311.16424
