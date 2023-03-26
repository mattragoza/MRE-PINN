# MRE-PINN

Physics-informed networks for elasticity reconstruction in magnetic resonance elastography.

![MRE-PINN examples](MICCAI-2023/images/patient_image_grid.png)

## Installation

Run the following to setup the conda environment and install it as a Jupyter notebook kernel:

```bash
mamba env create --file=environment.yml
mamba activate MRE-PINN
python -m ipykernel install --user --name=MRE-PINN
```

## Usage

To download the BIOQIC data sets:

```bash
sh download_data.sh
```

To train an MRE-PINN model:

```bash
python train.py
```
