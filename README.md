## Requirements

> - Python >= 3.9.16
> - PyTorch >= 1.12.1
> - Platforms: Ubuntu 22.04, CUDA 11.6
> - Run `conda env create -f requirements.yaml` to create a new conda environment.
> - Run `pip install -e torchlight`.

## Data Preparation

Run `python data_processing.py`.

## Training

```bash
# Train SkateFormer
python main.py --model_name skateformer --config ./config/train/assistive_furniture/SkateFormer_j.yaml

# Train STGCN
python main.py --model_name stgcn

```
