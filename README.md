# JAX + Flax Template

This is a JAX/Flax project template, mirroring the structure of a PyTorch Lightning template.
It includes:
- **Hydra** for configuration.
- **Flax** for model definition.
- **Optax** for optimization.
- **Orbax** for checkpointing.
- **WandB** & **TensorBoard** for logging.
- **JAX Data Loading** (simple numpy-based loader in `core`).

## Structure

- `configs/`: Hydra configurations.
- `data/`: Data loading logic.
- `models/`: Model definitions (Flax Modules).
- `utils/`: Helper functions.
- `core/`: JAX-specific utilities (DataLoader, Checkpointing).
- `train.py`: Main training loop.
- `eval.py`: Evaluation script.

## Usage

1. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate jax-flax
   ```

2. Train:
   ```bash
   python train.py
   ```

3. Evaluation:
   ```bash
   python eval.py resume=logs/TemplateDataModule/TemplateModel/YYYY-MM-DD_HH-MM-SS
   ```