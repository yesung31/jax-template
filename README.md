<!-- TEMPLATE INSTRUCTIONS: DELETE THIS SECTION BEFORE RELEASE -->
# How to use this template (JAX/Flax Version)

1.  **Rename**: Rename this folder to your project name.
2.  **Environment**: 
    - Create a virtual environment (e.g., `python -m venv .venv` or `conda create -n myenv python=3.12`).
    - Activate the environment.
    - Install dependencies: `uv pip install -r pyproject.toml` (or `pip install .`).
3.  **Implement**:
    - Add your model in `models/your_model.py`. It must inherit `core.model.Model`.
    - Add your network (Flax Module) in `models/networks/your_network.py`.
    - Add your data module in `data/{task}/{dataset}.py`. It must inherit `core.data.DataModule`.
4.  **Run**:
    - `python train.py model.name=YourModel data.name=TemplateDataModule`
    - Or update `configs/config.yaml` defaults.

---

# Project Name (JAX/Flax)

[Short description of the project]

## Installation

Prerequisites: Python 3.12+

1.  **Create a virtual environment**:
    
    Using standard Python venv:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

    Or using Conda:
    ```bash
    conda create -n jax python=3.12
    conda activate jax
    ```

2.  **Install dependencies**:
    
    Using [uv](https://github.com/astral-sh/uv) (recommended):
    ```bash
    uv pip install -r pyproject.toml
    ```

    Or using standard pip:
    ```bash
    pip install .
    ```

## Usage

To train the model:

```bash
python train.py
```

### Configuration

You can override parameters from the command line:

```bash
python train.py model.name=MyModel data.name=CIFAR10
```

`data.name` is used for the log directory name and class lookup in the `data` package.

### Multirun

You can run hyperparameter sweeps using the `-m` or `--multirun` flag:

```bash
python train.py -m max_epochs=5,10 seed=42,43
```

This creates a folder structure organized by the sweep timestamp, then data/model, and finally the job number:

```
logs/
└── multirun/
    └── 2026-01-22_10-00-00/
        ├── TemplateDataModule/TemplateModel/0/
        ├── TemplateDataModule/TemplateModel/1/
        ├── TemplateDataModule/TemplateModel/2/
        └── TemplateDataModule/TemplateModel/3/
```

### Resuming Training

You can resume training from a previously interrupted run using the `resume` parameter. This will automatically:
1.  Load the latest checkpoint from the `checkpoints` folder using Orbax.
2.  Reconnect to the previous WandB run ID.
3.  Continue logging in the same TensorBoard directory.

#### Single Run
Point `resume` to the specific run directory:
```bash
python train.py resume=logs/TemplateDataModule/TemplateModel/2026-01-22_10-00-00
```

## Logging

This project uses both **Weights & Biases (WandB)** and **TensorBoard** for logging.

### Weights & Biases

WandB is configured as follows:
- **Project**: The name of the current directory.
- **Run Name**: `{data.name}/{model.name}`.
- **Mode**: Controlled by `wandb` in `config.yaml` (online, offline, disabled).

### TensorBoard

TensorBoard logs are saved locally in the `logs/` directory.

To view logs:
```bash
tensorboard --logdir logs/
```

## Evaluation

To evaluate a trained model, provide the path to the run directory:

```bash
python eval.py resume=logs/TemplateDataModule/TemplateModel/2026-01-22_10-00-00
```

This script automatically restores the latest state from the `checkpoints` folder and uses the configuration found in the `.hydra` directory of that run.

## Project Structure

- `core/`: Universal components (Trainer, Model/Data base classes, Dataloader, Callbacks).
- `configs/`: Hydra configurations.
- `data/`: Data modules and specific dataset implementations.
- `models/`: Model wrappers (inheriting `core.model.Model`).
- `models/networks/`: Neural network architectures (Flax `nn.Module`).
- `logs/`: TensorBoard logs, WandB files, and Orbax checkpoints.
- `utils/`: Helper functions and Hydra resolvers.
