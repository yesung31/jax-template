import logging
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from absl import logging as absl_logging
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm

import data
import models
from core.checkpoint import CheckpointManager
from utils.helpers import get_resume_info, instantiate, register_resolvers

register_resolvers()


def setup_logging(log_dir):
    # Setup file logging
    log_file = log_dir / "train.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging to console
    logger.handlers = []

    # File handler (captures everything)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Redirect absl logs
    absl_logging.set_verbosity(absl_logging.INFO)
    absl_logging.set_stderrthreshold("error")  # Only show errors in terminal

    # Remove all handlers from absl logging and add our file handler
    absl_logger = logging.getLogger("absl")
    absl_logger.handlers = []
    absl_logger.addHandler(fh)
    absl_logger.propagate = False


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    setup_logging(log_dir)

    print(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    # Dynamic loading
    ModelClass = getattr(models, cfg.model.name)
    DataClass = getattr(data, cfg.data.name)

    print(f"Model: {ModelClass.__name__}")
    print(f"DataModule: {DataClass.__name__}")

    # Instantiate
    # Note: Model wrapper is instantiated without lr, as it's stateless regarding optimization.
    model_wrapper = instantiate(ModelClass, cfg.model)
    dm = instantiate(DataClass, cfg.data)

    # Setup Data
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Create Train State
    # We need a sample input to initialize parameters.
    # Assumes the first batch from train_loader works.
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0]

    rng, init_rng = jax.random.split(rng)

    # We pass the learning rate here, decoupling it from the model structure.
    # Assumes cfg.model.lr exists as per the config structure.
    state = model_wrapper.create_train_state(init_rng, sample_input.shape, cfg.model.lr)

    # Model Summary
    print("\nModel Summary:")
    try:
        summary = model_wrapper.net.tabulate(
            rng, sample_input, console_kwargs={"force_terminal": True}
        )
        print(summary)
    except Exception as e:
        print(f"Could not generate model summary: {e}")
    print("\n")

    # Compile functions
    @jax.jit
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(model_wrapper.loss_fn, argnums=0, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    @jax.jit
    def eval_step(state, batch):
        metrics = model_wrapper.eval_step(state, batch)
        return metrics

    # Logging & Checkpointing
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_manager = CheckpointManager(log_dir / "checkpoints")

    # WandB
    if cfg.wandb != "disabled":
        wandb.init(
            project=Path.cwd().name,
            name=f"{cfg.data.name}/{cfg.model.name}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb,
            dir=log_dir,
        )

    tb_writer = SummaryWriter(log_dir)

    # Resume
    latest_step, wandb_id = get_resume_info(log_dir) if cfg.resume else (None, None)
    if latest_step is not None:
        state = ckpt_manager.restore(step=latest_step, items=state)

    # Training Loop
    start_epoch = 0  # Simple handling, logic for step-based resume is more complex
    global_step = 0

    if latest_step is not None:
        global_step = latest_step
        # Approximate epoch start
        start_epoch = global_step // len(train_loader)

    print("Starting training...")
    total_steps = len(train_loader) * cfg.max_epochs
    pbar = tqdm(total=total_steps, initial=global_step, desc="Training", leave=True)
    # Placeholder for pbar metrics to ensure variable scope availability
    pbar_metrics = {}

    for epoch in range(start_epoch, cfg.max_epochs):
        # Train
        for batch in train_loader:
            # Transfer to JAX device if not already (JIT handles it but explicit is good)
            batch = jax.tree_util.tree_map(jnp.array, batch)
            state, loss, aux = train_step(state, batch)

            # Get metrics for progress bar
            pbar_metrics = {k: float(v) for k, v in aux["pbar"].items()}
            pbar.set_postfix(pbar_metrics)

            # Log metrics
            if global_step % 10 == 0:
                log_metrics = {k: float(v) for k, v in aux["log"].items()}
                log_metrics["epoch"] = epoch
                if wandb.run is not None:
                    wandb.log(log_metrics, step=global_step)
                for k, v in log_metrics.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(k, v, global_step)

            pbar.update(1)
            global_step += 1

        # Validate
        val_metrics_list = []
        for batch in val_loader:
            batch = jax.tree_util.tree_map(jnp.array, batch)
            metrics = eval_step(state, batch)
            val_metrics_list.append(metrics)

        # Aggregate metrics dynamically
        if val_metrics_list:
            avg_val_metrics = {}
            for k in val_metrics_list[0].keys():
                # Extract scalar value from JAX array if necessary
                avg_val_metrics[k] = np.mean([float(m[k]) for m in val_metrics_list])
            
            # Update pbar with validation metrics (formatted nicely)
            val_display = {k: f"{v:.4f}" for k, v in avg_val_metrics.items()}
            # Merge with current train metrics
            current_postfix = pbar.postfix if pbar.postfix else {}
            # If pbar.postfix is a string, we might have issues, but set_postfix handles dicts.
            # We recreate the dict to ensure clarity
            combined_metrics = {**pbar_metrics, **val_display}
            pbar.set_postfix(combined_metrics)

            if wandb.run is not None:
                wandb.log({**avg_val_metrics, "epoch": epoch}, step=global_step)
            for k, v in avg_val_metrics.items():
                tb_writer.add_scalar(k, v, global_step)

        # Save Checkpoint
        ckpt_manager.save(global_step, state)

    pbar.close()
    if wandb.run is not None:
        wandb.finish()
    tb_writer.close()


if __name__ == "__main__":
    main()
