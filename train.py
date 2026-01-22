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
        (loss, logits), grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

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
    for epoch in range(start_epoch, cfg.max_epochs):
        # Train
        train_metrics = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for batch in pbar:
            # Transfer to JAX device if not already (JIT handles it but explicit is good)
            batch = jax.tree_util.tree_map(jnp.array, batch)
            state, loss = train_step(state, batch)
            train_metrics.append(loss)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % 10 == 0:
                if wandb.run is not None:
                    wandb.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)
                tb_writer.add_scalar("train_loss", loss.item(), global_step)

            global_step += 1

        avg_train_loss = np.mean(train_metrics)
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}")

        # Validate
        val_metrics = []
        for batch in val_loader:
            batch = jax.tree_util.tree_map(jnp.array, batch)
            metrics = eval_step(state, batch)
            val_metrics.append(metrics)

        # Aggregate metrics
        avg_val_loss = np.mean([m["loss"] for m in val_metrics])
        avg_val_acc = np.mean([m["accuracy"] for m in val_metrics])

        print(f"Epoch {epoch}: Val Loss {avg_val_loss:.4f}, Val Acc {avg_val_acc:.4f}")

        if wandb.run is not None:
            wandb.log(
                {"val_loss": avg_val_loss, "val_acc": avg_val_acc, "epoch": epoch}, step=global_step
            )
        tb_writer.add_scalar("val_loss", avg_val_loss, global_step)
        tb_writer.add_scalar("val_acc", avg_val_acc, global_step)

        # Save Checkpoint
        ckpt_manager.save(global_step, state)

    if wandb.run is not None:
        wandb.finish()
    tb_writer.close()


if __name__ == "__main__":
    main()
