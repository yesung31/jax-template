import logging
import time
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

from core.callback import Callback
from core.checkpoint import CheckpointManager
from utils.helpers import get_resume_info


class Trainer:
    def __init__(self, cfg: DictConfig, model, datamodule, callbacks: list[Callback] | None = None):
        self.cfg = cfg
        self.model = model
        self.dm = datamodule
        self.callbacks = callbacks or []

        # Resolve output directory from Hydra
        try:
            self.log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        except Exception:
            # Fallback for manual runs without Hydra context (if ever needed)
            self.log_dir = Path.cwd() / "logs" / "manual"
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # State
        self.state = None
        self.global_step = 0
        self.rng = jax.random.PRNGKey(cfg.seed)

        # Checkpointing
        self.ckpt_manager = CheckpointManager(self.log_dir / "checkpoints")

        # Logging
        self.wandb_run = None
        self.tb_writer = None

    def _call_callbacks(self, method_name, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(self, self.model, *args, **kwargs)

    def _setup_logging(self):
        log_file = self.log_dir / "trainer.log"

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers = []

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Redirect absl logs
        absl_logging.set_verbosity(absl_logging.INFO)
        absl_logging.set_stderrthreshold("error")
        absl_logger = logging.getLogger("absl")
        absl_logger.handlers = []
        absl_logger.addHandler(fh)
        absl_logger.propagate = False

    def setup(self):
        """Initializes state, data, and loggers."""
        # Loggers
        if self.cfg.wandb != "disabled":
            self.wandb_run = wandb.init(
                project=Path.cwd().name,
                name=f"{self.cfg.data.name}/{self.cfg.model.name}",
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode=self.cfg.wandb,
                dir=self.log_dir,
            )
        self.tb_writer = SummaryWriter(str(self.log_dir))

        # Data
        self.dm.setup()

        # Train State Initialization
        # We need a sample batch to initialize parameters
        # Try to get it from train_dataloader, else val/test
        try:
            sample_loader = self.dm.train_dataloader()
        except Exception:
            sample_loader = self.dm.test_dataloader()

        sample_batch = next(iter(sample_loader))
        # Handle tuple batch (x, y) or dict batch {'input': x, ...}
        if isinstance(sample_batch, dict):
            sample_input = sample_batch["input"]
        elif isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0]
        else:
            sample_input = sample_batch

        self.rng, init_rng = jax.random.split(self.rng)
        self.state = self.model.create_train_state(init_rng, sample_input.shape, self.cfg.model.lr)

        # Resume logic
        latest_step = None
        if self.cfg.resume:
            latest_step, _ = get_resume_info(self.log_dir)

        if latest_step is not None:
            self.state = self.ckpt_manager.restore(step=latest_step, items=self.state)
            self.global_step = latest_step
            print(f"Resumed from step {latest_step}")

        # Print Summary
        print("\nModel Summary:")
        try:
            summary = self.model.net.tabulate(
                self.rng, sample_input, console_kwargs={"force_terminal": True}
            )
            print(summary)
        except Exception as e:
            print(f"Could not generate model summary: {e}")
        print("\n")

    def fit(self):
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()

        # JIT Functions
        @jax.jit
        def train_step(state, batch):
            def loss_fn(params):
                output = self.model.training_step(params, batch)
                return output.loss, output

            (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, output

        @jax.jit
        def eval_step(state, batch):
            return self.model.validation_step(state, batch)

        print("Starting training...")
        self._call_callbacks("on_train_start")

        steps_per_epoch = len(train_loader)

        # Determine start epoch
        start_epoch = self.global_step // steps_per_epoch

        pbar = tqdm(total=steps_per_epoch, desc="Training", leave=True)
        pbar_metrics = {}

        for epoch in range(start_epoch, self.cfg.max_epochs):
            self._call_callbacks("on_train_epoch_start")

            pbar.reset(total=steps_per_epoch)
            pbar.set_description(f"Epoch {epoch}")

            # --- TRAIN LOOP ---
            batch_idx = 0
            for batch in train_loader:
                self._call_callbacks("on_train_batch_start", batch=batch, batch_idx=batch_idx)

                batch = jax.tree_util.tree_map(jnp.array, batch)
                self.state, output = train_step(self.state, batch)

                self._call_callbacks(
                    "on_train_batch_end", outputs=output, batch=batch, batch_idx=batch_idx
                )

                # Update Pbar
                pbar_metrics = {k: float(v) for k, v in output.pbar.items()}
                pbar.set_postfix(pbar_metrics)

                # Logging
                if self.global_step % 10 == 0:
                    log_metrics = {k: float(v) for k, v in output.metrics.items()}
                    log_metrics["epoch"] = epoch

                    if self.wandb_run:
                        wandb.log(log_metrics, step=self.global_step)
                    for k, v in log_metrics.items():
                        if isinstance(v, (int, float)):
                            self.tb_writer.add_scalar(k, v, self.global_step)

                pbar.update(1)
                self.global_step += 1
                batch_idx += 1

            # --- VALIDATION LOOP ---
            self._call_callbacks("on_validation_start")

            val_metrics_list = []
            val_batch_idx = 0
            for batch in val_loader:
                self._call_callbacks(
                    "on_validation_batch_start", batch=batch, batch_idx=val_batch_idx
                )

                batch = jax.tree_util.tree_map(jnp.array, batch)
                output = eval_step(self.state, batch)
                val_metrics_list.append(output.metrics)

                self._call_callbacks(
                    "on_validation_batch_end", outputs=output, batch=batch, batch_idx=val_batch_idx
                )
                val_batch_idx += 1

            if val_metrics_list:
                avg_val_metrics = {}
                for k in val_metrics_list[0].keys():
                    avg_val_metrics[k] = np.mean([float(m[k]) for m in val_metrics_list])

                # Update Pbar with Val Metrics
                val_display = {k: f"{v:.4f}" for k, v in avg_val_metrics.items()}
                pbar.set_postfix({**pbar_metrics, **val_display})

                # Log Val Metrics
                if self.wandb_run:
                    wandb.log({**avg_val_metrics, "epoch": epoch}, step=self.global_step)
                for k, v in avg_val_metrics.items():
                    self.tb_writer.add_scalar(k, v, self.global_step)

            self._call_callbacks("on_validation_end")

            # Checkpoint
            self.ckpt_manager.save(self.global_step, self.state)

            self._call_callbacks("on_train_epoch_end")

        pbar.close()
        self._call_callbacks("on_train_end")
        self.teardown()

    def test(self, ckpt_path: str | None = None):
        """str | None
        Runs evaluation on the test set.
        Args:
            ckpt_path: Optional path to a checkpoint directory to restore from.
                       If None, uses the current state (or initialized random state).
        """
        if ckpt_path:
            # Override internal state with checkpoint
            path = Path(ckpt_path) / "checkpoints"
            if path.exists():
                mgr = CheckpointManager(path)
                self.state = mgr.restore(items=self.state)
                print(f"Restored model from {path}")
            else:
                print(f"Checkpoint not found at {path}, using current/random state.")

        test_loader = self.dm.test_dataloader()

        @jax.jit
        def eval_step(state, batch):
            return self.model.test_step(state, batch)

        print("Starting testing...")
        self._call_callbacks("on_test_start")

        test_metrics_list = []
        batch_idx = 0
        for batch in tqdm(test_loader, desc="Testing"):
            self._call_callbacks("on_test_batch_start", batch=batch, batch_idx=batch_idx)

            batch = jax.tree_util.tree_map(jnp.array, batch)
            output = eval_step(self.state, batch)
            test_metrics_list.append(output.metrics)

            self._call_callbacks(
                "on_test_batch_end", outputs=output, batch=batch, batch_idx=batch_idx
            )
            batch_idx += 1

        if test_metrics_list:
            avg_metrics = {}
            for k in test_metrics_list[0].keys():
                avg_metrics[k] = np.mean([float(m[k]) for m in test_metrics_list])

            print("\nTest Results:")
            for k, v in avg_metrics.items():
                print(f"{k}: {v:.4f}")
            print("\n")

        self._call_callbacks("on_test_end")
        self.teardown()

    def teardown(self):
        if self.wandb_run:
            wandb.finish()
        if self.tb_writer:
            self.tb_writer.close()
