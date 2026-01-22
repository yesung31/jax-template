import hydra
import jax
import jax.numpy as jnp
import numpy as np
import logging
from absl import logging as absl_logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import data
import models
from utils.helpers import instantiate, register_resolvers
from core.checkpoint import CheckpointManager

register_resolvers()

def setup_logging(log_dir):
    log_file = log_dir / "eval.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    absl_logging.set_verbosity(absl_logging.INFO)
    absl_logging.set_stderrthreshold('error')
    absl_logger = logging.getLogger('absl')
    absl_logger.handlers = []
    absl_logger.addHandler(fh)
    absl_logger.propagate = False

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    setup_logging(log_dir)
    
    print(f"Evaluating with config:\n{OmegaConf.to_yaml(cfg)}")
    
    rng = jax.random.PRNGKey(cfg.seed)

    # Dynamic loading
    ModelClass = getattr(models, cfg.model.name)
    DataClass = getattr(data, cfg.data.name)

    # Instantiate
    model_wrapper = instantiate(ModelClass, cfg.model)
    dm = instantiate(DataClass, cfg.data)
    dm.setup()
    test_loader = dm.test_dataloader()

    # Create Train State (dummy for structure)
    sample_batch = next(iter(test_loader))
    sample_input = sample_batch[0]
    rng, init_rng = jax.random.split(rng)
    # Pass LR (even if not used for eval, needed for shape/state creation)
    state = model_wrapper.create_train_state(init_rng, sample_input.shape, cfg.model.lr)

    # Load Checkpoint
    ckpt_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    # If explicit path provided in resume, use it.
    if cfg.resume:
       ckpt_path = Path(cfg.resume)
    
    if (ckpt_path / "checkpoints").exists():
         ckpt_manager = CheckpointManager(ckpt_path / "checkpoints")
         state = ckpt_manager.restore(items=state)
         print(f"Restored from {ckpt_path}")
    else:
        print("No checkpoint found. Evaluating with random weights.")

    @jax.jit
    def eval_step(state, batch):
        metrics = model_wrapper.eval_step(state, batch)
        return metrics

    # Test Loop
    test_metrics = []
    print("Starting evaluation...")
    for batch in tqdm(test_loader, desc="Testing"):
        batch = jax.tree_util.tree_map(jnp.array, batch)
        metrics = eval_step(state, batch)
        test_metrics.append(metrics)
    
    avg_test_loss = np.mean([m['loss'] for m in test_metrics])
    avg_test_acc = np.mean([m['accuracy'] for m in test_metrics])
    
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {avg_test_acc:.4f}")

if __name__ == "__main__":
    main()