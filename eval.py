import hydra
from omegaconf import DictConfig

import data
import models
from core.trainer import Trainer
from utils.helpers import instantiate, register_resolvers

register_resolvers()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Dynamic loading
    ModelClass = getattr(models, cfg.model.name)
    DataClass = getattr(data, cfg.data.name)

    # Instantiate
    model = instantiate(ModelClass, cfg.model)
    dm = instantiate(DataClass, cfg.data)

    # Trainer
    trainer = Trainer(cfg, model, dm)
    trainer.setup()
    trainer.test()


if __name__ == "__main__":
    main()
