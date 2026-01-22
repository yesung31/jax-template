import orbax.checkpoint
from pathlib import Path
import os

class CheckpointManager:
    def __init__(self, directory, max_to_keep=1):
        self.directory = Path(directory).resolve()
        self.options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.manager = orbax.checkpoint.CheckpointManager(
            self.directory, self.checkpointer, self.options
        )

    def save(self, step, items):
        self.manager.save(step, items)

    def restore(self, step=None, items=None):
        if step is None:
            step = self.manager.latest_step()
        if step is not None:
            return self.manager.restore(step, items=items)
        return None

    def latest_step(self):
        return self.manager.latest_step()
