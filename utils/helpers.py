from pathlib import Path

from omegaconf import OmegaConf


def register_resolvers():
    if not OmegaConf.has_resolver("resume_or_new"):
        OmegaConf.register_new_resolver("resume_or_new", lambda r, n: r if r else n)


def get_resume_info(log_dir):
    log_dir = Path(log_dir).resolve()
    ckpt_dir = log_dir / "checkpoints"

    latest_step = None
    if ckpt_dir.exists():
        # Orbax saves steps as directories, usually integers.
        # Find the largest integer directory.
        steps = []
        for d in ckpt_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                steps.append(int(d.name))
        if steps:
            latest_step = max(steps)
            print(f"Resuming from step: {latest_step}")

    wandb_id = None
    w_dir = log_dir / "wandb"
    if w_dir.exists():
        latest = w_dir / "latest-run"
        run = latest.resolve() if latest.exists() else None
        if not run:
            runs = sorted(
                [d for d in w_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
            )
            run = runs[-1] if runs else None
        if run:
            wandb_id = run.name.split("-")[-1]
            print(f"WandB ID: {wandb_id}")

    return latest_step, wandb_id


def instantiate(Class, cfg):
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    # Pass relevant sub-configs as kwargs, excluding 'name'.
    kwargs.pop("name", None)
    return Class(**kwargs)
