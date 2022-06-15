from data import build_dataset
from model import build_model
from configs import build_config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    seed_everything(42, workers=True)
    cfg = build_config()
    dataset = build_dataset(cfg)
    model = build_model(cfg)
    trainer = Trainer(
        gpus=cfg.NUM_GPUS, 
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            ModelCheckpoint()
        ],
        benchmark=True, 
        deterministic=True,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        default_root_dir=cfg.OUTPUT_DIR,
        check_val_every_n_epoch=cfg.CHECK_VAL_EVERY_N_EPOCH,
    )
    trainer.fit(model, datamodule=dataset, 
        ckpt_path=cfg.CKPT if hasattr(cfg, "CKPT") else None)