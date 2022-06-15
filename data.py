from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from pytorch_lightning import LightningDataModule
from timm.data import create_transform

class Cifar100DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def train_dataloader(self): 
        cfg = self.cfg
        transform = create_transform(224, is_training=True)
        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        return DataLoader(trainset, batch_size=cfg.SOLVER.BATCH_SIZE//cfg.NUM_GPUS, shuffle=True, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        cfg = self.cfg
        transform = create_transform(224, is_training=False)
        testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
        return DataLoader(testset, batch_size=cfg.SOLVER.BATCH_SIZE//cfg.NUM_GPUS, shuffle=False, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

datamodules = {
    "cifar100": Cifar100DataModule
}

def build_dataset(cfg):
    return datamodules[cfg.DATASET](cfg)
