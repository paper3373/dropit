import torch, timm
from torchvision.transforms import Normalize
from pytorch_lightning import seed_everything
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import time
from dropit import to_dropit, Contiguous, DropITer

if __name__ == "__main__":  
    seed_everything(42, workers=True)
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100).cuda()
    model.pre_logits = Contiguous()
    x = torch.rand(256,3,224,224).cuda()
    transform = Normalize(
        mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
        std=torch.tensor(IMAGENET_DEFAULT_STD)
    )
    x = transform(x)

    # strategy: random_c, mink_c, random, mink, parallel_mink, nsigma
    # c denotes channel 
    dropiter = DropITer("nsigma", gamma=0.8)
    to_dropit(model, dropiter)
    model = model.cuda()
    
    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        y = model(x)
        y.sum().backward()
    torch.cuda.synchronize()
    end = time.time()

    print((end - start) / 100 * 1000, "ms")
    print(torch.cuda.max_memory_allocated()/1024**3, "GB")
    