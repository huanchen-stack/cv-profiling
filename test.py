import torch
from toyResnet import ResBlock, ResBottleneckBlock, ResNet

from timer import Clock
from memorizer import MemRec

tt = Clock()
mr = MemRec()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


# create a model
resnet50 = ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=1000)
resnet50.to(device)
print("Self-Implemented Resnet50 is Ready")

for i in range(5):
    input = torch.rand(1, 3, 224, 224)
    resnet50.forward(input, tt, mr)

tt.report(sample=False)
mr.report(sample=False)





