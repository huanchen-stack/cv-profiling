import torch
from torchsummary import summary
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision

from timer import Clock
from memorizer import MemRec

device = "cuda:0" if torch.cuda.is_available() else "cpu"
usingcuda = device == "cuda:0"

tt = Clock()
mr = MemRec()

resnet = torchvision.models.resnet50(pretrained=True)
x = torch.rand(1, 3, 224, 224)

# for name, layer in resnet.named_children():
#     print(name, '\n', layer)

def itrResLayer(model, parentLayer, input, recursive=False):
    # make sure inner layers wont mess up residual steps
    x = input.clone()

    for name, layer in model.named_children():

        # flatten for the fc layer
        if name == "fc":
            x = torch.flatten(x, 1)  # the flatten step is not analyzed

        # update layer name (after flatten before fc layers)
        name = f"{parentLayer}_{name}"
        print(name)

        # recursively iterate all sequential layers
        if recursive and type(layer) == nn.Sequential:
            itrResLayer(layer, name, x, recursive=True)

        # find mem consumption
        tmp_x = torch.clone(x)
        with profile(
                activities=
                [
                    ProfilerActivity.CPU
                ] if not usingcuda else
                [
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                ],
                profile_memory=True, record_shapes=True
            ) as prof:
            with record_function("model_inference"):
                layer(tmp_x)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem(name, prof_report, usingcuda)

        # find runtime
        tt.tic(name)
        x = layer(x)
        tt.toc(name)

for i in range(10):
    itrResLayer(resnet, "", x, recursive=True)
    # break

tt.report(sample=False)
mr.report(sample=False)

