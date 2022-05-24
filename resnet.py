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

def _tensor_size(tensor):
    return f"{tensor.element_size() * tensor.nelement() / 1000000} Mb"

class ProfResnet(object):
    def __init__(self):
        super().__init__()
        self.residual = None
    
    def itrResLayer(self, model, parentLayer, input, recursive=False):
        # make sure inner layers wont mess up residual steps
        x = input.clone().to(device)

        for name, layer in model.named_children():

            # flatten for the fc layer
            if name == "fc":
                x = torch.flatten(x, 1)  # the flatten step is not analyzed
            
            # update residual
            if name[:-1] == "layer":
                # print(x.shape)
                self.residual = x.clone()

            # update layer name (after flatten before fc layers)
            name = f"{parentLayer}_{name}"
            # print(name)

            if recursive:
                if name.split("_")[-1] == 'downsample':
                    x = self.residual.clone()
                else:
                    self.itrResLayer(layer, name, x, recursive=True)

            # recursively iterate all sequential layers
            # if recursive and type(layer) == nn.Sequential:
            #     self.itrResLayer(layer, name, x, recursive=True)

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

            print(name, "_in ",  x.shape, ' ', _tensor_size(x), sep='')

            # find runtime
            layer(x)
            tt.tic(name)
            x = layer(x)
            tt.toc(name)
            # x = tmp

            print(name, "_out ", x.shape, ' ', _tensor_size(x), sep='')


if __name__ == "__main__":

    resnet = torchvision.models.resnet50(pretrained=True).to(device)
    x = torch.rand(1, 3, 800, 800).to(device)
    # residual = x.clone()

    for name, layer in resnet.named_children():
        print(name, '\n', layer)

    for i in range(2):
        profResnet = ProfResnet()
        profResnet.itrResLayer(resnet, "", x, recursive=False)
        # break

    tt.tic("resnet50")
    resnet(x)
    tt.toc("resnet50")
    tt.tic("resnet50")
    resnet(x)
    tt.toc("resnet50")

    tt.report(sample=False)
    mr.report(sample=False)

