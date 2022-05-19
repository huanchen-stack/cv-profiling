import torch
from torchsummary import summary
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from timer import Clock
from memorizer import MemRec


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (
                    i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)

    def forward(self, input, tt, mr):
        usingcuda = False

        # ----------------------------------------------------------------
        tmp_input = torch.clone(input)
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
                self.layer0(tmp_input)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem("layer0", prof_report, usingcuda)

        tt.tic("layer0")
        input = self.layer0(input)
        tt.toc("layer0")

        # ----------------------------------------------------------------
        tmp_input = torch.clone(input)
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
                self.layer1(tmp_input)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem("layer1", prof_report, usingcuda)

        tt.tic("layer1")
        input = self.layer1(input)
        tt.toc("layer1")

        # ----------------------------------------------------------------
        tmp_input = torch.clone(input)
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
                self.layer2(tmp_input)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem("layer2", prof_report, usingcuda)

        tt.tic("layer2")
        input = self.layer2(input)
        tt.toc("layer2")

        # ----------------------------------------------------------------
        tmp_input = torch.clone(input)
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
                self.layer3(tmp_input)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem("layer3", prof_report, usingcuda)

        tt.tic("layer3")
        input = self.layer3(input)
        tt.toc("layer3")

        # ----------------------------------------------------------------
        tmp_input = torch.clone(input)
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
                self.layer4(tmp_input)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem("layer4", prof_report, usingcuda)

        tt.tic("layer4")
        input = self.layer4(input)
        tt.toc("layer4")

        # ----------------------------------------------------------------
        tmp_input = torch.clone(input)
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
                self.gap(tmp_input)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem("gap", prof_report, usingcuda)        

        tt.tic("gap")
        input = self.gap(input)
        tt.toc("gap")

        # ----------------------------------------------------------------
        tmp_input = torch.clone(input)
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
                tmp_input = torch.flatten(tmp_input, start_dim=1)
                self.fc(tmp_input)
        prof_report = str(prof.key_averages().table()).split("\n")
        mr.get_mem("fc", prof_report, usingcuda)

        tt.tic("fc")
        # torch.flatten()
        # https://stackoverflow.com/questions/60115633/pytorch-flatten-doesnt-maintain-batch-size
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)
        tt.toc("fc")

        return input


# resnet50 = ResNet(3, ResBottleneckBlock, [
#                   3, 4, 6, 3], useBottleneck=True, outputs=1000)
# resnet50.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet50, (3, 224, 224))