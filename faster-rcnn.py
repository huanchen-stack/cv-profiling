import torch
# from torchsummary import summary
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision

from timer import Clock
from memorizer import MemRec

device = "cuda:0" if torch.cuda.is_available() else "cpu"
usingcuda = device == "cuda:0"

tt = Clock()
mr = MemRec()

from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, concat_box_prediction_layers


def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class Profiler():
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        self.images = torch.rand(1, 3, 224, 224).to(device)
        self.targets = None
        self.args = {}
        self.mem = False

    def _profile_helper(self, func, name):
        if not self.mem:
            tt.tic(name)
            func()
            tt.toc(name)
        else:
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
                    func()
            prof_report = str(prof.key_averages().table()).split("\n")
            mr.get_mem(name, prof_report, usingcuda)

    def get_original_images_sizes(self):
        # get original_image_sizes for postprocessing
        original_image_sizes = []
        for img in self.images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        self.args["original_image_sizes"] = original_image_sizes

    def transform(self):
        self.images, self.targets = self.model.transform(self.images, self.targets)

    def backbonefpn(self):
        # backbonefpn_body |FAKE| setup
        backbone = torchvision.models.resnet50(
            pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d
        )
        returned_layers = [1, 2, 3, 4]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        backbonefpn_body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # forward
        self.args["x"] = backbonefpn_body(self.images.tensors)

        # backbone_fpn setup 
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        out_channels = 256
        extra_blocks = LastLevelMaxPool()
        backbonefpn_fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        # forward
        self.args["features"] = backbonefpn_fpn(self.args["x"])
        # feature_ list setup
        self.args["features_"] = list(self.args["features"].values())
        
        # self.args["features"] = model.backbone(self.images.tensors)
        # self.args["features_"] = list(model.backbone(self.images.tensors).values())

    def backbonefpn_details(self):
        # backbonefpn_body |FAKE| setup
        backbone = torchvision.models.resnet50(
            pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d
        )
        returned_layers = [1, 2, 3, 4]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        backbonefpn_body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # forward
        def _backbonefpn_body():
            self.args["x"] = backbonefpn_body(self.images.tensors)
        self._profile_helper(_backbonefpn_body, "backbonefpn_body")

        # backbone_fpn setup 
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        out_channels = 256
        extra_blocks = LastLevelMaxPool()
        backbonefpn_fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        # forward
        def _backbonefpn_fpn():
            self.args["features"] = backbonefpn_fpn(self.args["x"])
        self._profile_helper(_backbonefpn_fpn, "backbonefpn_fpn")
        # feature_ list setup
        self.args["features_"] = list(self.args["features"].values())
        
        # self.args["features"] = model.backbone(self.images.tensors)
        # self.args["features_"] = list(model.backbone(self.images.tensors).values())

    def backbonefpn_fpn_details():
        pass

    def rpn_head(self):
        # RPNHead setup
        out_channels = self.model.backbone.out_channels
        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        # forward
        logits = []
        bbox_reg = []
        for feature in self.args["features_"]:
            t = rpn_head.conv(feature)
            logits.append(rpn_head.cls_logits(t))
            bbox_reg.append(rpn_head.bbox_pred(t))
        self.args["objectness"] = logits
        self.args["pred_bbox_deltas"] = bbox_reg

    def rpn_head_details(self):
        # RPNHead initialization
        out_channels = self.model.backbone.out_channels
        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        # forward
        logits = []
        bbox_reg = []
        for feature in self.args["features_"]:
            t = rpn_head.conv(feature)
            convs = []
            for name, layer in rpn_head.named_children():
                convs.append(layer)
            # details
            x = convs[0](feature)
            y = convs[1](x)
            z = convs[2](x)
            def _conv():
                convs[0](feature)
            def _logits():
                convs[1](x)
            def _regression():
                convs[2](x)
            self._profile_helper(_conv, "rpn_head_details_conv_per_feature")
            self._profile_helper(_logits, "rpn_head_details_logits_per_feature")
            self._profile_helper(_regression, "rpn_head_details_regression_per_feature")
            # enddetails
            logits.append(rpn_head.cls_logits(t))
            bbox_reg.append(rpn_head.bbox_pred(t))
        self.args["objectness"] = logits
        self.args["pred_bbox_deltas"] = bbox_reg

    def rpn_anchor_generator(self):
        anchors = self.model.rpn.anchor_generator(self.images, self.args["features_"])
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in self.args["objectness"]]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        self.args["objectness"], self.args["pred_bbox_deltas"] = concat_box_prediction_layers(self.args["objectness"], self.args["pred_bbox_deltas"])
        proposals = model.rpn.box_coder.decode(self.args["pred_bbox_deltas"].detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = model.rpn.filter_proposals(proposals, self.args["objectness"], self.images.image_sizes, num_anchors_per_level)
        self.args["proposals"], self.args["proposal_losses"] = boxes, scores

    def faster_rcnn_simulation(self):

        self._profile_helper(self.get_original_images_sizes, "get_original_images_sizes")
        self._profile_helper(self.transform, "transform")
        self._profile_helper(self.backbonefpn, "backbonefpn") 
        self.backbonefpn_details()
        # self.backbonefpn_fpn_details()  # uncomment to reveal details
        self._profile_helper(self.rpn_head, "rpn_head_on_{}_features".format(len(self.args["features_"])))
        self._profile_helper(self.rpn_anchor_generator, "rpn_anchor_generator")
        self.rpn_head_details()  # uncomment to reveal details

        


def faster_rcnn_simulation(model, images, targets=None):

    # get original_image_sizes for postprocessing
    original_image_sizes = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    # transform
    tt.tic("transform")
    images, targets = model.transform(images, targets)
    tt.toc("transform")

    # backbonefpn
    # prof = ProfResnet()
    # prof.itrResLayer(model.backbone.body, "backbonefpn", images.tensors, recursive=True)
    tmp = images.tensors.clone()
    tt.tic("backbone")
    tmp = model.backbone.body(tmp)
    tt.toc("backbone")
    
    tt.tic("backbonefpn")
    features = model.backbone(images.tensors)
    tt.toc("backbonefpn")

    print(type(features))

    # rpn
    tt.tic("rpn")
    proposals, proposal_losses = model.rpn(images, features, targets)
    tt.toc("rpn")

    # # roi
    tt.tic("roi")
    detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    tt.toc("roi")

    # # postprocess
    tt.tic("postprocess")
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
    tt.toc("postprocess")
    
    
    pass

if  __name__ == '__main__':

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
    images = torch.rand(1, 3, 224, 224).to(device)

    for name, layer in model.named_children():
        print(name)
        print(layer)
        print("------------------------------------------------")

    for i in range(2):
        prof = Profiler()
        # prof.mem = True  # uncomment to profile memory consumption
        prof.faster_rcnn_simulation()

    if not prof.mem:
        tt.report()
    else:
        mr.report()



