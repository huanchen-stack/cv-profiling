import torch
# from torchsummary import summary
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
from collections import OrderedDict

from timer import Clock
from memorizer import MemRec

device = "cuda:0" if torch.cuda.is_available() else "cpu"
usingcuda = device == "cuda:0"

tt = Clock()
mr = MemRec()

import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, concat_box_prediction_layers
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor


def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

def _tensor_size(tensor):
    tensor.to(device)
    return f"{tensor.element_size() * tensor.nelement() / 1000000} Mb"

class Profiler():
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        # self.backbone = torchvision.models.resnet50(
        #     pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d
        # )  # FAKE
        self.images = torch.rand(1, 3, 224, 224).to(device)
        self.targets = None
        self.args = {}
        self.mem = False  # spliting up time and mem profiling to avoid doublewriting args
        self.backonefpn = None

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

    def _size_helper(self, name, obj):
        if type(obj) == torch.Tensor:
            print(name, "::" , obj.shape, _tensor_size(obj) )
        
        elif type(obj) == torchvision.models.detection.image_list.ImageList:
            print(name, "::", obj.tensors.shape, _tensor_size(obj.tensors))
        
        elif type(obj) == OrderedDict:
            all = 0
            for i in range(len(list(obj.values()))):
                tmp = list(obj.values())[i]
                size = tmp.element_size() * tmp.nelement()
                print(name, end=f'[{i}] :: ')
                print(tmp.shape, _tensor_size(tmp))
                all += size
            print(name, '::', all)
        elif type(obj) == list:
            all = 0
            for i in range(len(obj)):
                tmp = obj[i]
                size = tmp.element_size() * tmp.nelement()
                print(name, end=f'[{i}] :: ')
                print(tmp.shape, _tensor_size(tmp))
                all += size
            print(name, '::', all)


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
        self._size_helper("transform_in([input])", self.images)
        self.images, self.targets = self.model.transform(self.images, self.targets)
        self._size_helper("transform_out([images])", self.images)

    def backbonefpn_details(self):
        # backbonefpn_body setup
        backbonefpn_body = self.model.backbone.body
        # forward
        def _backbonefpn_body():
            self.args["x"] = backbonefpn_body(self.images.tensors)
        self._profile_helper(_backbonefpn_body, "backbonefpn_body")

        # backbonefpn_fpn setup 
        backbonefpn_fpn = self.model.backbone.fpn
        # forward
        def _backbonefpn_fpn():
            self.args["features"] = backbonefpn_fpn(self.args["x"])
        self._profile_helper(_backbonefpn_fpn, "backbonefpn_fpn")

        # feature_ list setup
        self.args["features_"] = list(self.args["features"].values())
        
        # self.args["features"] = model.backbone(self.images.tensors)
        # self.args["features_"] = list(model.backbone(self.images.tensors).values())

    def backbonefpn_fpn_details(self):
        """The aim of this function is to prevent nested profiling when profiling memory."""
        # backbonefpn_body setup 
        # backbone = self.backbone
        # returned_layers = [1, 2, 3, 4]
        # return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        # # backbonefpn_body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # # backbonefpn_fpn params 
        # in_channels_stage2 = backbone.inplanes // 8
        # in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        # out_channels = 256
        # extra_blocks = LastLevelMaxPool()
        # # backbonefpn_fpn setup for blocks
        # backbonefpn_fpn = FeaturePyramidNetwork(
        #     in_channels_list=in_channels_list,
        #     out_channels=out_channels,
        #     extra_blocks=extra_blocks,
        # )
        backbonefpn_fpn = self.model.backbone.fpn

        def get_result_from_inner_blocks(x, idx):
            num_blocks = len(backbonefpn_fpn.inner_blocks)
            if idx < 0:
                idx += num_blocks
            out = x
            for i, module in enumerate(backbonefpn_fpn.inner_blocks):
                if i == idx:
                    # print(i, module)
                    def _inner():
                        module(x)
                    self._profile_helper(_inner, "backbonefpn_fpn_inner_{}".format(i))
                    self._size_helper(f"backbonefpn_fpn_inner_{i}_in", x)
                    out = module(x)
                    self._size_helper(f"backbonefpn_fpn_inner_{i}_out", out)
            return out

        def get_result_from_layer_blocks(x, idx):
            num_blocks = len(backbonefpn_fpn.layer_blocks)
            if idx < 0:
                idx += num_blocks
            out = x
            for i, module in enumerate(backbonefpn_fpn.layer_blocks):
                if i == idx:
                    # print(i, module)
                    def _layer():
                        module(x)
                    self._profile_helper(_layer, "backbonefpn_fpn_layer_{}".format(i))
                    self._size_helper(f"backbonefpn_fpn_layer_{i}_in", x)
                    out = module(x)
                    self._size_helper(f"backbonefpn_fpn_layer_{i}_out", out)
            return out
        
        # forward
        self._size_helper("backbonefpn_body_out([x])", self.args["x"])
        x = self.args["x"]
        names = list(x.keys())
        x = list(x.values())

        last_inner = get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]

            def _interpolate():
                inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            self._profile_helper(_interpolate, "backbonefpn_fpn_interpolate{}".format(idx))
            self._size_helper(f"fpn_interpolate{idx}_in", last_inner)
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            self._size_helper(f"fpn_interpolate{idx}_out", inner_top_down)
            self._size_helper(f"sum{idx}_topdown", inner_top_down)
            self._size_helper(f"sum{idx}_lateral", inner_lateral)
            last_inner = inner_lateral + inner_top_down
            
            results.insert(0, get_result_from_layer_blocks(last_inner, idx))

        # extra layer
        if backbonefpn_fpn.extra_blocks is not None:
            names.append("pool")
            def _extra():
                x.append(F.max_pool2d(x[-1], 1, 2, 0))
            self._profile_helper(_extra, "backbonefpn_fpn_extra")
            results = x

        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        self._size_helper(f"backbone_fpn_out([features])", out)

    def rpn_head(self):
        # RPNHead setup
        rpn_head = self.model.rpn.head
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
        # out_channels = self.model.backbone.out_channels
        # rpn_anchor_generator = _default_anchorgen()
        # rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_head = self.model.rpn.head
        # forward
        logits = []
        bbox_reg = []

        self._size_helper("rpn_head_in", self.args["features_"])
        idx = 0
        for feature in self.args["features_"]:
            t = rpn_head.conv(feature)
            convs = []
            for name, layer in rpn_head.named_children():
                convs.append(layer)

            # details
            self._size_helper(f"rpn_head_conv_in(on feature[{idx}])", feature)
            x = convs[0](feature)
            self._size_helper(f"rpn_head_conv_out(on feature[{idx}])", x)

            self._size_helper(f"rpn_head_logits_in(on feature[{idx}])", x)
            y = convs[1](x)
            self._size_helper(f"rpn_head_logits_out(on feature[{idx}])", y)

            self._size_helper(f"rpn_head_bbox_in(on feature[{idx}])", x)
            z = convs[2](x)
            self._size_helper(f"rpn_head_bbox_out(on feature[{idx}])", z)

            idx += 1

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

        self._size_helper("rpn_anchor_out([proposals])", self.args["proposals"])

    def roi_heads(self):
        self.args["detections"], self.args["detector_losses"] = self.model.roi_heads(
           self.args["features"], self.args["proposals"], self.images.image_sizes, self.targets
        )

    def roi_heads_details(self):
        # initialize
        # box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        # out_channels = out_channels
        # resolution = box_roi_pool.output_size[0]  # (output_size=7) -> output_size=(7, 7)
        # representation_size = 1024
        # box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        # representation_size = 1024
        # num_classes = 91  # by default
        # box_predictor = FastRCNNPredictor(representation_size, num_classes)

        box_roi_pool = self.model.roi_heads.box_roi_pool
        box_head = self.model.roi_heads.box_head
        box_predictor = self.model.roi_heads.box_predictor

        image_shapes = self.images.image_sizes

        def _box_roi_pool():
            self.args["box_features"] = box_roi_pool(self.args["features"], self.args["proposals"], self.images.image_sizes)
        self._profile_helper(_box_roi_pool, "roi_heads_box_roi_pool")
        self._size_helper("roi_head_box_roi_pool", self.args["box_features"])
        # box_features = box_roi_pool(self.args["features"], self.args["proposals"], self.images.image_sizes)

        def _box_head():
            self.args["box_features_"] = box_head(self.args["box_features"])
        self._profile_helper(_box_head, "roi_heads_box_head")
        self._size_helper("roi_heads_box_head_in", self.args["box_features"])
        self._size_helper("roi_heads_box_head_out", self.args["box_features_"])
        # box_features = box_head(box_features)

        def _box_predictor():
            self.args["class_logits"], self.args["box_regression"] = box_predictor(self.args["box_features_"])
        self._profile_helper(_box_predictor, "roi_heads_box_predictor")
        self._size_helper("roi_heads_box_predictor_cls_scores_in", self.args["box_features_"])
        self._size_helper("roi_heads_box_predictor_bbox_pred_in", self.args["box_features_"])
        self._size_helper("roi_heads_box_predictor_cls_scores_out", self.args["class_logits"])
        self._size_helper("roi_heads_box_predictor_bbox_pred_out", self.args["box_regression"])
        # class_logits, box_regression = box_predictor(box_features)

        def _postprocess_detections():
            self.model.roi_heads.postprocess_detections(
                self.args["class_logits"], self.args["box_regression"], 
                self.args["proposals"], self.images.image_sizes
            )
        self._profile_helper(_postprocess_detections, "roi_heads_postprocess_detections")
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(
                                    self.args["class_logits"], self.args["box_regression"], 
                                    self.args["proposals"], self.images.image_sizes
                                )
        # in per image inferences, num_images = 0, thus takes VERY small time/mem
        result = []
        losses = {}
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        self.args["detections"], self.args["detector_losses"] = result, losses

    def roi_heads_box_head_details(self):
        box_head = self.model.roi_heads.box_head
        x = self.args["box_features"]
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)

        def _fc6():
            F.relu(box_head.fc6(x))
        self._profile_helper(_fc6, "roi_heads_box_head_fc6")
        self._size_helper("roi_heads_box_head_fc6_in", x)
        x = F.relu(box_head.fc6(x))
        self._size_helper("roi_heads_box_head_fc6_out", x)

        def _fc7():
            F.relu(box_head.fc7(x))
        self._profile_helper(_fc7, "roi_heads_box_head_fc7")
        self._size_helper("roi_heads_box_head_fc7_in", x)
        x = F.relu(box_head.fc7(x))
        self._size_helper("roi_heads_box_head_fc7_out", x)

    def roi_heads_box_predictor_details(self):
        box_predictor = self.model.roi_heads.box_predictor
        x = self.args["box_features_"]
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)
        
        def _cls_score():
            box_predictor.cls_score(x)
        self._profile_helper(_cls_score, "roi_heads_box_predictor_cls_score(linear)")
        
        def _bbox_pred():
            box_predictor.bbox_pred(x)
        self._profile_helper(_bbox_pred, "roi_heads_box_predictor_bbox_pred(linear)")

    def faster_rcnn_simulation(self):

        self._profile_helper(self.get_original_images_sizes, "get_original_images_sizes")
        self._profile_helper(self.transform, "transform")

        self.backbonefpn_details()
        self.backbonefpn_fpn_details()  # uncomment to reveal details

        self._profile_helper(self.rpn_head, "rpn_head_on_{}_features".format(len(self.args["features_"])))
        self.rpn_head_details()  # uncomment to reveal details
        self._profile_helper(self.rpn_anchor_generator, "rpn_anchor_generator")

        self._profile_helper(self.roi_heads, "roi_heads")
        self.roi_heads_details()  # uncomment to reveal details

        self.roi_heads_box_head_details()  # uncomment to reveal details
        self.roi_heads_box_predictor_details()  # uncomment to reveal details
        


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



