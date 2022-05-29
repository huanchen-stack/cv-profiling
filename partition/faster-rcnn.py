"""
Different from the faster-rcnn.py in parent folder, this file aims to simulate only step by step inferences.
Profiling on larger structures is omitted. Refer to the original file to see profiling outputs on larger structures.
* This file also aims to autogenerate the architecture graph with dataflow, so a csv file must be written.
* For simplicity, also use csv file for time and memory consumptions
* This file also aims to perform sanity checks, so make sure the outputs are correct (as running the model as a whole)
Note: the next step focuses on running the code with multiprocesses, so certain changes must be made:
    1. the variable [x] is omitted
    2. whenever parallelism can be made, for loop must be expanded
    3. timer and memorizer operations must be applied accordingly
"""

import torch
# from torchsummary import summary
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
from collections import OrderedDict
from PIL import Image
import numpy as np
from skimage import io, transform
import os
from torchvision import transforms
import csv
from sigfig import round

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

# utils
def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

# utils
def _tensor_size(tensor):
    return f"{round(tensor.element_size() * tensor.nelement() / 1000000, sigfigs=4)} Mb"

def _size_helper(obj):
    if type(obj) == torch.Tensor:
        return  str(obj.size()).replace(', ', 'x'), _tensor_size(obj)
        # print(name, "::" , obj.shape, _tensor_size(obj) )  
    elif type(obj) == type([1, 2]):
        add = 0
        for tensor in obj:
            if type(tensor) != torch.Tensor:
                assert False, f"Expected a tensor or a list of tensors as input, a list of {type(tensor)} was given."
            add += tensor.element_size() * tensor.nelement() / 1000000
        return "List of Tensors", f"{round(add, sigfigs=4)} Mb" 
    else:
        assert False, f"Expected a tensor or a list of tensors as input, a {type(obj)} was given."

# load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

# load images
images = io.imread("input.jpg")
transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
images = transform(images)
images = torch.unsqueeze(images, dim=0).to(device)

# csv file for layer_vertices
layer_vertices = open("layer_vertices.csv", "w")

# csv file for data dependencies
dependencies = open("dependencies.csv", "w")

class Profiler():

    def __init__(self):
        super().__init__()
        self.model = model
        self.images = images
        self.targets = None
        self.args = {}

    def _profile_helper(self, func, name):
        """
        Output everything in layer_vertices.csv
        func() MUST NOT include any modifications on data.
        """
        # time
        tt.tic()
        func()
        tt.toc()

        # memory
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
        out = mr.get_mem(prof_report, usingcuda)

        # write in csv file
        if usingcuda:
            layer_vertices.write(f"{name}, {tt.get_time()}, {out[0]}, {out[1]}\n")
        else:
            layer_vertices.write(f"{name}, {tt.get_time()}, {out}\n")

    def get_original_images_sizes(self):
        # get original_image_sizes for postprocessing

        # warm up
        original_image_sizes = []
        for img in self.images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        self.args["original_image_sizes"] = original_image_sizes


        original_image_sizes = []
        for img in self.images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        # profiling
        def _get_original_images_sizes():
            original_image_sizes = []
            for img in self.images:
                val = img.shape[-2:]
                torch._assert(
                    len(val) == 2,
                    f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                )
                original_image_sizes.append((val[0], val[1]))
        self._profile_helper(_get_original_images_sizes, "get_original_images_sizes")

        # append data dependencies
        shape_, size_ = _size_helper(self.images)
        dependencies.write(f"_images_, get_original_images_sizes, {shape_}, {size_}\n")

    def transform(self):
        # warm up
        tmp_images, self.targets = self.model.transform(self.images, self.targets)

        self.model.transform(self.images, self.targets)
        # profiling
        def _transform():
            self.model.transform(self.images, self.targets)
        self._profile_helper(_transform, "transform")

        # append data dependencies
        shape_, size_ = _size_helper(self.images)
        dependencies.write(f"_images_, transform, {shape_}, {size_}\n")

        # update self.images
        self.images = tmp_images

    def backbonefpn_body(self):
        # load model and variables
        backbonefpn_body = self.model.backbone.body
        tmp_x = self.images.tensors.clone()
        self.args["x"] = []

        # warm up
        for name, layer in backbonefpn_body.named_children():
            tmp_x = layer(tmp_x)
            if name[0:5] == "layer":
                self.args["x"].append(tmp_x.clone())

        # profiling and dependencies
        last_name = "transform"
        tmp_x = self.images.tensors.clone()
        for name, layer in backbonefpn_body.named_children():
            # dependencies
            shape_, size_ = _size_helper(tmp_x)
            dependencies.write(f"{last_name}, {name}, {shape_}, {size_}\n")
            last_name = name
            layer(tmp_x)
            # profiling
            def _layer():
                layer(tmp_x)
            self._profile_helper(_layer, f"{name}")   
            # update tmp_x
            tmp_x = layer(tmp_x)

    def backbonefpn_fpn(self):
        # load model
        backbonefpn_fpn = self.model.backbone.fpn

        # load functions needed
        # attached with profiling
        def get_result_from_inner_blocks(x, idx):
            num_blocks = len(backbonefpn_fpn.inner_blocks)
            if idx < 0:
                idx += num_blocks
            out = x
            for i, module in enumerate(backbonefpn_fpn.inner_blocks):
                if i == idx:
                    # warm up
                    out = module(x)
                    module(x)
                    # profiling
                    def _inner():
                        module(x)
                    self._profile_helper(_inner, f"inner_{idx}")

            return out

        def get_result_from_layer_blocks(x, idx):
            num_blocks = len(backbonefpn_fpn.layer_blocks)
            if idx < 0:
                idx += num_blocks
            out = x
            for i, module in enumerate(backbonefpn_fpn.layer_blocks):
                if i == idx:
                    # warm up
                    out = module(x)
                    module(x)
                    # profiling
                    def _layer():
                        module(x)
                    self._profile_helper(_layer, "layer_{}".format(i))

            return out
        
        # forward

        # load variables
        x = self.args["x"]
        self.args["features_"] = []

        # initial last inner_{idx} and layer_{idx}
        last_inner = get_result_from_inner_blocks(x[-1], -1)
        # dependencies
        shape_, size_ = _size_helper(x[-1])
        dependencies.write(f"layer4, inner_3, {shape_}, {size_}\n")

        self.args["features_"].append(get_result_from_layer_blocks(last_inner, -1))
        shape_, size_ = _size_helper(last_inner)
        dependencies.write(f"inner_3, layer_3, {shape_}, {size_}\n")

        for idx in range(len(x) - 2, -1, -1):
            # inner_{idx}
            inner_lateral = get_result_from_inner_blocks(x[idx], idx)
            # dependencies
            shape_, size_ = _size_helper(x[idx])
            dependencies.write(f"layer{idx+1}, inner_{idx}, {shape_}, {size_}\n")
            
            feat_shape = inner_lateral.shape[-2:]

            # interpolate
            # warm up
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            F.interpolate(last_inner, size=feat_shape, mode="nearest")
            # profiling
            def _interpolate():
                F.interpolate(last_inner, size=feat_shape, mode="nearest")
            self._profile_helper(_interpolate, f"interpolate__{idx}")
            # dependencies
            shape_, size_ = _size_helper(last_inner)
            if idx == 2:
                dependencies.write(f"inner_3, interpolate__2, {shape_}, {size_}\n")
            else:
                dependencies.write(f"add__{idx+1}, interpolate__{idx}, {shape_}, {size_})\n")

            # addition
            last_inner = inner_lateral + inner_top_down
            # dependencies
            shape_, size_ = _size_helper(inner_lateral)
            dependencies.write(f"inner_{idx}, add__{idx}, {shape_}, {size_}\n")
            shape_, size_ = _size_helper(inner_top_down)
            dependencies.write(f"interpolate__{idx}, add__{idx}, {shape_}, {size_}\n")

            # layer_{idx}
            self.args["features_"].insert(0, get_result_from_layer_blocks(last_inner, idx))
            # dependencies
            shape_, size_ = _size_helper(last_inner)
            dependencies.write(f"add__{idx}, layer_{idx}, {shape_}, {size_}\n")

        # extra layer
        if backbonefpn_fpn.extra_blocks is not None:
            # names.append("pool")
            
            # warm up
            self.args["features_"].append(F.max_pool2d(self.args["features_"][-1], 1, 2, 0))
            F.max_pool2d(self.args["features_"][-2], 1, 2, 0)
            # profiling
            def _extra():
                F.max_pool2d(self.args["features_"][-2], 1, 2, 0)
            self._profile_helper(_extra, "extra")
            # dependencies
            shape_, size_ = _size_helper(self.args["features_"][-2])
            dependencies.write(f"layer_3, extra, {shape_}, {size_}\n")

        names = ['0', '1', '2', '3', 'pool']
        self.args["features"] = OrderedDict([(k, v) for k, v in zip(names, self.args["features_"])])

    def update_features(self):
        names = ['0', '1', '2', '3']
        x = OrderedDict([(k, v) for k, v in zip(names, self.args["x"])])
        self.args["features"] = self.model.backbone.fpn(x)
        self.args["features_"] = list(self.args["features"].values())

    def rpn_head(self):
        # load model
        rpn_head = self.model.rpn.head
        
        # forward
        logits = []
        bbox_reg = []

        convs = []
        convs.append(rpn_head.conv)
        convs.append(torch.nn.ReLU().to(device))
        conv = torch.nn.Sequential(*convs)

        idx = 0  # create feature index for model architecture graph
        for feature in self.args["features_"]:
            # conv layer
            # warm up
            t = conv(feature)
            conv(feature)
            # profiling
            def _conv():
                conv(feature)
            self._profile_helper(_conv, f"conv_f{idx}")
            # dependencies
            shape_, size_ = _size_helper(feature)
            if idx != 4:
                dependencies.write(f"layer_{idx}, conv_f{idx}, {shape_}, {size_}\n")
            else:
                dependencies.write(f"extra, conv_f{idx}, {shape_}, {size_}\n")

            # logits
            # warm up
            logits.append(rpn_head.cls_logits(t))
            rpn_head.cls_logits(t)
            # profiling
            def _logits():
                rpn_head.cls_logits(t)
            self._profile_helper(_logits, f"cls_logits_f{idx}")
            # dependencies
            shape_, size_ = _size_helper(t)
            dependencies.write(f"conv_f{idx}, cls_logits_f{idx}, {shape_}, {size_}\n")
            
            # bbox_pred
            # warm up 
            bbox_reg.append(rpn_head.bbox_pred(t))
            rpn_head.bbox_pred(t)
            # profiling
            def _bbox_reg():
                rpn_head.bbox_pred(t)
            self._profile_helper(_bbox_reg, f"bbox_pred_f{idx}")
            # dependencies
            shape_, size_ = _size_helper(t)
            dependencies.write(f"conv_f{idx}, bbox_pred_f{idx}, {shape_}, {size_}\n")
            
            idx += 1
        

        self.args["objectness"] = logits
        self.args["pred_bbox_deltas"] = bbox_reg

    def rpn_anchor_generator(self):
        # anchor generation
        # warm up
        anchors = self.model.rpn.anchor_generator(self.images, self.args["features_"])
        self.model.rpn.anchor_generator(self.images, self.args["features_"])
        # profiling
        def _anchor():
            self.model.rpn.anchor_generator(self.images, self.args["features_"])
        self._profile_helper(_anchor, "anchor_generator")
        # dependencies
        idx = 0
        for feature in self.args["features_"]:
            shape_, size_ = _size_helper(feature)
            if idx != 4:
                dependencies.write(f"layer_{idx}, anchor_generator, {shape_}, {size_}\n")
            else:
                dependencies.write(f"extra, anchor_generator, {shape_}, {size_}\n")
            idx += 1

        # postprocessing
        # warm up
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in self.args["objectness"]]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        self.args["objectness_"], self.args["pred_bbox_deltas_"] = concat_box_prediction_layers(self.args["objectness"], self.args["pred_bbox_deltas"])
        proposals = model.rpn.box_coder.decode(self.args["pred_bbox_deltas_"].detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = model.rpn.filter_proposals(proposals, self.args["objectness_"], self.images.image_sizes, num_anchors_per_level)
        self.args["proposals"], self.args["proposal_losses"] = boxes, scores

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in self.args["objectness"]]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        # self.args["objectness_"], self.args["pred_bbox_deltas"] = concat_box_prediction_layers(self.args["objectness"], self.args["pred_bbox_deltas"])
        concat_box_prediction_layers(self.args["objectness"], self.args["pred_bbox_deltas"])
        proposals = model.rpn.box_coder.decode(self.args["pred_bbox_deltas_"].detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = model.rpn.filter_proposals(proposals, self.args["objectness_"], self.images.image_sizes, num_anchors_per_level)
        # profiling
        def _postprocessing():
            num_images = len(anchors)
            num_anchors_per_level_shape_tensors = [o[0].shape for o in self.args["objectness"]]
            num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
            # self.args["objectness_"], self.args["pred_bbox_deltas"] = concat_box_prediction_layers(self.args["objectness"], self.args["pred_bbox_deltas"])
            concat_box_prediction_layers(self.args["objectness"], self.args["pred_bbox_deltas"])
            proposals = model.rpn.box_coder.decode(self.args["pred_bbox_deltas_"].detach(), anchors)
            proposals = proposals.view(num_images, -1, 4)
            boxes, scores = model.rpn.filter_proposals(proposals, self.args["objectness_"], self.images.image_sizes, num_anchors_per_level)
            # self.args["proposals"], self.args["proposal_losses"] = boxes, scores
        self._profile_helper(_postprocessing, "anchor_postprocessing")

        # dependencies
        shape_, size_ = _size_helper(anchors)
        dependencies.write(f"anchor_generator, anchor_postprocessing, {shape_}, {size_}\n")
        idx = 0
        for cls_logit in self.args["objectness"]:
            shape_, size_ = _size_helper(cls_logit)
            dependencies.write(f"cls_logits_f{idx}, anchor_postprocessing, {shape_}, {size_}\n")
            idx += 1
        idx = 0
        for bbox_pred in self.args["pred_bbox_deltas"]:
            shape_, size_ = _size_helper(bbox_pred)
            dependencies.write(f"bbox_pred_f{idx}, anchor_postprocessing, {shape_}, {size_}\n")
            idx += 1

    def roi_box_roi_pool(self):
        # load modules
        box_roi_pool = self.model.roi_heads.box_roi_pool
        box_head = self.model.roi_heads.box_head
        box_predictor = self.model.roi_heads.box_predictor

        # load parameters
        image_shapes = self.images.image_sizes
        names = ['0', '1', '2', '3', 'pool']
        self.args["features"] = OrderedDict([(k, v) for k, v in zip(names, self.args["features_"])])
        self.args["box_features"] = box_roi_pool(self.args["features"], self.args["proposals"], [(800, 800)])
        
        box_roi_pool(self.args["features"], self.args["proposals"], [(800, 800)])
        # profiling
        def _box_roi_pool():
            box_roi_pool(self.args["features"], self.args["proposals"], [(800, 800)])
        self._profile_helper(_box_roi_pool, "box_roi_pool")
        
        # dependencies
        idx = 0
        for feature in self.args["features_"]:
            shape_, size_ = _size_helper(feature)
            if idx != 4:
                dependencies.write(f"layer_{idx}, box_roi_pool, {shape_}, {size_}\n")
            else:
                dependencies.write(f"extra, box_roi_pool, {shape_}, {size_}\n")
            idx += 1
        shape_, size_ = _size_helper(self.args["proposals"])
        dependencies.write(f"anchor_postprocessing, box_roi_pool, {shape_}, {size_}\n")

    def roi_box_head(self):
        # load module
        box_head = self.model.roi_heads.box_head
        
        # load variables
        x = self.args["box_features"].clone()
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)

        # fc6
        # warm up
        tmp_x = F.relu(box_head.fc6(x))
        F.relu(box_head.fc6(x))
        # profiling
        def _fc6():
            F.relu(box_head.fc6(x))
        self._profile_helper(_fc6, "fc6")
        # dependencies
        shape_, size_ = _size_helper(x)
        dependencies.write(f"box_roi_pool, fc6, {shape_}, {size_}\n")
        # reload x
        x = tmp_x

        # fc7
        # warm up
        tmp_x = F.relu(box_head.fc7(x))
        F.relu(box_head.fc7(x))
        # profiling
        def _fc7():
            F.relu(box_head.fc7(x))
        self._profile_helper(_fc7, "fc7")
        # dependencies
        shape_, size_ = _size_helper(x)
        dependencies.write(f"fc6, fc7, {shape_}, {size_}\n")

        # update parameters
        self.args["box_features_"] = tmp_x

    def roi_box_predictor(self):
        # load module
        box_predictor = self.model.roi_heads.box_predictor
        # load parameters
        x = self.args["box_features_"].clone()
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)
        
        # cls_score
        # warm up
        self.args["class_logits"] = box_predictor.cls_score(x)
        box_predictor.cls_score(x)
        # profiling
        def _cls_score():
            box_predictor.cls_score(x)
        self._profile_helper(_cls_score, "cls_score")
        # dependencies
        shape_, size_ = _size_helper(x)
        dependencies.write(f"fc7, cls_score, {shape_}, {size_}\n")

        # bbox_pred_roi_
        # warm up
        self.args["box_regression"] = box_predictor.bbox_pred(x)
        box_predictor.bbox_pred(x)
        # profiling
        def _bbox_pred():
            box_predictor.bbox_pred(x)
        self._profile_helper(_bbox_pred, "bbox_pred_roi_")
        # dependencies
        shape_, size_ = _size_helper(x)
        dependencies.write(f"fc7, bbox_pred_roi_, {shape_}, {size_}\n")

    def postprocess_detection(self):
        # warm up
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(
                                    self.args["class_logits"], self.args["box_regression"], 
                                    self.args["proposals"], self.images.image_sizes
                                )
        self.model.roi_heads.postprocess_detections(
                self.args["class_logits"], self.args["box_regression"], 
                self.args["proposals"], self.images.image_sizes
            )
        # profiling
        def _postprocess_detections():
            self.model.roi_heads.postprocess_detections(
                self.args["class_logits"], self.args["box_regression"], 
                self.args["proposals"], self.images.image_sizes
            )
        self._profile_helper(_postprocess_detections, "postprocess_detections")
        # dependencies
        shape_, size_ = _size_helper(self.args["class_logits"])
        dependencies.write(f"cls_score, postprocess_detections, {shape_}, {size_}\n")
        shape_, size_ = _size_helper(self.args["box_regression"])
        dependencies.write(f"bbox_pred_roi_, postprocess_detections, {shape_}, {size_}\n")
        shape_, size_ = _size_helper(self.args["proposals"])
        dependencies.write(f"anchor_postprocessing, postprocess_detections, {shape_}, {size_}\n")
        dependencies.write(f"postprocess_detections, postprocess_resize, final_output, depends on #boxes")
        
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

    def postprocess(self):

        def resize_boxes(boxes, original_size, new_size):
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=boxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
                for s, s_orig in zip(new_size, original_size)
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = boxes.unbind(1)

            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            return torch.stack((xmin, ymin, xmax, ymax), dim=1)

        result = self.args["detections"]
        image_shapes = [(800, 800)]
        # original_image_sizes = self.args["original_image_sizes"]
        original_image_sizes = [(224, 224)]
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes

        print("============================================================")
        print(result)

    def faster_rcnn_simulation(self):
        # self.get_original_images_sizes()
        self.transform()
        self.backbonefpn_body()
        self.backbonefpn_fpn()
        # self.update_features()
        self.rpn_head()
        self.rpn_anchor_generator()
        self.roi_box_roi_pool()
        self.roi_box_head()
        self.roi_box_predictor()
        self.postprocess_detection()
        self.postprocess()


if  __name__ == '__main__':

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
    # images = torch.rand(1, 3, 224, 224).to(device)

    result_model = model(images)
    prof = Profiler()
    prof.faster_rcnn_simulation()

    print("--------------------------------------------------")
    print(result_model)

    print(tt.get_agg())
    model(images)
    tt.tic()
    model(images)
    tt.toc()
    print(tt.get_time())


# close the csv files
layer_vertices.close()
dependencies.close()

gv = open("graph.gv", "w")
gv.write("digraph G {\n\n")

layer_vertices = open("layer_vertices.csv", "r")
for line in layer_vertices.readlines():
    line = line.rstrip().split(", ")
    node = line[0]
    if len(line) == 4:
        label = f"{line[0]}\\ntime: {line[1]}\\ncpu: {line[2]} Mb\\ncuda: {line[3]} Mb"
    elif len(line) == 3:
        label = f"{line[0]}\\ntime: {line[1]}\\ncpu: {line[2]} Mb"
    gv.write(f"\t{node} [label=\"{label}\"]\n")
layer_vertices.close()

gv.write("\n")
dependencies = open("dependencies.csv", "r")
for line in dependencies.readlines():
    line = line.strip().split(', ')
    src = line[0]
    dst = line[1]
    shape = line[2]
    size = line[3]
    gv.write(f"\t{src} -> {dst} [label=\"{shape}\\n{size}\"]\n")
dependencies.close()

gv.write("}")
gv.close()