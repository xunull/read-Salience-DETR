from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector



class SalienceCriterion(nn.Module):
    def __init__(
            self,
            limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
            noise_scale: float = 0.0,
            alpha: float = 0.25,
            gamma: float = 2.0,
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, foreground_mask, targets, feature_strides, image_sizes):
        gt_boxes_list = [] # 各个image的gt （在图像上的真实的坐标，缩放到图像真实尺寸了）
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(boxes * scale_factor)

        mask_targets = []
        for level_idx, (mask, feature_stride) in enumerate(zip(foreground_mask, feature_strides)):
            feature_shape = mask.shape[-2:]
            coord_x, coord_y = self.get_pixel_coordinate(feature_shape, feature_stride, device=mask.device) # 尺寸hw
            masks_per_level = []
            for gt_boxes in gt_boxes_list:
                mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, level_idx)
                masks_per_level.append(mask)

            masks_per_level = torch.stack(masks_per_level)
            mask_targets.append(masks_per_level)
        mask_targets = torch.cat(mask_targets, dim=1) # [bs,s]
        foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1) # [bs,1,s]
        foreground_mask = foreground_mask.squeeze(1) # [bs,s]
        num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale).clamp_(min=1) # 正样本的数量
        salience_loss = (
                sigmoid_focal_loss(
                    foreground_mask,
                    mask_targets,
                    num_pos,
                    alpha=self.alpha,
                    gamma=self.gamma,
                ) * foreground_mask.shape[1]
        )
        return {"loss_salience": salience_loss}

    def get_pixel_coordinate(self, feature_shape, stride, device):
        height, width = feature_shape
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
            indexing="ij",
        ) # 特征图上各个点位 对应于原始图像上的点位坐标，取得中心坐标
        coord_y = coord_y.reshape(-1) # 推平了
        coord_x = coord_x.reshape(-1) # 推平
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
        # gt_label: (m,) gt_boxes: (m, 4)
        # coord_x: (h*w, )
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m) # 所有网格点跟每个gt的左上x坐标的距离值（有正有负）
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1] # 所有网格点跟每个gt的左上y坐标的距离值（有正有负）
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None] # 所有的网格点跟每个gt的右下x坐标的距离值（有正有负）
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None] # 所有的网格点跟每个gt的右下y坐标的距禗值（有正有负）
        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )  # [h*w, m, 4] 上面四个值在一起 m是gt的数量

        # the foreground queries must satisfy two requirements:
        # 1. the quereis located in bounding boxes
        # 2. the distance from queries to the box center match the feature map stride
        min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m] # 每个gt跟网格点距离的最小值（左上右下四个点的插值的最小）
        max_border_distances = torch.max(border_distances, dim=-1)[0] # [h*w, m] # 每个gt跟网格点距离的最大值（左上右下四个点的插值的最大）
        mask_in_gt_boxes = min_border_distances > 0 # mask的是网格点在gt内部的
        min_limit, max_limit = self.limit_range[level_idx] # limit_range ((-1, 64), (64, 128), (128, 256), (256, 99999))
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit) # 将网格点位限制在与gt边框一个距离范围内
        mask_pos = mask_in_gt_boxes & mask_in_level

        # scale-independent salience confidence
        row_factor = left_border_distance + right_border_distance # gt的宽度
        col_factor = top_border_distance + bottom_border_distance # gt的高度
        delta_x = (left_border_distance - right_border_distance) / row_factor # 网格点到坐上点x左边的距离 + 网格点到右下点x的距离 / gt的宽度
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        confidence = torch.sqrt(delta_x ** 2 + delta_y ** 2) / 2

        confidence_per_box = 1 - confidence # 和上面一行构成公式2
        confidence_per_box[~mask_in_gt_boxes] = 0 #  公式1，只有在gt内部的才有值，其他的都是0

        # process positive coordinates
        if confidence_per_box.numel() != 0:
            mask = confidence_per_box.max(-1)[0] # 各个网格点跟哪个gt的公式2的值最大
        else:
            mask = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

        # process negative coordinates
        mask_pos = mask_pos.long().sum(dim=-1) >= 1 # 这里求和表示至少和一个gt有关系
        mask[~mask_pos] = 0 # 如果和gt全部没有关系，那么就是负样本
        # 随机噪声，如果noise_scale为0就是没有添加噪声
        # add noise to add randomness
        mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
        return mask


# dn继承detr, salience继承dn detr
# SalienceDETR has the architecture similar to FocusDETR
class SalienceDETR(DNDETRDetector):
    def __init__(
            # model structure
            self,
            backbone: nn.Module,
            neck: nn.Module,
            position_embedding: nn.Module,
            transformer: nn.Module,
            criterion: nn.Module,
            postprocessor: nn.Module,
            focus_criterion: nn.Module,
            # model parameters
            num_classes: int = 91,
            num_queries: int = 900,
            denoising_nums: int = 100,
            # model variants
            aux_loss: bool = True,
            min_size: int = None,
            max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        # 生成噪声使用 就是对比去噪训练那个
        self.denoising_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=embed_dim,
            denoising_nums=denoising_nums,
            label_noise_prob=0.5,
            box_noise_scale=1.0,
        )
        self.focus_criterion = focus_criterion

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # extract features
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.neck(multi_level_feats)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            # 生成噪声
            noised_results = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_query = noised_results[0]
            noised_box_query = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_query = None
            noised_box_query = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None
        # [6,bs,1100,91/4]  [bs,900,91/4],foreground_mask list [bs,1,h,w]
        # feed into transformer
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )

        # hack implementation for distributed training
        outputs_class[0] += self.denoising_generator.label_encoder.weight[0, 0] * 0.0

        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            # dn的后处理
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

            # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss (一共21项，每个组有class，bbox，giou）一共6层decoder，加上一层encoder
            loss_dict = self.criterion(output, targets)
            # dn的loss（18项，只有decoder的6层）
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # 各个特征图的stride
            # compute focus loss
            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]
            # salience 相关的loss（相比于focus的fts部分的loss计算，这里面的内容更多）
            focus_loss = self.focus_criterion(foreground_mask, targets, feature_stride, images.image_sizes)

            loss_dict.update(focus_loss)

            # 公式12
            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k]) for k in loss_dict.keys() if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections
