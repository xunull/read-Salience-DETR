import copy
import math
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed
from util.misc import inverse_sigmoid


# 这个是FTS模块
# 最后输出维度为1，表示前景分数
# Focus DETR中的FTS模块的那个预测头
class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)  # 计算每个channel在所有的token上的均值
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


class SalienceTransformer(TwostageTransformer):
    def __init__(
            self,
            encoder: nn.Module,
            neck: nn.Module,
            decoder: nn.Module,
            num_classes: int,
            num_feature_levels: int = 4,
            two_stage_num_proposals: int = 900,
            level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
            # focus detr中为 cascade_set
            layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # salience parameters
        # 保存在模型中，不需要训练，如果载入模型需要有这两个值
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        # encoder的分类头
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        # encoder的坐标头
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        # focus detr中的Multi-category score predictor
        # 就是encoder的分类头
        self.encoder.enhance_mcsp = self.encoder_class_head
        # focus detr中的fts
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        # initialize encoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        # initialize alpha
        self.alpha.data.uniform_(-0.3, 0.3)

    def forward(
            self,
            # List[ [bs,256,h,w] ]
            multi_level_feats,
            # List[ [bs,h,w] ]
            multi_level_masks,
            # List[ [bs,256,h,w] ]
            multi_level_pos_embeds,
            # [bs,200,256]
            noised_label_query,
            # [bs,200,4]
            noised_box_query,
            # [query+200,query+200]
            attn_mask,
    ):
        # get input for encoder
        # 各层hw拉平 -> [bs,all hw,256]
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        # [bs,all hw]
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        # [bs,all hw,256]
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        # spatial_shapes 各层的hw
        # level_start_index[4,2] 各个level的起始位置
        # valid_ratios [2,4,2]
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)

        # 但这里，特征经过了筛选，根据proposal的有效规则，过滤了一些特征
        backbone_output_memory = self.gen_encoder_output_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )[0]

        # True False颠倒的
        # calculate filtered tokens numbers for each feature map
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        # [bs,4]
        # 有效的token的数量
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        # 有效的数量 * 需要focus的比例 level_filter_ratio tensor([0.4000, 0.8000, 1.0000, 1.0000], device='cuda:0')
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        # 在各个bs上，每一层最大的数量
        level_token_nums = focus_token_nums.max(0)[0]
        # 每个image上token的数量
        focus_token_nums = focus_token_nums.sum(-1)

        # from high level to low level
        batch_size = feat_flatten.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):  # 3 2 1 0  # 公式3 那一部分

            start_index = level_start_index[level_idx]

            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            # 取出该层级的token
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            # 取出对应的mask
            mask = mask_flatten[:, start_index:end_index]

            # update the memory using the higher-level score_prediction
            if level_idx != spatial_shapes.shape[0] - 1:
                # 在这个时间节点，score是上一轮的score（因此在后面，score需要变形成hw的形式），将上一轮的score上采样
                upsample_score = torch.nn.functional.interpolate(
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                # [bs,1,h,w] -> [bs,1,hw]
                upsample_score = upsample_score.view(batch_size, -1, spatial_shapes[level_idx].prod())
                # [bs,s(hw),1]
                upsample_score = upsample_score.transpose(1, 2)
                # 公式3中的MLP内的操作
                level_memory = level_memory + level_memory * upsample_score * self.alpha[level_idx]

            # Focus DETR的FTS头 这个在这个论文中就是salience score
            # predict the foreground score of the current layer
            score = self.enc_mask_predictor(level_memory)
            # squeeze后score [bs,s], mask的位置 都填上 score中的最小值
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())
            # [bs,s,1] -> [bs,1,s] -> [bs,1,h,w]
            score = score.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx])
            # 每一层有各自的topk数量 根据数量取出对应的分数，以及位置
            # get the topk salience index of the current feature map level
            level_score, level_inds = valid_score.topk(level_token_nums[level_idx], dim=1)
            # index要加上各层layer的偏移量
            level_inds = level_inds + level_start_index[level_idx]
            # 各个点位的显著分数
            salience_score.append(score)
            # 被选择的token的位置
            selected_inds.append(level_inds)
            selected_score.append(level_score)  # 被选择的token的分数
        # selected_score[::-1]的作用：在上面for以后，selected_score是从高层到低层的，::-1变换顺序
        selected_score = torch.cat(selected_score[::-1], 1)
        index = torch.sort(selected_score, dim=1, descending=True)[1]  # 在各自图像内进行排序
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)  # 同样，对selected_inds变换顺序，然后根据排序规则，重新排列位置

        # create layer-wise filtering
        num_inds = selected_inds.shape[1]  # 选择的index总数（各个image数量是一致的）
        # change dtype to avoid shape inference error during exporting ONNX

        cast_dtype = num_inds.dtype if torchvision._is_tracing() else torch.int64
        # 在各个encoder层使用多少token
        layer_filter_ratio = (num_inds * self.layer_filter_ratio).to(cast_dtype)
        # 从selected_inds中选择，每次都选取前面的token，得到每一层encoder使用的token
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]
        # 与上面相似，从高到底 变成从低到高
        salience_score = salience_score[::-1]
        # list -> [bs,s]
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        # mask部分填充最小值
        foreground_score = foreground_score.masked_fill(mask_flatten, foreground_score.min())

        # 输入encoder
        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,  # [bs,s,256]
            query_pos=lvl_pos_embed_flatten,  # [bs,s,256]
            query_key_padding_mask=mask_flatten,  # [bs,s]
            spatial_shapes=spatial_shapes,  # [4,2]
            level_start_index=level_start_index,  # 4
            valid_ratios=valid_ratios,  # [bs,4,2]
            # salience input
            foreground_score=foreground_score,  # [bs,s] 前景分数
            focus_token_nums=focus_token_nums,  # focus的token数量
            foreground_inds=selected_inds,  # 各个encoder层使用的前进的index
            multi_level_masks=multi_level_masks,  # 各个特征层的mask
        )


        if self.neck is not None:
            # 将memory按照各层的数量切分开
            feat_unflatten = memory.split(spatial_shapes.prod(-1).unbind(), dim=1)
            # 各层的memory reshape成[bs,256,h,w]
            feat_unflatten = dict((
                                      i,
                                      feat.transpose(1, 2).contiguous().reshape(-1, self.embed_dim, *spatial_shape),
                                  ) for i, (feat, spatial_shape) in enumerate(zip(feat_unflatten, spatial_shapes)))
            # 经过 transformer中的neck
            # 调用values 是因为返回的是dict
            feat_unflatten = list(self.neck(feat_unflatten).values())
            # 在变回 [bs,s,256]
            memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feat_unflatten], dim=1)
        # [bs,s,256] [bs,s,4]
        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        enc_outputs_class = self.encoder_class_head(output_memory)
        # [bs,s,4]
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals

        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        if torchvision._is_tracing():

            topk = torch.min(torch.tensor(self.two_stage_num_proposals * 4), enc_outputs_class.shape[1])
        else:
            # 900，每一层900
            topk = min(self.two_stage_num_proposals * 4, enc_outputs_class.shape[1])

        topk_scores, topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)
        # [bs,900,1]
        topk_index = self.nms_on_topk_index(
            topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
        ).unsqueeze(-1)
        # [bs,900,91]
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, self.num_classes))
        # [bs,900,4]
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()

        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            # 前面拼上噪声
            target = torch.cat([noised_label_query, target], 1)
            # 前面拼上噪声
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        # 输入decoder
        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        # salience_score 3.1的内容
        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score

    @staticmethod
    def fast_repeat_interleave(input, repeats):
        """torch.Tensor.repeat_interleave is slow for one-dimension input for unknown reasons. 
        This is a simple faster implementation. Notice the return shares memory with the input.

        :param input: input Tensor
        :param repeats: repeat numbers of each element in the specified dim
        :param dim: the dimension to repeat, defaults to None
        """
        # the following inplementation runs a little faster under one-dimension settings
        return torch.cat([aa.expand(bb) for aa, bb in zip(input, repeats)])  # 每一项都重复repeats次

    @torch.no_grad()
    def nms_on_topk_index(
            self, topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
    ):
        batch_size, num_topk = topk_scores.shape  # num_topk = 3600 每一层900个
        if torchvision._is_tracing():
            num_pixels = spatial_shapes.prod(-1).unbind()
        else:
            num_pixels = spatial_shapes.prod(-1).tolist()  # 4个hw的乘积
        # bs维度也拉平
        # flatten topk_scores and topk_index for batched_nms
        topk_scores, topk_index = map(lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        # [0,1,2,3]
        level_index = torch.arange(level_start_index.shape[0], device=level_start_index.device)
        # num_pixels 是四个尺度的hw的乘积（就是像素的数量） 这里将 各个特征图的宽，各个level的起始位置，level的index 各自重复num_pixels次，然后拉平，然后根据topk_index取出对应的值
        feat_width, start_index, level_idx = map(
            lambda x: self.fast_repeat_interleave(x, num_pixels)[topk_index],
            (spatial_shapes[:, 1], level_start_index, level_index),
        )
        #
        topk_spatial_index = topk_index - start_index
        # 余数是x坐标
        x = topk_spatial_index % feat_width
        # 整数是y坐标
        y = torch.div(topk_spatial_index, feat_width, rounding_mode="trunc")
        # 论文中的公式
        coordinates = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)
        # image的index
        # get unique idx for queries in different images and levels
        image_idx = torch.arange(batch_size).repeat_interleave(num_topk, 0)

        image_idx = image_idx.to(level_idx.device)
        # level_idx 最大是3  shape[0]是4*index 后 可以做到完美的偏移
        idxs = level_idx + level_start_index.shape[0] * image_idx
        # 在各个图片，各个level上进行nms计算
        # perform batched_nms
        indices = torchvision.ops.batched_nms(coordinates, topk_scores, idxs, iou_threshold)

        # stack valid index
        results_index = []
        if torchvision._is_tracing():
            min_num = torch.tensor(self.two_stage_num_proposals)
        else:
            min_num = self.two_stage_num_proposals
        # get indices in each image
        for i in range(batch_size):
            # image_idx[indices] 按index的顺序取出来，然后==i 就是各自的image 然后在从indices中取出（indices中存储的就是有意义的index）
            topk_index_per_image = topk_index[indices[image_idx[indices] == i]]
            # 更新min_num, 保证每个image都有min_num个
            if torchvision._is_tracing():
                min_num = torch.min(topk_index_per_image.shape[0], min_num)
            else:
                min_num = min(topk_index_per_image.shape[0], min_num)

            results_index.append(topk_index_per_image)
        # 在各自的image上去min_num个
        return torch.stack([index[:min_num] for index in results_index])


class SalienceTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            d_ffn=1024,
            dropout=0.1,
            n_heads=8,
            activation=nn.ReLU(inplace=True),
            n_levels=4,
            n_points=4,
            # focus parameter
            topk_sa=300,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.topk_sa = topk_sa
        # 正常的encoder只有一个attention，这里多了一个
        # pre attention, batch_first batch维度在前
        self.pre_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.pre_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.pre_attention.out_proj.weight)
        # initilize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
            self,
            query,
            query_pos,
            value,  # focus parameter
            reference_points,
            spatial_shapes,
            level_start_index,
            query_key_padding_mask=None,
            # focus parameter
            score_tgt=None,
            foreground_pre_layer=None,
    ):
        # 在91个类别上的分数 * 前景分数 [bs, foreground s]
        mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
        # 在选出前300个 index
        select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
        # [bs,300,256]
        select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # 前300个query
        select_tgt = torch.gather(query, 1, select_tgt_index)
        # 同上
        select_pos = torch.gather(query_pos, 1, select_tgt_index)

        query_with_pos = key_with_pos = self.with_pos_embed(select_tgt, select_pos)
        # [bs,300,256]
        tgt2 = self.pre_attention(
            query_with_pos,
            key_with_pos,
            select_tgt,  # 上面两个加了post，tgt是没有加的，相当于value
        )[0]

        select_tgt = select_tgt + self.pre_dropout(tgt2)

        select_tgt = self.pre_norm(select_tgt)
        # 将结果放置在query中
        query = query.scatter(1, select_tgt_index, select_tgt)

        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )

        query = query + self.dropout1(src2)

        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class SalienceTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        # learnt background embed for prediction
        self.background_embedding = PositionEmbeddingLearned(200, num_pos_feats=self.embed_dim // 2)

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    # Deformable DETR 生成点位
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [n, s, l, 2]
        return reference_points

    def forward(
            self,
            query,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_pos=None,
            query_key_padding_mask=None,
            # salience input
            foreground_score=None,
            focus_token_nums=None,  # 每一个image的前景token数量（第一层encoder使用的数量就是这个值）
            foreground_inds=None,  # 每一层encoder使用的index
            multi_level_masks=None,
    ):
        # 生成grid点位
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points  # ori表示全部的all hw 以后下面每一层的reference_points都是从这里取出来的(是每一层用的)
        ori_pos = query_pos
        value = output = query
        for layer_id, layer in enumerate(self.layers):
            # [bs,s,256]
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dim)
            # 从all hw数量中取出要使用的前景token
            query = torch.gather(output, 1, inds_for_query)
            # 同上
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            # 同上
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            # ori_reference_points.view(b, n, -1) [bs,all hw,4,2] -> [bs,all hw,8]
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)  # 膨胀出一个维度重复八次，为了取出4个点位的2个坐标 一共八个
            ).view(b, -1, s, p)
            # 这个就是encoder的分类头 [bs,s,91]
            score_tgt = self.enhance_mcsp(query)

            # 参数与focus detr相同，次序变了
            query = layer(
                query,  # 使用前景采样过的
                query_pos,  # 使用前景采用过的
                value,  # all hw value不能采用，value要保证使用所有，将query减少，达到focus的意义
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
                # 每个token的前景各个类别上的分数
                score_tgt,
                # 前景分数
                foreground_pre_layer,
            )

            # 对query进行处理
            outputs = []
            # for bs 每个image 选出来的前景的数量是不同的
            for i in range(foreground_inds[layer_id].shape[0]):
                # 这里面的部分就是Focus中的代码
                # 多出来的下面这两行就是将Focus DETR中的代码变得更加清晰些

                # 前景分数
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                # 对应的query
                query_no_pad = query[i][:focus_token_nums[i]]
                # output [bs,all hw,256]
                # output在最开始是第一个query的值，这里会每次根据一层encoder的输出之后，将得到query（经过encoder的输出）替换到output中
                # 这样会是的每层的topk token都是在query的前几个位置，也都在output的前几个位置（一次覆盖一次）
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),  # (x,) -> (x,256)
                        query_no_pad,
                    )
                )
            output = torch.stack(outputs)
        # add learnt embedding for background
        if multi_level_masks is not None:
            # background_embedding 是可学习的位置编码
            # row_embed [200,128] col_embed [200,128] [bs,256,h,w] -> [bs,256,hw] -> [bs,hw,256]
            # 这里传入mask，仅仅使用其shape，并未使用内容
            background_embedding = [
                self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
            ]
            # [bs,s,256]
            background_embedding = torch.cat(background_embedding, dim=1)
            # 被选中为前景的，对应位置都设为0，因为不是background
            background_embedding.scatter_(1, inds_for_query, 0)
            # padding的部分也不是background
            background_embedding *= (~query_key_padding_mask).unsqueeze(-1)

            output = output + background_embedding

        return output


class SalienceTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            d_ffn=1024,
            n_heads=8,
            dropout=0.1,
            activation=nn.ReLU(inplace=True),
            n_levels=4,
            n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
            self,
            query,
            query_pos,
            reference_points,
            value,
            spatial_shapes,
            level_start_index,
            self_attn_mask=None,
            key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class SalienceTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        # dab中的做法
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        self.class_head = nn.ModuleList([nn.Linear(self.embed_dim, num_classes) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([MLP(self.embed_dim, self.embed_dim, 4, 3) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
            self,
            query,
            reference_points,
            value,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            key_padding_mask=None,
            attn_mask=None,
    ):
        outputs_classes = []
        outputs_coords = []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query)) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # 坐标修正
            # iterative bounding box refinement
            reference_points = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords
