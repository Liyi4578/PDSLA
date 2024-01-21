import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init, Scale
from mmcv.runner import force_fp32
from mmcv.ops import deform_conv2d
from mmdet.core import distance2bbox, multi_apply, bbox_overlaps, reduce_mean, filter_scores_and_topk, select_single_mlvl, bbox2distance
from mmdet.models import HEADS, AnchorFreeHead,build_loss
from mmdet.models.dense_heads.paa_head import levels_to_images 

from mmcv.cnn import ConvModule



EPS = 1e-12
INF = 1e8


class ScalePrior:
    def __init__(self) -> None:
        # 0 64 128 256 512 all
        self.reg_ranges = ((0,64), (64,128), (128,256),(256,512),(512,INF))
        # self.reg_scales = ((-INF,6), (6,7), (7,8),(8,9),(9,10))
        self.expand_reg_ranges = ((-1,128), (-1,256), (64,512),(128,INF),(256,INF))
        # 6 7 8 9 10
        # self.reg_ranges = ((0+64)/2.0, (64+128)/2.0, (128+256)/2.0,(256+512)/2.0,(512+1024)/2.0)
    
    def __call__(self,x):
        '''
        x: [num_gts,]
        '''

        weights = []
        for l,reg_range in enumerate(self.expand_reg_ranges):
            mask = (x >= reg_range[0]) & (x <= reg_range[1])
            weights.append(mask.float())

        weights = torch.stack(weights,dim=0)
        # weights = weights / (weights.max(dim=0)[0] + EPS)
        weights.clamp_(min=0.2)
        # print(weights)
        return weights
    

class Prior(nn.Module):
    def __init__(self,strides=(8, 16, 32, 64, 128)):

        super(Prior, self).__init__()
        self.mean = 0. 
        self.sigma = 1.0 
        self.strides = strides
        self.scale_prior = ScalePrior()

    def forward(self, anchor_points_list, gt_bboxes,inside_gt_bbox_mask):
        num_gts = len(gt_bboxes)
        num_points = sum([len(item) for item in anchor_points_list])
        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points,num_gts)

        gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) 
        gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        gt_hw = (torch.stack((gt_w, gt_h), dim=1)).float()
        gt_hw = gt_hw[None]
        gt_size = torch.where(gt_h>gt_w,gt_h,gt_w)
        scale_prior = self.scale_prior(gt_size)
        # print('scale_prior',scale_prior)
        center_prior_list = []
        for idx,(slvl_points, stride) in enumerate(zip(anchor_points_list, self.strides)):
            # slvl_points: points from single level in FPN, has shape (h*w, 2)
            # single_level_points has shape (h*w, num_gt, 2)
            single_level_points = slvl_points[:, None, :].expand(
                (slvl_points.size(0), len(gt_bboxes), 2))
            gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
            gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None]

            # distance has shape (num_points, num_gt, 2)
            # distance = (((single_level_points - gt_center) / (torch.sqrt(gt_hw) / 2.0 * math.sqrt(float(stride))) - self.mean)**2) # / gt_hw  float(stride)
            # center_prior = torch.exp(-distance / (2 * self.sigma**2)).prod(dim=-1)

            distance = single_level_points - gt_center
            sigma =  gt_hw * float(stride) / 4.0
            center_prior = torch.exp(-distance**2 / (2*sigma)).prod(dim=-1)
            temp = torch.exp(-distance**2 / (2*sigma))
            

            # center_prior_list.append(center_prior)
            lv_prior = center_prior * scale_prior[idx]
            center_prior_list.append(lv_prior)
        
        center_prior_weights = torch.cat(center_prior_list, dim=0)
        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights



@HEADS.register_module()
class PDSLAHead(AnchorFreeHead):
    def __init__(self,
                 *args,
                 aux_reg = False,
                 prior_offset=0.5,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                    type='Normal',
                    layer='Conv2d',
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='conv_cls',
                        std=0.01,
                        bias_prob=0.01)),
                 **kwargs):

        self.aux_reg = aux_reg
        self.obj = False
        super().__init__(*args, 
                         conv_bias=True,
                         norm_cfg=norm_cfg,
                         init_cfg=init_cfg,
                         **kwargs)

        self.prior = Prior(self.strides)
        self.prior_generator.offset = prior_offset
        self.fg_per_gt_dict = {}


    def init_weights(self):
        super(PDSLAHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=4.0)
        normal_init(self.conv_objectness, std=0.01)
        if self.aux_reg:
            normal_init(self.conv_aux_reg, std=0.01)

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            if self.aux_reg and i  == self.stacked_convs - 1:
                groups = 4
            else:
                groups = 1
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    groups = groups,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        # implicit reg aux
        if self.aux_reg:
            self.conv_reg = nn.Conv2d(64,4,1, padding=0,groups=4)
        else:
            self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def _init_layers(self):
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.aux_reg:
            self.conv_aux_reg = nn.Conv2d(self.feat_channels, 64, 3, padding=1,groups=4)
            self.conv_objectness = nn.Conv2d(64, 1, 3, padding=1)
        else:
            self.conv_objectness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    
    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)
    
    def forward_single(self, x, scale, stride):

        cls_feat = x
        reg_feat = x

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        if self.aux_reg:
            reg_feat = self.conv_aux_reg(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        objectness = self.conv_objectness(reg_feat)
        
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = F.relu(bbox_pred)
        bbox_pred = bbox_pred * stride

        return cls_score, bbox_pred, objectness
    
    def _loss_single(self, cls_score, objectness, reg_loss, gt_labels,
                             prior_weights, inside_gt_bbox_mask,img_meta,featmap_sizes=None):
        '''
        per img
        '''
        num_gts = len(gt_labels)
        num_points = cls_score.shape[0]
        joint_conf = (cls_score * objectness)
        p_cls = joint_conf[:, gt_labels]

        ious = (1-reg_loss).clamp(min=0.0) 

        pred_score =  ious**0.8 * p_cls**0.2
        
        pred_score_max = pred_score.max(dim=0)[0]
        prior_weight = (1-pred_score_max).clamp(min=0.1)
        map2p = lambda prior,pred_score:  prior**prior_weight * pred_score**(1-prior_weight)  * torch.exp(5*pred_score)
        p_pos = map2p(prior=prior_weights,pred_score=pred_score)
        p_pos = p_pos.detach()
        
        # this line is not so important....
        # (0.5 * exp(0.5*5)) / (exp(5)) = 0.041
        
        p_pos[ious < ious.max(dim=0)[0] / 2.0] = 0.0

        norm_p_pos = p_pos/p_pos.max(dim=0)[0].clamp(min=EPS)
        
        

        
        # from utils import draw_points_on_img,draw_heatmap_on_img
        # # # all_level_points = self.prior_generator.grid_priors(featmap_sizes, joint_conf.dtype,
        # # #                                    joint_conf.device)
        # hm = norm_p_pos.max(dim=1)[0]
        # # # draw_points_on_img(hm,all_level_points,hws=featmap_sizes,img_path=img_meta['ori_filename'],name="scaled_pred_" + img_meta['ori_filename'])
        # draw_heatmap_on_img(hm,hws=featmap_sizes,img_path=img_meta['ori_filename'],
        #          output_name="pdsla_la_" + img_meta['ori_filename'],origin_size=img_meta['img_shape'])
        # hm = prior_weights.max(dim=1)[0]
        # draw_heatmap_on_img(hm,hws=featmap_sizes,img_path=img_meta['ori_filename'],
        #         output_name="scaled_prior_" + img_meta['ori_filename'],origin_size=img_meta['img_shape'])

        fun = lambda x: torch.sigmoid((x-0.5)*10)
        b1 = fun(torch.tensor(0.0))
        b2 = fun(torch.tensor(1.0))
        def w_map(x):
            return (fun(x)-b1) / (b2-b1)

        pos_weight = w_map(norm_p_pos)

        p_pos_cls = joint_conf.new_zeros(*joint_conf.shape)
        if num_gts > 0:
            with torch.no_grad():
                num_gt_cls,gt_cls_idx = gt_labels.unique(return_inverse=True)
                for idx in range(len(num_gt_cls)):
                    temp_value,temp_idx = norm_p_pos[:,gt_cls_idx==idx].max(dim=1)
                    p_pos_cls[:,num_gt_cls[idx]] = temp_value


        w_cls = torch.where(p_pos_cls > 0.05,p_pos_cls* 0.5,joint_conf**2.0).detach()
        loss_cls = w_cls * F.binary_cross_entropy(joint_conf,p_pos_cls, reduction='none')

        loss_cls = loss_cls.sum()
        pos_avg_factor = (norm_p_pos).sum()# norm_p_pos.new_tensor(num_gts)# torch.tensor(num_gts * 9.0)# norm_p_pos.sum()

        w_reg = pos_weight
        loss_bbox = w_reg.detach() * reg_loss  * 1.25
        loss_bbox = loss_bbox.sum()
        reg_avg_factor =  w_reg.sum()
        
        
        return loss_cls, loss_bbox, pos_avg_factor,reg_avg_factor

    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)

        all_num_gt = sum([len(gt_bbox) for gt_bbox in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        inside_gt_bbox_mask_list, bbox_targets_list = self.get_targets(
            all_level_points, gt_bboxes)

        prior_weight_list = []
        for gt_bboxe, inside_gt_bbox_mask in zip(gt_bboxes, inside_gt_bbox_mask_list):
            prior_weight_list.append(self.prior(all_level_points,gt_bboxe,inside_gt_bbox_mask))

        mlvl_points = torch.cat(all_level_points, dim=0)

        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)
        
        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)

        for bbox_pred, gt_bboxe, inside_gt_bbox_mask in zip(
                bbox_preds, bbox_targets_list, inside_gt_bbox_mask_list):
            temp_num_gt = gt_bboxe.size(1)
            expand_mlvl_points = mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2)
            gt_bboxe = gt_bboxe.reshape(-1, 4)
            expand_bbox_pred = bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 4).reshape(-1, 4)

            decoded_bbox_preds = distance2bbox(expand_mlvl_points,
                                               expand_bbox_pred)
            decoded_target_preds = distance2bbox(expand_mlvl_points, gt_bboxe)

            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds,
                weight=None,
                reduction_override='none')
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))

        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]

        mean_num_gt = reduce_mean(
            bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)


        loss_cls_pos_list, loc_loss_list,pos_avg_factor_list,reg_avg_factor_list = multi_apply(self._loss_single, cls_scores,
                                    objectnesses, reg_loss_list, gt_labels,prior_weight_list,
                                        inside_gt_bbox_mask_list,img_metas,featmap_sizes=featmap_sizes)
        pos_avg_factor = sum(item.data.sum()
                            for item in pos_avg_factor_list).float()
        pos_avg_factor = reduce_mean(pos_avg_factor).clamp_(min=1)
        reg_avg_factor = sum(item.data.sum()
                            for item in reg_avg_factor_list).float()
        reg_avg_factor = reduce_mean(reg_avg_factor).clamp_(min=1)

        self.fg_per_gt_dict['num_fg_per_gt'] = pos_avg_factor / mean_num_gt

        cls_loss = sum(loss_cls_pos_list) / pos_avg_factor
        loc_loss = sum(loc_loss_list) / reg_avg_factor

        loss = dict(loss_cls_pos=cls_loss, loss_loc=loc_loss)

        return loss

    def get_targets(self, points, gt_bboxes_list):
        concat_points = torch.cat(points, dim=0)
        inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(
            self._get_target_single, gt_bboxes_list, points=concat_points)
        return inside_gt_bbox_mask_list, bbox_targets_list

    def _get_target_single(self, gt_bboxes, points):
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None]
        ys = ys[:, None]
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts),
                                                         dtype=torch.bool)

        return inside_gt_bbox_mask, bbox_targets

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        assert len(cls_scores) == len(bbox_preds) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            score_factor_list = select_single_mlvl(score_factors, img_id)

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list
    
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors = []
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            score_factor = score_factor.permute(1, 2,0).reshape(-1).sigmoid()

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            scores = cls_score.sigmoid()
            results = filter_scores_and_topk(
                scores*score_factor[:,None], cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            _, labels, keep_idxs, filtered_results = results
            scores = scores[keep_idxs, labels]
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)



def test():
    import numpy as np
    import cv2
    from mmcv import Config, DictAction
    from mmdet.utils import replace_cfg_vals
    from mmdet.models import build_head

    config_file = "configs/pass.py"
    cfg = Config.fromfile(config_file)
    cfg = replace_cfg_vals(cfg)
    cfg.nms_pre=1000
    cfg.min_bbox_size=0
    cfg.score_thr=0.05
    cfg.max_per_img=100
    cfg.with_nms=True
    cfg.nms = dict(type='nms', iou_threshold=0.6)

    img_metas = [{'filename': 'D:/Liyi/Datasets/coco_dataset/coco/val2017/000000397133.jpg', 
                'ori_filename': '000000397133.jpg', 
                'ori_shape': (427, 640, 3), 
                'img_shape': (800, 1199, 3), 
                'pad_shape': (800, 1216, 3), 
                'scale_factor': [1.8734375, 1.8735363, 1.8734375, 1.8735363], 
                'flip': False, 'flip_direction': None, 
                'img_norm_cfg': {'mean': [0., 0., 0.], 'std': [1., 1., 1.], 'to_rgb': False}},
                {'filename': 'D:/Liyi/Datasets/coco_dataset/coco/val2017/000000397133.jpg', 
                'ori_filename': '000000397133.jpg', 
                'ori_shape': (427, 640, 3), 
                'img_shape': (800, 1199, 3), 
                'pad_shape': (800, 1216, 3), 
                'scale_factor': [1.8734375, 1.8735363, 1.8734375, 1.8735363], 
                'flip': False, 'flip_direction': None, 
                'img_norm_cfg': {'mean': [0., 0., 0.], 'std': [1., 1., 1.], 'to_rgb': False}}]
    
    cfg.model.bbox_head['train_cfg'] = cfg.model.train_cfg
    # print(cfg.model.bbox_head)
    my_head = PDSLAHead(80,256,strides=[8, 16, 32, 64, 128])# build_head(cfg.model.bbox_head)# ATSS1Head(80, 256)
    my_head.init_weights()
    print('num_base_priors:',my_head.num_base_priors)
    feats = [torch.randn(2, 256, 640//s, 640//s) for s in [8, 16, 32, 64, 128]]
    outs = my_head.forward(feats)
    # assert len(cls_scores) == len(my_head.scales)
    for idx in range(len(outs[0])):
        print(outs[0][idx].shape)
        print(outs[1][idx].shape)
        print(outs[2][idx].shape)
        print('-'*36)
    

    gt_bboxes =  [[[10, 10, 20, 20],
                   [20,200,100, 260],
                   [30,250,80, 310],
                   [80,100,120,150]],
                 [[110, 110, 120, 120],
                   [20,20,100, 40],
                   [40,50,140, 150],
                   [200,150,233,200]]]
    for idx,gt_bboxes_img in enumerate(gt_bboxes):
        canvas = np.zeros([640,640,3])
        for box in gt_bboxes_img:
            cv2.rectangle(canvas, (box[0]*2, box[1]*2), (box[2]*2, box[3]*2), (204, 204, 51), 1)
        cv2.imwrite(f'canvas_{idx}.png',canvas)

    gt_bboxes = [torch.tensor(gt_bbox).float()*2 for gt_bbox in gt_bboxes]

    gt_labels = [torch.tensor([1,2,3,4]) for _ in range(2)] # 
    losses = my_head.loss(*outs,gt_bboxes=gt_bboxes,gt_labels=gt_labels,
                img_metas=img_metas)
            
    print(losses)
    
    res = my_head.get_bboxes(*outs,img_metas=img_metas,cfg=cfg)
    print(len(res))


if __name__ == '__main__':
    test()