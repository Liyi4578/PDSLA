# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import DETECTORS,SingleStageDetector


@DETECTORS.register_module()
class MyDetector(SingleStageDetector):
    
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MyDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        if hasattr(self.bbox_head, 'epoch'):
            self.bbox_head.epoch = epoch
        
    def set_iter(self,cur_iter):
        # used to debug
        if hasattr(self.bbox_head, 'cur_iter'):
            self.bbox_head.cur_iter = cur_iter
    
    def get_fg_per_gt(self):
        if hasattr(self.bbox_head, 'fg_per_gt_dict'):
            return self.bbox_head.fg_per_gt_dict
        else:    
            return None