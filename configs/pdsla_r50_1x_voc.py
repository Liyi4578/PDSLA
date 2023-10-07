_base_ = [
    '_base_/datasets/voc0712.py',
    '_base_/default_runtime.py'
]

custom_imports = dict(
	imports=['pdsla_head','my_detector','vx_hook'],
	allow_failed_imports=False
)

model = dict(
    type='MyDetector',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='PDSLAHead',
        aux_reg = True,
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    train_cfg = None,
    test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100,
            with_nms=True)
    )
    
        
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)

optimizer = dict(
    type='SGD', lr=0.01, paramwise_cfg=dict(norm_decay_mult=0.), momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(
    policy='step', 
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.001,
    step=[3])
# lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12


# custom hooks
custom_hooks = [dict(type='SetVXInfoHook',priority=89)] # 