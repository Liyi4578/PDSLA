_base_ = [
	'./pdsla_r50_1x.py'
]
model = dict(
	backbone=dict(
		type='ResNeXt',
		depth=101,
		groups=64,
		base_width=4,
		num_stages=4,
		out_indices=(0, 1, 2, 3),
		frozen_stages=1,
		norm_cfg=dict(type='BN', requires_grad=True),
		norm_eval=True,
		style='pytorch',
		init_cfg=dict(
			type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'
		)
	),
	bbox_head=dict(
		aux_reg=True
	)
)



# need set it to 4 in pdsla_r50_1x.py
# data = dict(
#	  samples_per_gpu=4,
#	  workers_per_gpu=4,
# )

		
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(
		type='Resize',
		img_scale=[(1333, 480), (1333, 960)],
		multiscale_mode='range',
		keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(1333, 800),
		flip=False,
		transforms=[
			dict(type='Resize', 
				img_scale=[(1333, 480), (1333, 640), (1333, 800), (1333, 960),
						   (1333, 1120), (1333, 1280)],
				multiscale_mode='value',
				keep_ratio=True),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			# dict(type='ImageToTensor', keys=['img']),
			dict(type='DefaultFormatBundle'),
			dict(type='Collect', keys=['img']),
		])
]


data = dict(train=dict(pipeline=train_pipeline),test=dict(pipeline=test_pipeline))