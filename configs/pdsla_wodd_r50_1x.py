_base_ = './pdsla_r50_1x.py'

model = dict(
    bbox_head=dict(
        aux_reg = False
        )
    )
