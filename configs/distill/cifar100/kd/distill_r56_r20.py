_base_ = '../../base.py'
# model settings
model = dict(
    type='Distill',
    pretrained=False,
    teacher_model_url='teacher.pth',
    backbone=dict(
        type='ResNetv1',
        depth=20,
        num_filters=[16, 16, 32, 64],
        block_name='basicblock',
        num_classes=100),
    teacher_model=dict(
        type='ResNetv1',
        depth=56,
        num_filters=[16, 16, 32, 64],
        block_name='basicblock',
        num_classes=100),
    head=dict(
        type='ClsHead',
        loss_config=dict(type='KLDivergence', T=4.0),
        use_num_classes=False,
        with_fc=False))

data_source_cfg = dict(type='ClsSourceCifar100', root='data/cifar/')
dataset_type = 'ClsDataset'
img_norm_cfg = dict(
    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
data = dict(
    imgs_per_gpu=64,  # total 64
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]
custom_hooks = []
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='step', step=[150, 180, 210])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 240
# log setting
log_config = dict(interval=100)
# export config
export = dict(export_neck=True)

find_unused_parameters = True
