# model settings
model = dict(
    type='Classification',
    train_preprocess=['mixUp'],
    pretrained=False,
    mixup_cfg=dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=1000),
    backbone=dict(
        type='VisionTransformer',
        img_size=[224],
        num_classes=1000,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.1),
    head=dict(
        type='ClsHead',
        loss_config={
            'type': 'SoftTargetCrossEntropy',
        },
        with_fc=False))
