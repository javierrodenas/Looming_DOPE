dataset_type = 'CocoDataset'
data_root = '/media/HDD_4TB_1/Datasets/AICrowd_newval'
categories = ['water', 'pear', 'egg', 'grapes', 'butter', 'jam', 'apple', 'mixed-salad-chopped-without-sauce', 'cheese', 'tomato-sauce', 'carrot', 'onion', 'beef-cut-into-stripes-only-meat', 'rice-noodles-vermicelli', 'banana', 'mixed-vegetables', 'savoury-puff-pastry', 'wine-white', 'dried-meat', 'fresh-cheese', 'red-radish', 'bread-fruit', 'oil-vinegar-salad-dressing', 'tomato', 'cauliflower', 'potato-gnocchi', 'wine-red', 'pasta-linguini-parpadelle-tagliatelle', 'french-beans', 'almonds', 'dark-chocolate', 'mandarine', 'croissant', 'sushi', 'berries', 'biscuits', 'thickened-cream-35', 'corn', 'celeriac', 'alfa-sprouts', 'chickpeas', 'leaf-spinach', 'rice', 'chocolate-cookies', 'pineapple', 'tart', 'coffee-with-caffeine', 'focaccia', 'soup-vegetable', 'potatoes-steamed', 'spaetzle', 'frying-sausage', 'lasagne-meat-prepared', 'boisson-au-glucose-50g', 'ma1-4esli', 'peanut-butter', 'chips-french-fries', 'veggie-burger', 'country-fries', 'yaourt-yahourt-yogourt-ou-yoghourt-natural', 'hummus', 'fish', 'beer', 'peanut', 'pizza-margherita-baked', 'pickle', 'ham-cooked', 'cake-chocolate', 'soup-of-lentils-dahl-dhal', 'pumpkin', 'witloof-chicory', 'vegetable-au-gratin-baked', 'tea-peppermint', 'soup-pumpkin', 'quiche-with-cheese-baked-with-puff-pastry', 'mango', 'cucumber', 'pasta', 'beef-filet', 'walnut', 'soft-cheese', 'sweet-pepper', 'sauce-soya', 'green-olives', 'roll-of-half-white-or-white-flour-with-large-void', 'cappuccino', 'flakes-oat', 'mayonnaise', 'chicken', 'orange', 'goat-cheese-soft', 'tuna', 'tomme', 'apple-pie', 'rosti', 'broccoli', 'beans-kidney', 'white-cabbage', 'ketchup', 'salt-cake-vegetables-filled', 'pistachio', 'feta', 'salmon', 'avocado', 'sauce-pesto', 'salad-rocket', 'peas', 'bacon-frying', 'salami', 'lentils', 'mushrooms', 'mashed-potatoes-prepared-with-full-fat-milk-with-butter', 'fennel', 'chocolate-mousse', 'corn-crisps', 'sweet-potato', 'bircherma1-4esli-prepared-no-sugar-added', 'beetroot-steamed-without-addition-of-salt', 'sauce-savoury', 'leek', 'milk', 'tea', 'fruit-salad', 'bread-rye', 'salad-lambs-ear', 'red-cabbage', 'praline', 'black-olives', 'mozzarella', 'bacon-cooking', 'pomegranate', 'hamburger-bread-meat-ketchup', 'curry-vegetarian', 'honey', 'juice-orange', 'cookies', 'mixed-nuts', 'breadcrumbs-unspiced', 'beef-sirloin-steak', 'shrimp-prawn-large', 'sour-cream', 'greek-salad', 'sauce-roast', 'zucchini', 'greek-yaourt-yahourt-yogourt-ou-yoghourt', 'cashew-nut', 'meat-terrine-pata-c', 'chicken-cut-into-stripes-only-meat', 'couscous', 'craape-plain', 'tofu', 'water-mineral', 'ham-croissant', 'juice-apple', 'falafel-balls', 'egg-scrambled-prepared', 'brioche', 'bread-pita', 'blue-mould-cheese', 'quinoa', 'crisps', 'beef', 'butter-spread-puree-almond', 'beef-minced-only-meat', 'hazelnut-chocolate-spread-nutella-ovomaltine-caotina', 'chocolate', 'nectarine', 'ice-tea', 'applesauce-unsweetened-canned', 'syrup-diluted-ready-to-drink', 'sugar-melon', 'rusk-wholemeal', 'pancakes', 'milk-chocolate', 'pork', 'dairy-ice-cream', 'guacamole', 'sausage', 'fruit-coulis', 'water-with-lemon-juice', 'brownie', 'lemon', 'veal-sausage', 'dates', 'roll-with-pieces-of-chocolate', 'croissant-with-chocolate-filling', 'sesame-seeds', 'cottage-cheese', 'cream-cheese', 'tea-verveine', 'tiramisu', 'grits-polenta-maize-flour', 'pasta-noodles', 'artichoke', 'mixed-seeds', 'omelette-plain', 'hazelnut', 'kiwi', 'dried-raisins', 'kolhrabi', 'plums', 'beetroot-raw', 'cream', 'fajita-bread-only', 'apricots', 'kefir-drink', 'bread', 'wine-rosa-c', 'watermelon-fresh', 'green-asparagus', 'white-asparagus', 'peach']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Translate',
            'prob': 0.5,
            'level': 6
        }, {
            'type': 'Shear',
            'prob': 0.5,
            'level': 6
        }, {
            'type': 'ContrastTransform',
            'prob': 0.3,
            'level': 4
        }, {
            'type': 'ColorTransform',
            'prob': 0.5,
            'level': 7
        }],
                  [{
                      'type': 'Rotate',
                      'prob': 0.5,
                      'level': 6
                  }, {
                      'type': 'Translate',
                      'prob': 0.5,
                      'level': 6
                  }, {
                      'type': 'EqualizeTransform',
                      'prob': 0.3
                  }, {
                      'type': 'BrightnessTransform',
                      'prob': 0.3,
                      'level': 6
                  }]]),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.01,
        dataset=dict(
            type='CocoDataset',
            classes=categories,
            ann_file=
            '/media/HDD_4TB/javi/FoodChallenge/new_dataset/train/annotations_grouped_th20-60_212.json',
            img_prefix='/media/HDD_4TB/javi/FoodChallenge/new_dataset/train/images',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='AutoAugment',
                    policies=[[{
                        'type': 'Translate',
                        'prob': 0.5,
                        'level': 6
                    }, {
                        'type': 'Shear',
                        'prob': 0.5,
                        'level': 6
                    }, {
                        'type': 'ContrastTransform',
                        'prob': 0.3,
                        'level': 4
                    }, {
                        'type': 'ColorTransform',
                        'prob': 0.5,
                        'level': 7
                    }],
                              [{
                                  'type': 'Rotate',
                                  'prob': 0.5,
                                  'level': 6
                              }, {
                                  'type': 'Translate',
                                  'prob': 0.5,
                                  'level': 6
                              }, {
                                  'type': 'EqualizeTransform',
                                  'prob': 0.3
                              }, {
                                  'type': 'BrightnessTransform',
                                  'prob': 0.3,
                                  'level': 6
                              }]]),
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Translate',
                    'prob': 0.5,
                    'level': 6
                }, {
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 6
                }, {
                    'type': 'ContrastTransform',
                    'prob': 0.3,
                    'level': 4
                }, {
                    'type': 'ColorTransform',
                    'prob': 0.5,
                    'level': 7
                }],
                          [{
                              'type': 'Rotate',
                              'prob': 0.5,
                              'level': 6
                          }, {
                              'type': 'Translate',
                              'prob': 0.5,
                              'level': 6
                          }, {
                              'type': 'EqualizeTransform',
                              'prob': 0.3
                          }, {
                              'type': 'BrightnessTransform',
                              'prob': 0.3,
                              'level': 6
                          }]]),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=categories,
        ann_file=
        '/media/HDD_4TB/javi/FoodChallenge/new_dataset/val/annotations_grouped_th20-60_212.json',
        img_prefix='/media/HDD_4TB/javi/FoodChallenge/new_dataset/val/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=categories,
        ann_file=
        'media/HDD_4TB/javi/FoodChallenge/new_dataset/val/annotations_grouped_th20-60_212.json',
        img_prefix='media/HDD_4TB/javi/FoodChallenge/new_dataset/val/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    periods=[6, 6, 6, 6, 10, 10],
    restart_weights=[1, 0.8, 0.6, 0.4, 0.3, 0.2],
    min_lr=1e-05)
total_epochs = 100
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1000,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
model = dict(
    type='SCNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='SCNetRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=212,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=212,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=212,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='SCNetMaskHead',
            num_convs=12,
            in_channels=256,
            conv_out_channels=256,
            num_classes=212,
            conv_to_res=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        glbctx_head=dict(
            type='GlobalContextHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=212,
            loss_weight=3.0,
            conv_to_res=True),
        feat_relay_head=dict(
            type='FeatureRelayHead',
            in_channels=1024,
            out_conv_channels=256,
            roi_feat_size=7,
            scale_factor=2)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
work_dir = 'scnet_50_grouping_thr20-60_212_test'
gpu_ids = range(0, 1)
