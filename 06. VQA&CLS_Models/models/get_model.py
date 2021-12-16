""" Model list
    * VQA model
    * cls models
    * Detection model
"""

from models.vqa_model import VQAModel
from models.efficient_cls import StyleClassificationModel
from models.efficient_cls import CropedClassificationModel
from mmdet.apis import init_detector


def get_model(model_str: str, cfg) -> 'model':
    if model_str == 'vqa_model':
        return VQAModel(cfg['MODEL']['num_targets'])
    elif model_str == 'feature_model':
        return CropedClassificationModel()
    elif model_str == 'style_model':
        return StyleClassificationModel()
    elif model_str == 'detection':
        config = './configs/swin/mask_rcnn_swin_small_inference.py'
        checkpoint = './work_dirs/swin+cascade_kfashion1/best_bbox_mAP_50.pth'
        return init_detector(config, checkpoint)


if __name__ == '__main__':
    pass
