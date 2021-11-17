"""
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
%%capture
!pip install pyyaml==5.1
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
!pip install transformers
"""

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tqdm.auto import tqdm
from detectron2 import model_zoo
from detectron2.layers import nms
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.structures.boxes import Boxes
from detectron2.structures.image_list import ImageList
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from get_df import df_path


def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    cfg['MODEL']['DEVICE']='cuda'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
    return cfg


def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model


def prepare_image_inputs(cfg, img_list, model, device):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1)).to(device)

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1).to(device)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1).to(device)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images, model.backbone.size_divisibility)
    return images, batched_inputs


def get_features(model, images):
    features = model.backbone(images.tensor)
    return features


def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals


def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)
    
    try:
        box_features = box_features.reshape(1, 1000, 1024) # depends on your config and batch size
    except:
        box_features = box_features.unsqueeze(0)
    return box_features, features_list


def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas


def get_box_scores(cfg, proposals, pred_class_logits, pred_proposal_deltas):
    input_shape = pred_proposal_deltas.size()[-1]
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    num_classes = pred_class_logits.size()[-1]
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
    
    outputs = FastRCNNOutputLayers(
        input_shape=input_shape,
        box2box_transform=box2box_transform,
        num_classes=num_classes,
        smooth_l1_beta=smooth_l1_beta,
    )
    
    boxes = outputs.predict_boxes((pred_class_logits, pred_proposal_deltas), proposals)
    scores = outputs.predict_probs((pred_class_logits, pred_proposal_deltas), proposals)
    return boxes, scores


def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)
    return output_boxes


def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.cpu().detach()
    
    try:
        cls_boxes = output_boxes.tensor.cpu().detach().reshape(1000,80,4)
    except:
        n, _ = output_boxes.tensor.cpu().size()
        cls_boxes = output_boxes.tensor.cpu().detach().reshape(n//80,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf


def filter_boxes(keep_boxes, max_conf, min_boxes=10, max_boxes=100):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes


def get_visual_embeds(cfg, model, image, device):
    images, batched_inputs = prepare_image_inputs(cfg, [image], model, device)
    features = get_features(model, images)
    proposals = get_proposals(model, images, features)
    box_features, features_list = get_box_features(model, features, proposals)
    pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)
    boxes, scores = get_box_scores(cfg, proposals, pred_class_logits, pred_proposal_deltas)
    output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]

    temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
    keep_boxes, max_conf = [],[]
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)

    keep_boxes = [filter_boxes(keep_box, mx_conf) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
    visual_embeds = [box_feature[keep_box.copy()] for box_feature, keep_box in zip(box_features, keep_boxes)]
    return torch.stack(visual_embeds)


def img2embeds(cfg, model, path, device):
    img = plt.imread(path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    visual_embeds = get_visual_embeds(cfg, model, img_bgr, device)
    return visual_embeds


def main(df_path):
    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg = load_config_and_model_weights(cfg_path)
    model = get_model(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('df_path')

    image_embed_paths = []
    for path in tqdm(df['image_path'].values):
        npy_path = '/'.join(['/'.join(path.split('/')[:-2]), 'embed_data', path.split('/')[-1][:-3]+'npy'])
        image_embed_paths.append(npy_path)

        visual_embeds = img2embeds(cfg, model, path, device)[0]
        visual_embeds = visual_embeds.cpu().detach().numpy()
        np.save(npy_path, visual_embeds)

    df['image_embed_path'] = image_embed_paths
    df.to_csv(df_path, index=False)


if __name__ == "__main__":
    main(df_path)
