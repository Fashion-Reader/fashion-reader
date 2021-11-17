
import torch
import numpy as np


from torch.nn import functional as F
from sklearn.metrics import accuracy_score


def get_accuracy_all(CFG, outputs, labels):
    labels = labels.T.detach().cpu().numpy()

    cate_pred = torch.argmax(outputs['cate_logits'], dim=1).detach().cpu().numpy()
    style_pred = torch.argmax(outputs['style_logits'], dim=1).detach().cpu().numpy()
    neckline_pred = torch.argmax(outputs['neckline_logits'], dim=1).detach().cpu().numpy()
    pattern_pred = torch.argmax(outputs['pattern_logits'], dim=1).detach().cpu().numpy()

    button_pred = F.sigmoid(outputs['button_logits']).detach().cpu().numpy()
    lace_pred = F.sigmoid(outputs['lace_logits']).detach().cpu().numpy()
    zipper_pred = F.sigmoid(outputs['zipper_logits']).detach().cpu().numpy()
    pocket_pred = F.sigmoid(outputs['pocket_logits']).detach().cpu().numpy()
    slit_pred = F.sigmoid(outputs['slit_logits']).detach().cpu().numpy()
    buckle_pred = F.sigmoid(outputs['buckle_logits']).detach().cpu().numpy()

    sub_group_pred = torch.argmax(outputs['sub_group_logits'], dim=1).detach().cpu().numpy()
    sub_category_pred = torch.argmax(outputs['sub_cate_logits'], dim=1).detach().cpu().numpy()
    sub_color_pred = torch.argmax(outputs['sub_color_logits'], dim=1).detach().cpu().numpy()

    cate_accuracy = accuracy_score(labels[0], cate_pred)
    style_accuracy = accuracy_score(labels[1], style_pred)
    neckline_accuracy = accuracy_score(labels[2], neckline_pred)
    pattern_accuracy = accuracy_score(labels[3], pattern_pred)
    
    button_accuracy = accuracy_score(labels[4], np.where(button_pred >= 0.5, 1, 0))
    lace_accuracy = accuracy_score(labels[5], np.where(lace_pred >= 0.5, 1, 0))
    zipper_accuracy = accuracy_score(labels[6], np.where(zipper_pred >= 0.5, 1, 0))
    pocket_accuracy = accuracy_score(labels[7], np.where(pocket_pred >= 0.5, 1, 0))
    slit_accuracy = accuracy_score(labels[8], np.where(slit_pred >= 0.5, 1, 0))
    buckle_accuracy = accuracy_score(labels[9], np.where(buckle_pred >= 0.5, 1, 0))
    
    sub_group_accuracy = accuracy_score(labels[10], sub_group_pred)
    sub_category_accuracy = accuracy_score(labels[11], sub_category_pred)
    sub_color_accuracy = accuracy_score(labels[12], sub_color_pred)
    
    accuracy_all = {"cate_accuracy": cate_accuracy,
                    "style_accuracy": style_accuracy,
                    "neckline_accuracy": neckline_accuracy,
                    "pattern_accuracy": pattern_accuracy,
                    "button_accuracy": button_accuracy,
                    "lace_accuracy": lace_accuracy,
                    "zipper_accuracy": zipper_accuracy,
                    "pocket_accuracy": pocket_accuracy,
                    "slit_accuracy": slit_accuracy,
                    "buckle_accuracy": buckle_accuracy,
                    "sub_group_accuracy": sub_group_accuracy,
                    "sub_category_accuracy": sub_category_accuracy,
                    "sub_color_accuracy": sub_color_accuracy,
                    }
    return accuracy_all
