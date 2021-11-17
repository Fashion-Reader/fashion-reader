
from torch import nn


def get_loss(CFG, outputs, labels):
    labels = labels.T
    ce_loss = nn.CrossEntropyLoss().to(CFG.device)
    bce_loss = nn.BCEWithLogitsLoss().to(CFG.device)

    cate_loss = ce_loss(outputs['cate_logits'], labels[0].long())
    style_loss = ce_loss(outputs['style_logits'], labels[1].long())
    neckline_loss = ce_loss(outputs['neckline_logits'], labels[2].long())
    pattern_loss = ce_loss(outputs['pattern_logits'], labels[3].long())

    button_loss = bce_loss(outputs['button_logits'], labels[4].unsqueeze(1).float())
    lace_loss = bce_loss(outputs['lace_logits'], labels[5].unsqueeze(1).float())
    zipper_loss = bce_loss(outputs['zipper_logits'], labels[6].unsqueeze(1).float())
    pocket_loss = bce_loss(outputs['pocket_logits'], labels[7].unsqueeze(1).float())
    slit_loss = bce_loss(outputs['slit_logits'], labels[8].unsqueeze(1).float())
    buckle_loss = bce_loss(outputs['buckle_logits'], labels[9].unsqueeze(1).float())

    sub_group_loss = ce_loss(outputs['sub_group_logits'], labels[10].long())
    sub_category_loss = ce_loss(outputs['sub_cate_logits'], labels[11].long())
    sub_color_loss = ce_loss(outputs['sub_color_logits'], labels[12].long())

    total_loss = cate_loss + style_loss + neckline_loss + pattern_loss + button_loss + lace_loss + zipper_loss + pocket_loss + slit_loss + buckle_loss + sub_group_loss + sub_category_loss + sub_color_loss
    return total_loss
