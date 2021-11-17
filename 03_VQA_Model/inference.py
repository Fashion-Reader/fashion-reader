
import cv2
import torch
import numpy as np
import pandas as pd


from torch.nn import functional as F
from transformers import BertTokenizer

from models.FR_model import FRModel
from modules.utils import run_query, url_to_image
from preprocessing.get_df import df_path
from preprocessing.get_image_embed import load_config_and_model_weights, get_model, get_visual_embeds


class CFG:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg = load_config_and_model_weights(cfg_path)
    image_embed_model = get_model(cfg)

    save_model_path = './model/Epoch7_Accuracy0.7295.pt'
    main_model = torch.load(save_model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_cate(item_type):
    # 임블리 쇼핑몰 기준
    top = ["니트","블라우스","셔츠","후드/맨투맨","긴팔티","반팔티","슬리브리스","베이직","스트라이프","프린팅","이너웨어"]
    bot = ["일자핏","스키니","부츠컷","와이드","데님","슬랙스","점프수트","레깅스","반바지/치마바지","미니스커트","미디스커트","롱스커트"]
    outer = ["자켓","점퍼","코트","가디건"]
    onep = ["롱원피스","미니원피스","플라워,패턴","투피스,세트"]

    if item_type in top:
        cate = '상의'
    elif item_type in bot:
        cate = '하의'
    elif item_type in outer:
        cate = '아우터'
    else:
        cate = '원피스'
    return cate


def get_language_input(cate):
    ko2en = {'대분류':'group',
            '상의':'top',
            '아우터':'outerwear',
            '원피스':'one-piece',
            '하의':'bottom',
            '카테고리':'category',
            '스타일':'style',
            '넥라인':'neckline',
            '무늬':'pattern',
            '단추':'button',
            '레이스':'lace',
            '지퍼':'zipper',
            '포켓':'pocket',
            '슬릿':'slit',
            '버클':'buckle',
            '서브 대분류':'sub group',
            '서브 카테고리':'sub category',
            '서브 색상':'sub color'}

    tmp = ' [MASK] '.join([en for en in ko2en.values() if not en in ['group','top','outerwear','one-piece','bottom']])+' [MASK]'
    language_input = 'group '+ ko2en[cate]+' '+tmp
    return language_input


def get_tokens(CFG, language_input):
    tokens = CFG.tokenizer([language_input], padding='max_length', max_length=50)
    input_ids = torch.tensor(tokens["input_ids"])
    attention_mask = torch.tensor(tokens["attention_mask"])
    token_type_ids = torch.tensor(tokens["token_type_ids"])
    
    # [MASK] = 103 token
    mask_pos_ids = torch.tensor([1 if token == 103 else 0 for token in input_ids.squeeze()]).unsqueeze(dim=0)
    return input_ids, attention_mask, token_type_ids, mask_pos_ids


def get_visual_embeds(CFG, img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    visual_embeds = get_visual_embeds(CFG.cfg, CFG.image_embed_model, img_bgr, CFG.device)
    visual_embeds = visual_embeds.cpu().detach().numpy()
    visual_embeds = torch.tensor([visual_embeds])
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    return visual_embeds, visual_attention_mask, visual_token_type_ids


def get_label2value():
    df = pd.read_csv(df_path)
    label2cate = {v:k for k,v in dict(df[['카테고리', '카테고리_ID']].values).items()}
    label2style = {v:k for k,v in dict(df[['스타일', '스타일_ID']].values).items()}
    label2neck = {v:k for k,v in dict(df[['넥라인', '넥라인_ID']].values).items()}
    label2pattern = {v:k for k,v in dict(df[['무늬', '무늬_ID']].values).items()}
    label2subgroup = {v:k for k,v in dict(df[['서브 대분류', '서브 대분류_ID']].values).items()}
    label2subcate = {v:k for k,v in dict(df[['서브 카테고리', '서브 카테고리_ID']].values).items()}
    label2subcolor = {v:k for k,v in dict(df[['서브 색상', '서브 색상_ID']].values).items()}
    return label2cate, label2style, label2neck, label2pattern, label2subgroup, label2subcate, label2subcolor


def get_detail(outputs):
    button_pred = F.sigmoid(outputs['button_logits']).detach().cpu().numpy()
    lace_pred = F.sigmoid(outputs['lace_logits']).detach().cpu().numpy()
    zipper_pred = F.sigmoid(outputs['zipper_logits']).detach().cpu().numpy()
    pocket_pred = F.sigmoid(outputs['pocket_logits']).detach().cpu().numpy()
    slit_pred = F.sigmoid(outputs['slit_logits']).detach().cpu().numpy()
    buckle_pred = F.sigmoid(outputs['buckle_logits']).detach().cpu().numpy()

    button_ans = np.where(button_pred >= 0.5, 1, 0)
    lace_ans = np.where(lace_pred >= 0.5, 1, 0)
    zipper_ans = np.where(zipper_pred >= 0.5, 1, 0)
    pocket_ans = np.where(pocket_pred >= 0.5, 1, 0)
    slit_ans = np.where(slit_pred >= 0.5, 1, 0)
    buckle_ans = np.where(buckle_pred >= 0.5, 1, 0)
    
    det2ans = {'단추':button_ans,'레이스':lace_ans,'지퍼':zipper_ans,'포켓':pocket_ans,'슬릿':slit_ans,'버클':buckle_ans}
    return ','.join([k for k,v in det2ans.items() if v == 1])

def inference(CFG, inputs):
    inputs = tuple(inp.to(CFG.device) for inp in inputs)
    outputs = CFG.main_model(inputs)

    label2value = get_label2value()
    cate_ans = label2value[0][int(outputs['cate_logits'].argmax().cpu().detach().numpy())]
    style_ans = label2value[1][int(outputs['style_logits'].argmax().cpu().detach().numpy())]
    neckline_ans = label2value[2][int(outputs['neckline_logits'].argmax().cpu().detach().numpy())]
    pattern_ans = label2value[3][int(outputs['pattern_logits'].argmax().cpu().detach().numpy())]
    
    detail_ans = get_detail(outputs)
    
    subgroup_ans = label2value[4][int(outputs['sub_group_logits'].argmax().cpu().detach().numpy())]
    subcate_ans = label2value[5][int(outputs['sub_cate_logits'].argmax().cpu().detach().numpy())]
    subcolor_ans = label2value[6][int(outputs['sub_color_logits'].argmax().cpu().detach().numpy())]

    return {"카테고리": cate_ans,
            "스타일": style_ans,
            "넥라인": neckline_ans,
            "무늬": pattern_ans,
            "디테일": detail_ans,
            "서브 대분류": subgroup_ans,
            "서브 카테고리": subcate_ans,
            "서브 색상": subcolor_ans,
            }


def get_results(CFG, item_type, item_img_links):
    cate = get_cate(item_type)
    language_input = get_language_input(cate)
    input_ids, attention_mask, token_type_ids, mask_pos_ids = get_tokens(CFG, language_input)

    img = url_to_image(item_img_links)
    visual_embeds, visual_attention_mask, visual_token_type_ids = get_visual_embeds(CFG, img)

    inputs = (input_ids, attention_mask, token_type_ids, visual_embeds, visual_attention_mask, visual_token_type_ids, mask_pos_ids)
    results = inference(CFG, inputs)
    return results


if __name__ == '__main__':
    query = """
    select * 
    from products
    limit 10
    """
    df = run_query(query)

    row = df.iloc[0]
    results = get_results(CFG, row['item_type'], row['item_img_links'])
