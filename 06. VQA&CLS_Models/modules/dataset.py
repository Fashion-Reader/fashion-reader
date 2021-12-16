from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as A
import torch.utils.data as data
import transformers
import numpy as np
import pandas as pd
import cv2
import os
import sys
import torch

# style에 label 부여
style_labels = {
    "클래식": 0,
    "매니시": 1,
    "페미닌": 2,
    "히피": 3,
    "모던": 4,
    "컨트리": 5,
    "젠더리스": 6,
    "스포티": 7,
    "레트로": 8,
    "밀리터리": 9,
    "프레피": 10,
    "톰보이": 11,
    "로맨틱": 12,
    "웨스턴": 13,
    "소피스트케이티드": 14,
    "리조트": 15,
    "키치": 16,
    "키덜트": 17,
    "스트리트": 18,
    "섹시": 19,
    "오리엔탈": 20,
    "아방가르드": 21,
    "힙합": 22,
    "펑크": 23
}


class VqaDataset(data.Dataset):
    def __init__(self, data_dir, mode):

        self.data = pd.read_csv(self.data_dir+f'vqa_{mode}.csv')

        # Save answer list
        answer_list_path = os.path.join(self.result_dir, 'answers.csv')
        self.answer_list = self.data_df['answer'].unique().tolist()
        pd.DataFrame({'answer': self.answer_list}).to_csv(
            answer_list_path, index=False)

        # Set Tokenizer
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(
            'xlm-roberta-base')

        self.max_token = 30

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        extracted_feature_path = self.data['npy_path'][idx]

        # question을 불러온다
        question = self.data_df['question'][idx]
        answer = self.data_df['answer'][idx]

        # BERT기반의 Tokenizer로 질문을 tokenize한다.
        tokenized = self.tokenizer.encode_plus("".join(question),
                                               None,
                                               add_special_tokens=True,
                                               max_length=self.max_token,
                                               truncation=True,
                                               #  padding=True)
                                               pad_to_max_length=True)

        # BERT기반의 Tokenize한 질문의 결과를 변수에 저장
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']

        # detection model에서 뽑은 (5,1048) 차원의 array를 불러온다
        npy_feature = np.load(extracted_feature_path)
        tensor_feature = torch.FloatTensor(npy_feature)

        if answer not in self.answer_list:
            print(f"Unexpected Target Token! {answer}")
            sys.exit()
        answer_id = self.answer_list.index(answer)

        # 전처리가 끝난 질문, 응답, 이미지 데이터를 반환
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'answer': torch.tensor(answer_id, dtype=torch.long),
                'image': tensor_feature}


class CropedDataset(data.Dataset):

    def __init__(self, data_dir, mode):
        """
        CropedDataset을 initialize 합니다.
        """
        self.data_dir = data_dir
        self.transform = A.Compose([
            A.Resize(356, 356),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(
                0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ])

        # printing과 neckline의 정보를 가진 csv파일을 불러옵니다.
        self.data = pd.read_csv(self.data_dir+f'feature_{mode}.csv')

    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """
        image_path = self.data['image_path'][index]
        image = cv2.imread(self.data_dir+image_path)

        # BBOX 정보를 이용하여 image를 자릅니다
        x = self.data['x'][index]
        y = self.data['y'][index]
        w = self.data['w'][index]
        h = self.data['h'][index]
        image = image[y:y+h, x:x+w]

        # 레이블을 불러옵니다
        print_label = self.data['print_label'][index]
        neck_label = self.data['neck_label'][index]

        # Augmentation을 적용해줍니다
        image_transform = self.transform(image=image)['image']

        return image_transform, print_label, neck_label

    def __len__(self):
        return len(self.data)


class StyleDataset(data.Dataset):
    style_num_classes = 24
    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mode):
        """
        StyleDataset을 initialize 합니다.
        """
        self.data_dir = data_dir
        self.transform = A.Compose([
            A.Resize(356, 356),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(
                0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ])
        self.data = pd.read_csv(self.data_dir+f'style_{mode}.csv')

    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """

        # 해당 path에서 image를 불러옵니다.
        image_path = self.data['image_path'][index]
        image = cv2.imread(self.data_dir+image_path)

        # 이미지를 Augmentation 시킵니다.
        image_transform = self.transform(image=image)['image']

        # 레이블을 불러옵니다.
        style_label = style_labels[self.data['style_label'][index]]

        return image_transform, style_label

    def __len__(self):
        return len(self.data)


def get_datasets(cfg):
    # DATA parameter들을 불러옵니다
    TRAIN_TYPE = cfg['TRAIN']['type']
    DATA_DIR = cfg['DATA']['dir']
    NUM_WORKERS = cfg['DATALOADER']['num_workers']
    PIN_MEMORY = cfg['DATALOADER']['pin_memory']
    BATCH_SIZE = cfg['TRAIN']['batch_size']

    # Train process에 따라 dataset을 불러옵니다
    if TRAIN_TYPE == 'feature':
        ds_train = CropedDataset(data_dir=DATA_DIR, mode='train')
        ds_valid = CropedDataset(data_dir=DATA_DIR, mode='valid')
    elif TRAIN_TYPE == 'style':
        ds_train = StyleDataset(data_dir=DATA_DIR, mode='train')
        ds_valid = StyleDataset(data_dir=DATA_DIR, mode='valid')
    elif TRAIN_TYPE == 'VQA':
        ds_train = VqaDataset(data_dir=DATA_DIR, mode='train')
        ds_valid = VqaDataset(data_dir=DATA_DIR, mode='valid')
        # 데이터셋 생성

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=PIN_MEMORY,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )

    return dl_train, dl_valid
