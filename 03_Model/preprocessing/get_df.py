
import os
import json
import glob
import pandas as pd

from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from kfashion_image_rename import save_img_path


def load_json(file_name):
    with open(file_name) as json_file:
        json_data = json.load(json_file)
    return json_data

def load_dataset_info(label_dir_path):
    label_paths = glob.glob(os.path.join(label_dir_path,'*','*'))

    lst = []
    for label_path in tqdm(label_paths, total=len(label_paths)):
        data = load_json(label_path)
        file_dir = label_path.split('/')[-2]
        file_name = data['이미지 정보']['이미지 파일명']
        
        items = []
        for keyword in ['아우터', '하의', '원피스', '상의']:
            item = data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][keyword][0]
            if item.keys():
                dic = {'file_dir':file_dir, 'file_name':file_name, '대분류':keyword}
                for k, v in item.items():
                    if type(v) == str:
                        dic[f'{k}'] = v
                    else:
                        dic[f'{k}'] = ", ".join(v)
                items.append(dic)
        
        style = data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['스타일'][0]

        def f(dic):
            for k, v in style.items():
                if type(v) == str:
                    dic['스타일'] = v
                else:
                    dic['스타일'] = ", ".join(v)
            return dic
        
        items = list(map(f, items))
        lst.extend(items)

    df = pd.DataFrame(lst)
    return df

def preprocessing(df, save_img_path):
    _df = df[['file_dir', 'file_name', '대분류', '카테고리', '색상', '스타일', '넥라인', '프린트', '디테일']]
    _df.columns = ['file_dir', 'file_name', '대분류', '카테고리', '색상', '스타일', '넥라인', '무늬', '디테일']
    _df = _df[_df['카테고리'].isnull() == False]
    _df = _df.fillna('없음')

    tmp = _df.groupby(['file_dir', 'file_name']).apply(lambda x:list(zip(x['대분류'], x['카테고리'], x['색상']))).reset_index()
    tmp.columns = ['file_dir', 'file_name', 'tmp']

    _df = pd.merge(_df, tmp, how='left', on=['file_dir', 'file_name'])
    _df['_tmp'] =  list(map(lambda x:set([tuple(x)]), list(zip(_df['대분류'], _df['카테고리'], _df['색상']))))
    _df['tmp'] = _df['tmp'].apply(set)
    _df['tmp'] = _df['tmp'] - _df['_tmp']

    _df['sub_item_num'] = _df['tmp'].apply(len)
    _df = _df[_df['sub_item_num'] <= 1]
    _df['tmp'] = _df['tmp'].apply(list)

    _df['서브 대분류'] = _df['tmp'].apply(lambda x:list(x[0])[0] if x else None)
    _df['서브 카테고리'] = _df['tmp'].apply(lambda x:list(x[0])[1] if x else None)
    _df['서브 색상'] = _df['tmp'].apply(lambda x:list(x[0])[2] if x else None)\

    _df = _df[['file_dir', 'file_name', '대분류', '카테고리', '색상', '스타일', '넥라인', '무늬', '디테일', '서브 대분류', '서브 카테고리', '서브 색상']]
    _df['image_path'] = save_img_path + _df['file_dir'] + '/' + _df['file_name'].apply(lambda x:x.lower())
    _df = _df.fillna('없음')
    return _df

def label_encoding(_df):
    groups = ['상의','아우터','원피스','하의']
    categories = ['니트웨어', '브라탑', '블라우스', '셔츠', '탑', '티셔츠', '후드티', '가디건', '베스트', '재킷', '점퍼', '짚업', '코트', '패딩', '드레스', '점프수트', '래깅스', '스커트', '조거팬츠', '청바지', '팬츠']
    colors = ['와인', '실버', '화이트', '네이비', '옐로우', '카키', '그레이', '브라운', '베이지', '스카이블루', '레드', '블루', '퍼플', '오렌지', '블랙', '핑크', '그린', '라벤더', '네온', '민트', '골드']
    styles = ['페미닌', '리조트', '아방가르드', '소피스트케이티드', '키치', '펑크', '컨트리', '레트로', '클래식', '모던', '힙합', '젠더리스', '스트리트', '밀리터리', '오리엔탈', '로맨틱', '히피', '웨스턴', '섹시', '프레피', '매니시', '톰보이', '스포티']
    neckline = ['노카라', '라운드넥', '보트넥', '브이넥', '스위트하트', '스퀘어넥', '오프숄더', '원숄더', '유넥', '터틀넥', '홀터넥', '후드']
    printing = ['레터링', '무지', '플로럴', '스트라이프', '그래픽',  '믹스', '페이즐리', '호피', '체크', '도트', '아가일', '깅엄',  '카무플라쥬', '♥', '그라데이션', '타이다이']
    detail = ['단추','레이스','지퍼','포켓','슬릿','버클']

    columns = ['카테고리', '스타일', '넥라인', '무늬', '디테일', '서브 대분류', '서브 카테고리', '서브 색상']
    for col in columns:
        if col in ['대분류', '서브 대분류']:
            _df[col] = _df[col].apply(lambda x:x if x in groups else '없음')
        elif col in ['카테고리', '서브 카테고리']:
            _df[col] = _df[col].apply(lambda x:x if x in categories else '없음')
        elif col in ['서브 색상']:
            _df[col] = _df[col].apply(lambda x:x if x in colors else '없음')
        elif col in ['스타일']:
            _df[col] = _df[col].apply(lambda x:x if x in styles else '없음')
        elif col in ['넥라인']:
            _df[col] = _df[col].apply(lambda x:x if x in neckline else '없음')
        elif col in ['무늬']:
            _df[col] = _df[col].apply(lambda x:x if x in printing else '없음')
        elif col in ['디테일']:
            _df[col] = _df[col].apply(lambda x:list(set(detail) & set(map(lambda x:x.strip(), x.split(',')))))
            _df[col] = _df[col].apply(lambda x:x if len(x) > 0 else ['없음'])
        else:
            raise

    for column in columns:
        if column == '디테일':
            for det in detail:
                _df[det] = _df['디테일'].apply(lambda x:1 if det in x else 0)
            continue

        items = _df[column].values
        
        encoder = LabelEncoder()
        encoder.fit(items)
        labels = encoder.transform(items)
        _labels = labels.reshape(-1,1)
            
        oh_encoder = OneHotEncoder()
        oh_encoder.fit(_labels)
        oh_labels = oh_encoder.transform(_labels)
            
        _df[column+'_ID'] = labels
    return _df

def add_language_input(_df):
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
    _df['language_input'] = _df['대분류'].apply(lambda x:'group '+ko2en[x]+' '+tmp)
    return _df

def save_df(df, df_path):
    df.to_csv(df_path, index=False)

def main(label_dir_path, df_path):
    df = load_dataset_info(label_dir_path)
    df = preprocessing(df, save_img_path)
    df = label_encoding(df)
    df = add_language_input(df)
    save_df(df, df_path)


if __name__ == "__main__":
    label_dir_path = '../data/라벨링데이터'
    df_path = '../data_table.csv'
    main(label_dir_path, df_path)
