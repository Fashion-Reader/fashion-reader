
import os
import glob
import matplotlib.pyplot as plt


from tqdm.auto import tqdm


def image_rename(image_dir_path, save_img_path):
    img_paths = glob.glob(os.path.join(image_dir_path,'*','*'))

    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    for img_path in tqdm(img_paths):
        cate = img_path.split('/')[-2]
        name = img_path.split('/')[-1].lower()
        
        _save_img_path = os.path.join(save_img_path, cate)
        if not os.path.exists(_save_img_path):
            os.mkdir(_save_img_path)
        
        _save_img_path = os.path.join(_save_img_path, name)
        
        img = plt.imread(img_path)
        plt.imsave(_save_img_path, img)


if __name__ == "__main__":
    image_dir_path = '../data/원천데이터'
    save_img_path = '../new_data/'
    image_rename(image_dir_path, save_img_path)
