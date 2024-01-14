import os
import random
import shutil

def random_split(image_train_dir:str, image_val_dir:str,
                 label_train_dir:str, label_val_dir:str,
                 val_ratio:float):
    random.seed(42)
    
    files = [f[:-4] for f in os.listdir(image_train_dir) if f.endswith('.jpg')]
    
    num_val = int(len(files)/2*val_ratio)
    
    random.shuffle(files)
    
    val_files = files[:num_val]
    
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)
    
    for file in val_files:
        src_path = os.path.join(label_train_dir, file+'.txt')
        dest_path = os.path.join(label_val_dir, file+'.txt')
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        src_path = os.path.join(image_train_dir, file+'.jpg')
        dest_path = os.path.join(image_val_dir, file+'.jpg')
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)

if __name__ == '__main__':
    random_split('./tt100k_2021/images/train','./tt100k_2021/images/val','./tt100k_2021/labels/train','./tt100k_2021/labels/val', 0.2)