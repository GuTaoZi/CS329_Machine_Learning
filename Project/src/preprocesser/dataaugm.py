import json
from collections import defaultdict
import os
import imgaug.augmenters as iaa
import imgaug as ia
import cv2

def get_file_names(directory):
    """Return a list of filenames in the given directory."""
    return [file for file in os.listdir(directory)]

def read_json(path):
    data = None
    with open(path) as f:
        data = json.load(f)
    if data is None:
        print(f'failed to read file: {path}.')
        exit(1)
    return data

def count_nc():
    data = read_json('./tt100k_2021/annotations_all.json')
    result = defaultdict(int)
    
    for imgs in data['imgs'].items():
        for obj in imgs[1]['objects']:
            result[obj['category']] += 1
    
    print('{')
    for item in sorted(result.items()):
        print(f'\t\"{item[0]}\": {item[1]},')
    print('}')

def exist_sign(json_arr, sign_code : str):
    for item in json_arr:
        if item['category'] == sign_code:
            return True
    return False

def inlist(strlist, ustr):
    # for item in strlist:
    #     item:str
    #     if item == ustr:
    #         return True
    # return False
    return (ustr in strlist)

def dataaugimg(imgdir, imgname, confdir, conf, label_dict):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # apply gaussian blur with 50% probability
        iaa.ContrastNormalization((0.75, 1.5)),  # contrast normalization
        iaa.Multiply((0.8, 1.2), per_channel=0.2),  # random brightness
        iaa.Affine(
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )  # affine transformations: rotate and shear
    ], random_order=True)
    
    image_path = os.path.join(imgdir, imgname + '.jpg')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert bbox data to imgaug format
    ia_bboxes = [ia.BoundingBox(x1=bbox['bbox']['xmin'], y1=bbox['bbox']['ymin'],
                                x2=bbox['bbox']['xmax'], y2=bbox['bbox']['ymax'],
                                label=label_dict[bbox['category']]) for bbox in conf]
    bbs = ia.BoundingBoxesOnImage(ia_bboxes, shape=image.shape)

    # Apply the augmentation
    image_bbs_aug_list = [seq(image=image, bounding_boxes=bbs) for _ in range(3)]

    # Save or process the augmented image and bboxes
    # output_path = os.path.join(output_directory, f"aug_{image_file}")
    # cv2.imwrite(output_path, image_aug)
    
    for i, (image_aug, bbs_aug) in enumerate(image_bbs_aug_list):
        image_out_path = os.path.join(imgdir, imgname + f'_{i}.jpg')
        image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_out_path, image_aug)
        conf_out_path = os.path.join(confdir, imgname + f'_{i}.txt')
        with open(conf_out_path, 'w') as f:
            for iabb in bbs_aug:
                iabb : ia.BoundingBox
                iid = iabb.label
                minx = iabb.x1 / 2048.0
                maxx = iabb.x2 / 2048.0
                miny = iabb.y1 / 2048.0
                maxy = iabb.y2 / 2048.0
                x_center = (minx+maxx)/2
                y_center = (miny+maxy)/2
                x_len = maxx-minx
                y_len = maxy-miny
                f.write(f'{iid} {x_center} {y_center} {x_len} {y_len}\n')
    
    print(f'aug {image_path} done.')

def read_label_dict(data):
    ret = {}
    for i, obj in enumerate(data['types']):
        ret[obj] = i
    return ret

if __name__ == '__main__':
    print("start")
    data = read_json('./tt100k_2021/annotations_all.json')
    test_imgs = get_file_names('./tt100k_2021/images/test')
    train_imgs = get_file_names('./tt100k_2021/images/train')
    val_imgs = get_file_names('./tt100k_2021/images/val')
    label_dict = read_label_dict(data)
    print("read all")
    
    for imgs in data['imgs'].items():
        if exist_sign(imgs[1]['objects'], 'pl15'):
            id = imgs[1]['id']
            print(f'find exist {id}')
            if inlist(test_imgs, imgs[0]+'.jpg'):
                dataaugimg('./tt100k_2021/images/test', imgs[0], './tt100k_2021/labels/test', imgs[1]['objects'], label_dict)
            if inlist(train_imgs, imgs[0]+'.jpg'):
                dataaugimg('./tt100k_2021/images/train', imgs[0], './tt100k_2021/labels/train', imgs[1]['objects'], label_dict)
            if inlist(val_imgs, imgs[0]+'.jpg'):
                dataaugimg('./tt100k_2021/images/val', imgs[0], './tt100k_2021/labels/val', imgs[1]['objects'], label_dict)