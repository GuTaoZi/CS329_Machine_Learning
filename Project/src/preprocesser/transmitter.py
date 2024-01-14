import json
import yaml
from PIL import Image

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error: {e}")
        return 1, 1

raw_json_path = './tt100k_2021/annotations_all.json'
new_yaml_path = './tt100k_2021/tt100k_2021.yaml'

category_dict = {}

if __name__=='__main__':
    with open(raw_json_path, 'r') as f:
        meta_data = json.load(f)

    trans_yaml = {}
    trans_yaml['path'] = './tt100k_2021'
    trans_yaml['train'] = 'train/'
    trans_yaml['val'] = 'val/'
    trans_yaml['test'] = 'test/'
    trans_yaml['nc'] = len(meta_data['types'])
    trans_yaml['names']={}
    for id, uu in enumerate(meta_data['types']):
        trans_yaml['names'][id] = uu
        category_dict[uu] = id

    with open(new_yaml_path, 'w') as f:
        yaml.dump(trans_yaml, f)
    
    imgs_len = len(meta_data['imgs'].items())
    done_len = 0
    print(imgs_len)
    
    for key, imgs in meta_data['imgs'].items():
        # x, y = get_image_dimensions('./tt100k_2021/images/'+imgs['path'])
        # if x != 2048 or y != 2048:
        #     print('AAAAAA: ', key)
        x, y = 2048, 2048
        label_path : str
        label_path = './tt100k_2021/labels/'+imgs['path']
        label_path = label_path[:-4] + '.txt'
        with open(label_path, 'w') as f:
            for obj in imgs['objects']:
                iid = category_dict[obj['category']]
                minx = obj['bbox']['xmin'] / 2048.0
                maxx = obj['bbox']['xmax'] / 2048.0
                miny = obj['bbox']['ymin'] / 2048.0
                maxy = obj['bbox']['ymax'] / 2048.0
                x_center = (minx+maxx)/2
                y_center = (miny+maxy)/2
                x_len = maxx-minx
                y_len = maxy-miny
                f.write(f'{iid} {x_center} {y_center} {x_len} {y_len}\n')
        done_len += 1
        if done_len % 500 == 0 or done_len==imgs_len:
            print(f'\r{done_len}/{imgs_len}', end="")
    print('')