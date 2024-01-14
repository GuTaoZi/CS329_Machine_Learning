from ultralytics import YOLO
import os
import sys
from PIL import Image

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('useage: self_test <path_to_weights>')
    img_list = [os.path.join('./self_test/in/', f) for f in os.listdir('./self_test/in/')]
    model = YOLO(os.path.join(sys.argv[1], 'best.pt'))
    
    results = model(img_list)
    
    for id, r in enumerate(results):
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
        im.save(f'./self_test/out/{id}.jpg')