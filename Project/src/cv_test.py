from ultralytics import YOLO
import os
import sys
from PIL import Image
import cv2
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: self_test <path_to_weights>')
    img_list = [os.path.join('./self_test/in/', f) for f in os.listdir('./self_test/in/')]
    model = YOLO(os.path.join(sys.argv[1], 'best.pt'))
    
    results = model(img_list)
    
    output_video_path = './self_test/out/tt1.mp4'
    # Get video properties
    frame_width = int(4000)
    frame_height = int(3000)
    fps = 1
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    for id, r in enumerate(results):
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
        out.write(np.asarray(im_array)) # BGR

    # Release the video capture and writer objects
    out.release()
    
    print(f'Video saved to: {output_video_path}')
    cv2.destroyAllWindows()
