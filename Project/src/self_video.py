from ultralytics import YOLO
import os
import sys
from PIL import Image
import cv2
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: self_test <path_to_weights>')
    
    video_path = './self_test/video/3_enh.mp4'
    output_video_path = './self_test/out/eca_output_video_3_enh.mp4'  # Change the output video format if needed
    model = YOLO(os.path.join(sys.argv[1], 'best.pt'))
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 60
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    results = model(video_path, stream=True)
    
    for r in results:
        im_array = r.plot()
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
        # Write the frame to the output video
        out.write(np.asarray(im_array))
    
    # Release the video capture and writer objects
    cap.release()
    out.release()
    
    print(f'Video saved to: {output_video_path}')
    cv2.destroyAllWindows()
