'''
Author: GuTaoZi gutao.official@gmail.com
Date: 2023-11-29 14:05:58
LastEditors: GuTaoZi gutao.official@gmail.com
LastEditTime: 2023-11-29 15:58:04
FilePath: \Lab10_Exercise\experiment.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "./samples/vehicles"
neg_dir = "./samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "./videos/test_video.mp4"


def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """
    # TODO: You need to adjust hyperparameters
    # Extract HOG features from images in the sample directories and
    # return results and parameters in a dict.
    feature_data = processFiles(
        pos_dir, neg_dir, recurse=True, hog_features=True, color_space="gray"
    )

    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)

    # TODO: You need to adjust hyperparameters of loadClassifier() and detectVideo()
    #       to obtain a better performance

    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector().loadClassifier(classifier_data=classifier_data)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap)


def experiment2():
    feature_data = processFiles(
        pos_dir,
        neg_dir,
        recurse=True,
        color_space="luv",
        channels=[0, 1, 2],
        hog_features=True,
        hist_features=True,
        spatial_features=True,
        pix_per_cell=(8, 8),
        cells_per_block=(2, 2),
        hog_bins=20,
        hist_bins=16,
        spatial_size=(20, 20),
    )

    classifier_data = trainSVM(feature_data=feature_data, C=1000)

    detector = Detector(
        init_size=(90, 90),
        x_overlap=0.5,
        y_step=0.01,
        y_range=(0.55, 0.89),
        scale=1.3,
    )

    detector.loadClassifier(classifier_data=classifier_data)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(
        video_capture=cap, num_frames=9, threshold=120, draw_heatmap_size=0.3
    )


# def experiment2
#    ...

if __name__ == "__main__":
    experiment2()
    # experiment2() may you need to try other parameters
    # experiment3 ...
