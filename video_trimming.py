import os
import sys
import argparse
import logging

import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def trim_video(video_path, fps, skip_frame, time_interval, trim_folder, offset):
    """
    Saving the frames of the video from start_second to end_second
    The name of each frame is "originalname_framecount.jpg".

    Example
    trim("test.mp4",24, 1, 2, "trim")
    // 24 frames from 1s to 2s of the "test.mp4" video (test_24.jpg, test_25.jpg, ...) are saved  in "trim" folder.

    Parameters:
    ----------
    video_path : str
        Path to video to extract
    fps : int
        Frame per second of the video
    skip_frame: int
        How many frames we want to skip between each extracted frame?
    time_interval: list
        List of sorted time interval (in second) to extract frames
        E.g: [[1, 3], [4, 6]]
    trim_folder : str
        Folder contains extracted frame
    off_set: int
        Offset for cut index
    """

    video_name = os.path.basename(video_path)

    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()

    count = 1
    cut_idx = 0

    cut_path = os.path.join(trim_folder,
                            "cut_{}".format(str(cut_idx+offset).zfill(6)),
                            "frames")
    os.makedirs(cut_path, exist_ok=True)

    while success:
        start_frame = time_interval[cut_idx][0] * fps
        end_frame = time_interval[cut_idx][1] * fps

        if start_frame <= count <= end_frame:
            if (count - start_frame) % skip_frame == 0:
                frame_name = "{}_{}.jpg".format(str(cut_idx+offset).zfill(6), count)
                frame_path = os.path.join(cut_path, frame_name)

                logging.info("Saving frame {} in {}".format(
                    count, video_path))

                cv2.imwrite(frame_path, img)

        elif count > end_frame:
            cut_idx += 1
            if cut_idx == len(time_interval):
                break
            # Make folder for new cut
            cut_path = os.path.join(trim_folder,
                                    "cut_{}".format(str(cut_idx+offset).zfill(6)),
                                    "frames")
            os.makedirs(cut_path, exist_ok=True)

        success, img = vidcap.read()

        # Workaround when failed to read a frame
        if cut_idx < len(time_interval) and not success:
            success = True
        elif success:
            count += 1


def extract_video_with_timming_file(video_path, fps, skip_frame, timming_file,
                                    trim_folder, offset):
    """
    Read timming interval from a txt file and performing extracting from the video

    The first line in the timing line contains N - the number of interval to extract. Each line
    in next N line has two integers start second and end second separated by a white space.
    Example of timming file
    `timing.txt`
    3
    1 2
    4 8
    9 10

    Parameters:
    ----------
    video_path : str
        Path to video to extract
    fps : int
        Frame per second of the video
    skip_frame: int
        How many frames we want to skip between each extracted frame?
    timming_file: str
        The file contains timing intervals
    off_set: int
        Offset for cut index
    """
    with open(timming_file, "r") as f:
        n = int(f.readline())
        time_interval = []
        for i in range(n):
            start, end = map(int, f.readline().split())
            time_interval.append([start, end])
            # Storing each extracted interval in a folder
            # extracted_folder = os.path.join(trim_folder, "{}_{}".format(start, end))
            # os.makedirs(extracted_folder, exist_ok=True)
        trim_video(video_path, fps, skip_frame, time_interval, trim_folder,
                   offset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to video")
    parser.add_argument("timming_file", help="Path to timming file")
    parser.add_argument("extracted_folder", help="Folder storing cut")
    parser.add_argument("offset", type=int,  help="Offset for cut index")
    parser.add_argument("-v", "--verbose", action="store_true")
    agrs = parser.parse_args()
    if agrs.verbose:
        logging.basicConfig(level=logging.INFO)

    os.makedirs(agrs.extracted_folder, exist_ok=True)

    extract_video_with_timming_file(agrs.video_path, 24, 6, agrs.timming_file,
                                    agrs.extracted_folder, agrs.offset)
