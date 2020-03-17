import os

import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def trim_video(video_path, fps, skip_frame, start_second, end_second, trim_folder):
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
    start_second: int
        Start of the interval to extract in second
    end_second : int
        End of the interval to extract in second
    trim_folder : str
        Folder contains extracted frame
    """

    video_name = os.path.basename(video_path)

    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    count = 1
    start_frame = start_second * fps
    end_frame = end_second * fps
    while success:
        if start_frame <= count <= end_frame:
            if (count - start_frame) % skip_frame == 0:
                frame_name = "{}_{}.jpg".format(video_name, count)
                frame_path = os.path.join(trim_folder, frame_name)
                cv2.imwrite(frame_path, img)
        elif count > end_frame:
            break
        success, img = vidcap.read()
        count += 1


def extract_video_with_timming_file(video_path, fps, skip_frame, timming_file,
                                    trim_folder):
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
    """
    with open(timming_file, "r") as f:
        n = int(f.readline())
        for i in range(n):
            start, end = map(int, f.readline().split())
            # Storing each extracted interval in a folder
            extracted_folder = os.path.join(trim_folder, "{}_{}".format(start, end))
            os.makedirs(extracted_folder, exist_ok=True)
            trim_video(video_path, fps, skip_frame, start, end,
                       extracted_folder)


if __name__ == "__main__":
    os.makedirs("trim", exist_ok=True)
    # trim_video("./test.avi", 24,  0, 1, "trim")
    extract_video_with_timming_file("./op1.mp4", 24, 8, "./op1.txt", "trim")
