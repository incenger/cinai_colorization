import cv2
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='data/test', type=str, help='Path to testing data folder')
opt = parser.parse_args()

PATH = opt.path

path_cuts = glob.glob(PATH + '/*')
for folder in path_cuts:
    print(folder)
    path_frames = glob.glob(folder + '/frames/*')
    ref_frame = glob.glob(folder + '/ref/*')
    print(path_frames)
    print(ref_frame)
    print()

    if not os.path.isdir(folder + '/gray'):
        os.mkdir(folder + '/gray')

    for path in path_frames:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        name = os.path.split(path)[-1]
        cv2.imwrite(folder + '/gray/gray_' + name, img[:, :, 0])

print('done')
