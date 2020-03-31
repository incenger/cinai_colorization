import cv2
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--origin_folder', help='Original folder to prepare testing data')
parser.add_argument('--new_folder', help='New folder to store testing data')
opt = parser.parse_args()

ori_folder = opt.origin_folder
folder = opt.new_folder

path_frames = glob.glob(ori_folder + '/frames/*')
ref_frame = glob.glob(ori_folder + '/ref/*')
print(path_frames)
print(ref_frame)

if not os.path.isdir(folder):
    os.mkdir(folder)
if not os.path.isdir(folder + '/frames'):
    os.mkdir(folder + '/frames')
if not os.path.isdir(folder + '/ref'):
    os.mkdir(folder + '/ref')

for path in path_frames:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    name = os.path.split(path)[-1]
    cv2.imwrite(folder + '/frames/' + name, img[:, :, 0])

ref = cv2.imread(ref_frame[0])
name = os.path.split(ref_frame[0])[-1]
cv2.imwrite(folder + '/ref/ori.jpg', ref)

print('done')

# img = cv2.imread('data/cut_300/frames/000_3888.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('test/gray.jpg', img)

# img = cv2.imread('data/cut_300/frames/000_3888.jpg')
# res = cv2.resize(img, (112, 64), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite('test/res.jpg', res)


# import numpy as np
# img = cv2.imread('test/res.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
# gray = cv2.imread('test/gray.jpg', cv2.IMREAD_GRAYSCALE)
# up = np.stack((gray, img[:, :, 1], img[:, :, 2]), 2)
# up = cv2.cvtColor(up, cv2.COLOR_LAB2BGR)
# cv2.imwrite('test/up.jpg', up)