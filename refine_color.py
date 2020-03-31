from sklearn.neighbors import KDTree
from skimage.util import compare_images
from skimage.color import rgb2lab
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2

def refine_color(img, ref):
    h, w, c = ref.shape

    out = np.copy(img)

    ref_img = ref.reshape(-1,3)

    ref_colors = np.unique(ref_img, axis=0)

    tree = KDTree(ref_colors)

    for x in range(h):
        for y in range(w):
            color = out[x, y]
            idx = tree.query(color.reshape(1, -1),
                                       return_distance=False)
            ref_color = ref_colors[idx].reshape(1, 3)
            out[x, y] = ref_color
    return out


if __name__ == "__main__":
    ref = io.imread("~/Downloads/ref/011_29712.jpg")
    img = io.imread("~/Downloads/ref/res2.jpg")
    print("Skimage")
    sk_lab = rgb2lab(img)
    for c in range(3):
        min_value = np.min(sk_lab[:, :, c])
        max_value = np.max(sk_lab[:, :, c])
        avg = np.average(sk_lab[:, :, c])
        print(f"Min = {min_value}, Max = {max_value}, Avg = {avg}")
    print("CV2")
    cv_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    for c in range(3):
        min_value = np.min(cv_lab[:, :, c])
        max_value = np.max(cv_lab[:, :, c])
        avg = np.average(cv_lab[:, :, c])
        print(f"Min = {min_value}, Max = {max_value}, Avg = {avg}")
    # out = refine_color(img, ref)
    # fig, axes = plt.subplots(1, 3)

    # axes[0].imshow(ref)
    # axes[1].imshow(img)
    # axes[2].imshow(out)

    # diff_ref_img = np.average(compare_images(ref, img))
    # diff_ref_out = np.average(compare_images(ref, out))

    # axes[1].set_title(f"Diff = {diff_ref_img}")
    # axes[2].set_title(f"Diff = {diff_ref_out}")
    # plt.show()


