import cv2
import numpy as np
import skimage.measure as measure
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

def extract_components_by_color(image, min_area=10, count_thres=20):
        """
        :param image: rgb image
        :param min_area:
        :return: a dictionary of components
        """
        # Convert rgb image to (h, w, 1)
        b, r, g = cv2.split(image)
        processed_image = b + 300 * (g + 1) + 300 * 300 * (r + 1)
        uniques, counts = np.unique(processed_image, return_counts=True)
        uniques = uniques[np.where(counts > count_thres)]
        index = 0
        result = {}
        print(uniques.shape)
        for unique in tqdm(uniques):
            # Get coords by color
            # if unique == 16016015 or unique == 255:
            #     continue
            rows, cols = np.where(processed_image == unique)
            image_tmp = np.zeros_like(processed_image)
            image_tmp[rows, cols] = 255
            # Get components
            labels = measure.label(image_tmp, connectivity=1, background=0)
            for region in measure.regionprops(labels, intensity_image=processed_image):
                if region['area'] <= min_area:
                    continue
                result[index] = region
                index += 1
        return result

def visualize(img, components, ax):
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    for region in components.values():
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

if __name__ == "__main__":
    img = cv2.imread("../bg_remove.png")
    resized = cv2.resize(img, (112, 64))
    components = extract_components_by_color(img,min_area=300,
                                             count_thres=100)
    resized_components = extract_components_by_color(resized)
    print(len(components))
    print(len(resized_components))
    fig, axs = plt.subplots(1, 2)
    visualize(img, components, axs[0])
    visualize(resized, resized_components, axs[1])
    plt.show()


