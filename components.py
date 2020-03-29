import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as measure
import skimage.transform as transform
from skimage.filters import threshold_otsu
from tqdm import tqdm


def extract_components_by_color(image, min_area=10, count_thres=20):
    """
    Extracting component regions from the image

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
    for unique in tqdm(uniques):
        # Get coords by color
        # if unique == 16016015 or unique == 255:
        #     continue
        rows, cols = np.where(processed_image == unique)
        image_tmp = np.zeros_like(processed_image)
        image_tmp[rows, cols] = 255
        # Get components
        labels = measure.label(image_tmp, connectivity=1, background=0)
        for region in measure.regionprops(labels,
                                          intensity_image=processed_image):
            if region['area'] <= min_area:
                continue
            result[index] = region
            index += 1
    return result


def averageColor(img, coords):
    """
    Change pixels in regions of image to be the same pixel values.

    Parameters:
    ----------
    img : numpy array
        The image to be modified
    coords: numpy array, shape (N, 2)
        Coordinates of pixels in regions to be modified

    Returns
    ------

    """
    colors = img[coords[:, 0], coords[:, 1]]

    # Try KMeans
    # n_colors = 3
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    # flags = cv2.KMEANS_RANDOM_CENTERS

    # _, labels, palette = cv2.kmeans(colors.astype(np.float32), n_colors, None, criteria, 10, flags)
    # _, counts = np.unique(labels, return_counts=True)

    # dominant = palette[np.argmax(counts)]

    # Just take average
    dominant = np.average(colors, axis=0)
    img[coords[:, 0], coords[:, 1]] = dominant.astype(np.uint8)


def post_process(image):
    """
    Post process colorized image

    Parameters
    ----------
    image: numpy array, shape (H, W, 3)
        Image in RGB color space

    Returns
    -------
    out: numpy array, shape (H, W, 3)
        Postprocessed image in RGB color space
    """
    # image = cv2.imread('./up_0.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Binarize
    # thres = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    # thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 23, 5)

    # Filter
    kernel = np.ones((5, 5), np.uint8)
    thres = cv2.dilate(255 - thres, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)

    processed_image = 255 - thres
    # plt.imshow(processed_image, cmap=plt.cm.gray)

    cv2.imwrite("temp_sketch.jpg", processed_image)
    processed_image = cv2.imread("./temp_sketch.jpg")

    comps = extract_components_by_color(processed_image,
                                        min_area=10000,
                                        count_thres=100)

    out = np.copy(image)

    for comp in comps.values():
        averageColor(out, comp.coords)
        # plt.imshow(comp.filled_image, cmap=plt.cm.gray)
        # plt.show()

    return out

if __name__ == "__main__":
    image = cv2.imread("./up_2.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = post_process(image)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[1].imshow(out)
    axs[1].set_title("PostProcessed")
    cv2.imwrite("post_process.jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    plt.show()
