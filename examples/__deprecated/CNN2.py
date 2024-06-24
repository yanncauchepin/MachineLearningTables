import numpy as np


def conv(image, im_filter):
    """
    :param image: grayscale image as a 2-dimensional numpy array
    :param im_filter: 2-dimensional numpy array
    """

    # input dimensions
    height = image.shape[0]
    width = image.shape[1]

    # output image with reduced dimensions
    im_c = np.zeros((height - len(im_filter) + 1,
                     width - len(im_filter) + 1))

    # iterate over all rows and columns
    for row in range(len(im_c)):
        for col in range(len(im_c[0])):
            # apply the filter
            for i in range(len(im_filter)):
                for j in range(len(im_filter[0])):
                    im_c[row, col] += image[row + i, col + j] * im_filter[i][j]

    # fix out-of-bounds values
    im_c[im_c > 255] = 255
    im_c[im_c < 0] = 0

    # plot images for comparison
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.figure()
    plt.imshow(image, cmap=cm.Greys_r)
    plt.show()

    plt.imshow(im_c, cmap=cm.Greys_r)
    plt.show()
