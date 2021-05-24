import numpy as np
from skimage.draw import polygon, polygon_perimeter
from scipy.ndimage.morphology import distance_transform_cdt


def create_polygon(shape, vertices, perimeter=False):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero
    """
    base_array = np.zeros(shape, dtype=np.float32)  # Initialize your array of zeros

    if not isinstance(vertices, list):
        vertices = [vertices]

    for v in vertices:
        r = v[:, 1]  # Row, or Y
        c = v[:, 0]  # Column, or X
        if perimeter:
            rr, cc = polygon_perimeter(r, c)
        else:
            rr, cc = polygon(r, c)

        base_array[rr, cc] = 1

    return base_array


def x_accumulated_image(mask, bi_d=False):
    """
    Input: mask of 0/1 values of shape -> [bs, width, height] or [width, height]
    Output: image where pixel (x,y) contains number of pixels at (x',y, x' >= x)
    with a mask of 1 of same shape as input
    If bi_d, then also return for x'<=x
    """

    assert len(mask.shape) in [2, 3], 'Mask can either have rank 2 or 3!'

    out_im = np.zeros(mask.shape, dtype=np.float32)
    mask_w = mask.shape[-1]
    out_im[..., :, mask_w - 1] = mask[..., :, mask_w - 1]
    for c in range(mask_w - 1)[::-1]:
        out_im[..., :, c] = out_im[..., :, c + 1] + mask[..., :, c]

    if bi_d:
        out_im_rev = np.zeros(mask.shape, dtype=np.float32)
        out_im_rev[..., :, 0] = mask[..., :, 0]
        for c in range(1, mask_w):
            out_im_rev[..., :, c] = out_im_rev[..., :, c - 1] + mask[..., :, c]

        return out_im, out_im_rev

    return out_im


def y_accumulated_image(mask, bi_d=False):
    """
    Input: mask of 0/1 values of shape -> [bs, width, height] or [width, height]
    Output: image where pixel (x,y) contains number of pixels at (x,y', y' >= y)
    with a mask of 1 of same shape as input
    If bi_d, then also return for y'<=y
    """

    assert len(mask.shape) in [2, 3], 'Mask can either have rank 2 or 3!'

    out_im = np.zeros(mask.shape, dtype=np.float32)
    mask_h = mask.shape[-2]
    out_im[..., mask_h - 1, :] = mask[..., mask_h - 1, :]

    for r in range(mask_h - 1)[::-1]:
        out_im[..., r, :] = out_im[..., r + 1, :] + mask[..., r, :]

    if bi_d:
        out_im_rev = np.zeros(mask.shape, dtype=np.float32)
        out_im_rev[..., 0, :] = mask[..., 0, :]
        for r in range(1, mask_h):
            out_im_rev[..., r, :] = out_im_rev[..., r - 1, :] + mask[..., r, :]

        return out_im, out_im_rev

    return out_im


def create_shape_mask(shape, vertices, bi_d=True, add_x=True):
    # print(shape)
    # print(vertices)
    # print(vertices.shape)
    polygon_array = create_polygon(shape, vertices)

    h, w = polygon_array.shape

    mask_y = y_accumulated_image(polygon_array, bi_d)

    if add_x:
        mask_x = x_accumulated_image(polygon_array, bi_d)
        return mask_y, mask_x

    return mask_y


def create_intersect_mask(num_v, max_v):
    """
    Creates intersect mask as needed by polygon_intersection_new
    in batch_poly_utils (for a single example)
    """
    intersect_mask = np.zeros((max_v, max_v), dtype=np.float32)

    for i in range(num_v - 2):
        for j in range((i + 2) % num_v, num_v - int(i == 0)):
            intersect_mask[i, j] = 1.

    return intersect_mask


def dt_image(shape, vertices, thresh=5):
    """
    Create a distance transform image given the size of
    the canvas and vertices of the polygon. Any distance
    less than the threshold is set to 0
    """
    mask = create_polygon(shape, vertices, perimeter=True)
    # Invert
    mask = 1 - mask
    dt_mask = distance_transform_cdt(mask, metric='taxicab').astype(np.float32)

    dt_mask[dt_mask < thresh] = 0
    return dt_mask


def _test_dt_image():
    # Test dt_image
    gt = np.array([[70., 30.], [50., 90.], [10., 80.], [40., 10.]])
    dt_mask = dt_image((100, 100), gt)

    from matplotlib import pyplot as plt
    plt.imshow(dt_mask)
    plt.show()


if __name__ == '__main__':
    _test_dt_image()