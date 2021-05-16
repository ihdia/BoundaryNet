from skimage.io import imread
import skimage.color as color
import cv2
import torch
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt
from Utils.poly_point_isect import isect_polygon__naive_check
import copy
import Utils.dpfuncdist as dputils
import random


def check_self_intersection(poly):
    # The polygon MUST be in float
    return isect_polygon__naive_check(poly)


def count_self_intersection(polys, grid_size):
    """
    :param polys: Nx1 poly
    :return: number of polys that have self-intersection
    """
    new_polys = []
    isects = []
    for poly in polys:
        poly = get_masked_poly(poly, grid_size)
        poly = class_to_xy(poly, grid_size).astype(np.float32)
        isects.append(check_self_intersection(poly.tolist()))

    return np.array(isects, dtype=np.float32)


def create_folder(path):
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return nodes[np.argmin(dist_2)]


def closest_node_index(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def rgb_img_read(img_path):
    """
    Read image and always return it as a RGB image (3D vector with 3 channels).
    """
    img = imread(img_path)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)

    # Deal with RGBA
    img = img[..., :3]

    if img.dtype == 'uint8':
        # [0,1] image
        img = img.astype(np.float32) / 255

    return img


def get_full_mask_from_instance(min_area, instance):
    img_h, img_w = instance['img_height'], instance['img_width']

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for component in instance['components']:
        p = np.array(component['poly'], np.int)
        if component['area'] < min_area:
            continue
        else:
            draw_poly(mask, p)

    return mask


def get_full_mask_from_xy(poly, grid_size, patch_w, starting_point,
                          out_h, out_w):
    poly = get_masked_poly(poly, grid_size)
    poly = class_to_xy(poly, grid_size)
    poly = poly0g_to_poly01(poly, grid_size)
    poly = poly * patch_w
    poly[:, 0] += starting_point[0]
    poly[:, 1] += starting_point[1]
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    poly = poly.astype(np.int)
    draw_poly(mask, poly)

    return mask, poly


def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates

    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]

    return poly


def poly0g_to_poly01(polygon, grid_side):
    """
    [0, grid_side] coordinates to [0, 1].
    Note: we add 0.5 to the vertices so that the points
    lie in the middle of the cell.
    """
    result = (polygon.astype(np.float32) + 0.5) / grid_side

    return result


def get_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    mask[poly[:, 1], poly[:, 0]] = 1.

    return mask


def get_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    cv2.polylines(mask, [poly], True, [1])

    return mask


def class_to_grid(poly, out_tensor, grid_size):
    """
    NOTE: Torch function
    accepts out_tensor to do it inplace

    poly: [batch, ]
    out_tensor: [batch, 1, grid_size, grid_size]
    """
    out_tensor.zero_()
    # Remove old state of out_tensor

    b = 0
    for i in poly:
        if i < grid_size * grid_size:
            x = (i % grid_size).long()
            y = (i / grid_size).long()
            out_tensor[b, 0, y, x] = 1
        b += 1

    return out_tensor


def xy_to_class(poly, grid_size):
    """
    NOTE: Torch function
    poly: [bs, time_steps, 2]

    Returns: [bs, time_steps] with class label
    for x,y location or EOS token
    """
    batch_size = poly.size(0)
    time_steps = poly.size(1)

    poly[:, :, 1] *= grid_size
    poly = torch.sum(poly, dim=-1)

    poly[poly < 0] = grid_size ** 2
    # EOS token

    return poly


def gather_feature(id, feature):
    feature_id = id.unsqueeze_(2).long().expand(id.size(0),
                                                id.size(1),
                                                feature.size(2)).detach()

    cnn_out = torch.gather(feature, 1, feature_id).float()
    # print(id, feature_id)
    return cnn_out


def masks_from_poly(pred, width, height):
    masks = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(pred, list):
        pred = [pred]
    for p in pred:
        masks = draw_poly(masks, p)

    return masks


def prepare_gcn_component(pred_polys, grid_sizes, max_poly_len, n_adj=3):
    batch_array_feature_indexs = []

    curr_p = pred_polys[0]
    p_index = []

    curr_p_grid_size = np.floor(curr_p).astype(np.int32)
    curr_p_grid_size[:, 0] *= grid_sizes[0]
    curr_p_index = np.sum(curr_p_grid_size, axis=-1)

    p_index.append(curr_p_index)

    array_feature_indexs = np.zeros((len(grid_sizes), max_poly_len), np.float32)

    array_feature_indexs[:, :max_poly_len] = np.array(p_index)

    batch_array_feature_indexs.append(array_feature_indexs)

    adj_matrix = create_adjacency_matrix_cat(pred_polys.shape[0], n_adj, max_poly_len)


    return {
        'feature_indexs': torch.Tensor(np.stack(batch_array_feature_indexs, axis=0)),
        'adj_matrix': torch.Tensor(adj_matrix)
    }


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def extreme_points(polygon, pert=0):
    def find_point(ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return sel_id

    x = polygon[:, 0]
    y = polygon[:, 1]

    ex_0 = find_point(np.where(y >= np.max(y) - pert))
    ex_1 = find_point(np.where(y <= np.min(y) + pert))
    ex_2 = find_point(np.where(x >= np.max(x) - pert))
    ex_3 = find_point(np.where(x <= np.min(x) + pert))
    return polygon[ex_0], polygon[ex_1], polygon[ex_2], polygon[ex_3]


def make_gt(labels, sigma=10, h=224, w=224):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    if labels is None:
        gt = make_gaussian((h, w), center=(h // 2, w // 2), sigma=sigma)
    else:
        labels = np.array(labels)

        if labels.ndim == 1:
            labels = labels[np.newaxis]
        gt = np.zeros(shape=(h, w), dtype=np.float64)
        for ii in range(labels.shape[0]):
            gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=np.float32)

    return gt


def create_adjacency_matrix_cat(batch_size, n_adj, n_nodes):
    a = np.zeros([batch_size, n_nodes, n_nodes])

    for t in range(batch_size):
        for i in range(n_nodes):
            for j in range(int(-n_adj / 2), int(n_adj / 2 + 1)):
                if j != 0:
                    a[t][i][(i + j) % n_nodes] = 1
                    a[t][(i + j) % n_nodes][i] = 1

    return a.astype(np.float32)


def class_to_xy(poly, grid_size):
    """
    NOTE: Numpy function
    poly: [bs, time_steps] or [time_steps]

    Returns: [bs, time_steps, 2] or [time_steps, 2]
    """
    x = (poly % grid_size).astype(np.int32)
    y = (poly / grid_size).astype(np.int32)

    out_poly = np.stack([x, y], axis=-1)

    return out_poly


def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, [poly], 255)

    return mask


def get_masked_poly(poly, grid_size):
    """
    NOTE: Numpy function

    Given a polygon of shape (N,), finds the first EOS token
    and masks the predicted polygon till that point
    """
    if np.max(poly) == grid_size ** 2:
        # If there is an EOS in the prediction
        length = np.argmax(poly)
        poly = poly[:length]
        # This automatically removes the EOS

    return poly


def poly0g_to_index(polygon, grid_side=112):
    result = []
    for item in polygon:
        result.append(item[0] + item[1] * grid_side)
    return result


def mask_and_flatten_poly(polygons, masks, grid_side):
    result = []
    for i in range(len(polygons)):

        if masks[i]:
            result.append(polygons[i][0] + polygons[i][1] * grid_side)
        else:
            result.append(grid_side ** 2)

    return np.array(result)


def local_prediction_2xy(output_dim, t_vertices):
    """
    Convert a list of vertices index into a list of xy vertices
    """
    side = output_dim / 2

    x = t_vertices % output_dim - side

    y = ((t_vertices) / output_dim) - side

    return x, y


def dt_targets_from_class(poly, grid_size, dt_threshold):
    """
    NOTE: numpy function!
    poly: [bs, time_steps], each value in [0, grid*size**2+1)
    grid_size: size of the grid the polygon is in
    dt_threshold: threshold for smoothing in dt targets

    returns:
    full_targets: [bs, time_steps, grid_size**2+1] array containing
    dt smoothed targets to be used for the polygon loss function
    """
    full_targets = []
    for b in range(poly.shape[0]):
        targets = []
        for p in poly[b]:
            t = np.zeros(grid_size ** 2 + 1, dtype=np.int32)
            t[p] += 1

            if p != grid_size ** 2:  # EOS
                spatial_part = t[:-1]
                spatial_part = np.reshape(spatial_part, [grid_size, grid_size, 1])

                # Invert image
                spatial_part = -1 * (spatial_part - 1)
                # Compute distance transform
                spatial_part = distance_transform_cdt(spatial_part, metric='taxicab').astype(np.float32)
                # Threshold
                spatial_part = np.clip(spatial_part, 0, dt_threshold)
                # Normalize
                spatial_part /= dt_threshold
                # Invert back
                spatial_part = -1. * (spatial_part - 1.)

                spatial_part /= np.sum(spatial_part)
                spatial_part = spatial_part.flatten()

                t = np.concatenate([spatial_part, [0.]], axis=-1)

            targets.append(t.astype(np.float32))
        full_targets.append(targets)

    return np.array(full_targets, dtype=np.float32)


if __name__ == '__main__':
    poly = np.array([[5, 5], [8, 8], [8, 5]])
    img = np.zeros((2, 10, 10), np.uint8)
    img = draw_poly(img[0], poly)
    print(img)