import torch
import torch.nn.functional as F
import numpy as np
import Utils2.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """

    assert(output.size() == label.size())

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_pos_pix = torch.mul(w_void, loss_pos_pix)
        loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = torch.sum(loss_pos_pix)
    loss_neg = torch.sum(loss_neg_pix)

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss

def fp_edge_loss(gt_edges, edge_logits):
    """
    Edge loss in the first point network

    gt_edges: [batch_size, grid_size, grid_size] of 0/1
    edge_logits: [batch_size, grid_size*grid_size]
    """
    edges_shape = gt_edges.size()
    gt_edges = gt_edges.view(edges_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges)

    return torch.mean(loss)

def fp_vertex_loss(gt_verts, vertex_logits):
    """
    Vertex loss in the first point network
    
    gt_verts: [batch_size, grid_size, grid_size] of 0/1
    vertex_logits: [batch_size, grid_size**2]
    """
    verts_shape = gt_verts.size()
    gt_verts = gt_verts.view(verts_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(vertex_logits, gt_verts)

    return torch.mean(loss)



def poly_mathcing_loss(pnum, pred, gt, loss_type="L1"):
    batch_size = pred.size()[0]
    pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
    for b in range(batch_size):
        for i in range(pnum):
            pidx = (np.arange(pnum) + i) % pnum
            pidxall[b, i] = pidx

    pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

    # import ipdb;
    # ipdb.set_trace()
    feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), gt.size(2)).detach()
    feature_id = feature_id.to(device)

    gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)

    pred_expand = pred.unsqueeze(1)
    # print("pred:",pred_expand.shape)
    gt_expand = gt_expand.float().to(device)
    # print("gt:",gt_expand.shape)

    dis = pred_expand - gt_expand
    # dis = torch.max((0.5-1) * dis,0.5 * dis)
    # print("asdfg",dis.shape)
    if loss_type == "L2":
        dis = (dis ** 2).sum(3).sqrt().sum(2)
    elif loss_type == "L1":
        # dis = torch.abs(dis).sum(3)
        # # dis = torch.abs(dis**2)
        dis = torch.abs(dis).sum(3).sum(2)
        # # # # # # print("dis",dis.shape)
        # topk1, indices = torch.topk(dis, 200)
        # # # # # # # print("topktopk",topk1.shape)
        # dis = topk1.sum(2)

    elif loss_type == "SL1":
        less_than_one = dis < 10
        less_than_one = less_than_one.float()
        dis = (less_than_one * 0.05 * dis ** 2) + (1 - less_than_one) * (dis - 0.05)
        dis = torch.abs(dis).sum(3).sum(2)


    min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
    min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)

    min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
                            expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
    gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

    return gt_right_order, torch.mean(min_dis)


