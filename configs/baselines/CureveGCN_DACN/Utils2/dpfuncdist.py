
import cv2
import copy
import numpy as np

##########################################################
height = 224;
width = 224;


#############################################################
def cutallpha(im_hxwx4_8u, newsz):
    alpha1 = im_hxwx4_8u[:, :, 3]
    row, col = np.where(alpha1 > 0)
    rowmin = np.min(row)
    rowmax = np.max(row)
    colmin = np.min(col)
    colmax = np.max(col)
    im_hxwx3_8u = im_hxwx4_8u[rowmin:rowmax + 1, colmin:colmax + 1, :3]
    al_hxw_8u = alpha1[rowmin:rowmax + 1, colmin:colmax + 1]
    im_hxwx3_8u = cv2.resize(im_hxwx3_8u, newsz)
    al_hxw_8u = cv2.resize(al_hxw_8u, newsz)
    return im_hxwx3_8u, al_hxw_8u


def im2countaur(im_hxw_8u):
    alpha1 = cv2.copyMakeBorder(im_hxw_8u, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
    ret, thresh = cv2.threshold(alpha1, 10, 255, 0)
    thresh, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    assert len(cnts) == 1
    maxcnt = cnts[0]

    pgtnp = np.array(maxcnt, dtype=np.float32)
    pgtnp = np.reshape(pgtnp, newshape=(-1, 2))
    pgtnp = pgtnp - 3.0

    pgtnp = (pgtnp + 0.5) / height
    return pgtnp


def imdrawcontour2(pointsnp, linecolor, pointcolor, ima=None):
    if ima is None:
        ima = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    pointsnp = (pointsnp * height - 0.5).astype(np.int32)
    pnum = pointsnp.shape[0]
    for i in range(pnum):
        pbe = pointsnp[i];
        pen = pointsnp[(i + 1) % pnum]
        if np.any(pbe < 0):
            continue
        if np.any(pbe > height - 1):
            continue
        if np.any(pen < 0):
            continue
        if np.any(pen > height - 1):
            continue
        cv2.circle(ima, (pbe[0], pbe[1]), 2, pointcolor, thickness=-1)
        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor, thickness=1)
        # print (pbe[0], pbe[1])
    return ima


def imdrawcontour(pointsnp1, pointsnp2, linecolor1, linecolor2, linecolor3, ima=None, name="matching.png"):
    if ima is None:
        ima = np.zeros(shape=(height , width , 3), dtype=np.uint8)
    pointsnp1 = (pointsnp1 * height - 0.5).astype(np.int32)
    pointsnp2 = (pointsnp2 * height - 0.5).astype(np.int32)

    pointsnp1 = pointsnp1
    pointsnp2 = pointsnp2


    pnum = pointsnp1.shape[0]
    for i in range(pnum):
        pbe = pointsnp1[i]
        pen = pointsnp1[(i + 1) % pnum]
        # cv2.circle(ima, (pbe[0], pbe[1]), 2, pointcolor, thickness=-1)
        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor1, thickness=1)

        pbe = pointsnp2[i]
        pen = pointsnp2[(i + 1) % pnum]
        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor2, thickness=1)

        pbe = pointsnp1[i]
        pen = pointsnp2[i]

        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor3, thickness=1)
    cv2.imwrite(name, ima)


    return ima


def imdrawcontour3(pointsnp1, pointsnp2, match1, match2, linecolor1, linecolor2, linecolor3, ima=None):
    if ima is None:
        ima = np.zeros(shape=(height , width , 3), dtype=np.uint8)

    pointsnp1 = (pointsnp1 * height - 0.5).astype(np.int32)
    pointsnp2 = (pointsnp2 * height - 0.5).astype(np.int32)

    pointsnp1 = pointsnp1
    pointsnp2 = pointsnp2

    pnum = pointsnp1.shape[0]
    for i in range(pnum):
        pbe = pointsnp1[i];
        pen = pointsnp1[(i + 1) % pnum]
        if np.any(pbe < 0):
            continue
        if np.any(pbe > height - 1):
            continue
        if np.any(pen < 0):
            continue
        if np.any(pen > height - 1):
            continue
        # cv2.circle(ima, (pbe[0], pbe[1]), 2, pointcolor, thickness=-1)
        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor1, thickness=1)

    pnum = pointsnp2.shape[0]
    for i in range(pnum):
        pbe = pointsnp2[i];
        pen = pointsnp2[(i + 1) % pnum]
        if np.any(pbe < 0):
            continue
        if np.any(pbe >  height - 1):
            continue
        if np.any(pen < 0):
            continue
        if np.any(pen > height - 1):
            continue
        # cv2.circle(ima, (pbe[0], pbe[1]), 2, pointcolor, thickness=-1)
        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor2, thickness=1)

    for i, m1 in enumerate(match1):
        m2 = match2[i]
        pbe = pointsnp1[m1]
        pen = pointsnp2[m2]

        if np.any(pbe < 0):
            continue
        if np.any(pbe > height - 1):
            continue
        if np.any(pen < 0):
            continue
        if np.any(pen > height - 1):
            continue
        # cv2.circle(ima, (pbe[0], pbe[1]), 2, pointcolor, thickness=-1)
        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor3, thickness=1)
        # cv2.imshow('match', ima)

        cv2.imwrite("matching.png", ima);
        # cv2.waitKey()
        # print (pbe[0], pbe[1])
    return ima


def accumulatedis(cnt_nx2_32f):

    n1 = cnt_nx2_32f.shape[0]
    cntdiff = cnt_nx2_32f[1:] - cnt_nx2_32f[:-1]
    cntdis = np.sqrt(np.sum(cntdiff ** 2, axis=1))
    cntsum = np.zeros(shape=(n1,), dtype=np.float32)
    dis = 0
    for i in range(n1 - 1):
        dis += cntdis[i]
        cntsum[i + 1] = dis
    cntsum /= dis
    return cntsum


def circleless(dis_n_32f, value, thres):
    diff = dis_n_32f - value
    diffright = diff % 1
    diffleft = -diff % 1
    diffmiin = np.min([diffright, diffleft], axis=0)
    drange = diffmiin < thres
    return drange

def pointresamplingnp(pgtnp_px2, newpnum):

    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        tfpsample = np.concatenate(psample, axis=0)
        return tfpsample

def matchpoints(contour1_n1x2, contour2_n2x2):
    n1 = contour1_n1x2.shape[0]
    n2 = contour2_n2x2.shape[0]

    contour1_n1x2 = contour1_n1x2 * 200
    contour2_n2x2 = contour2_n2x2 * 200


    # extract feature
    w = 0.02
    feature1_n1xc = w * contour1_n1x2
    feature2_n2xc = w * contour2_n2x2

    dist_n1xn2xc = np.expand_dims(feature1_n1xc, axis=1) - np.expand_dims(feature2_n2xc, axis=0)
    dist_n1xn2 = np.exp(-np.sqrt(np.sum(dist_n1xn2xc ** 2, axis=2)) / 10.0)


    v = np.max(dist_n1xn2)
    p = np.argmax(dist_n1xn2)
    i2 = p % n2
    i1 = (p - i2) / n2

    feature1_n11xc = np.concatenate((feature1_n1xc[i1:], feature1_n1xc[:i1], feature1_n1xc[i1:i1 + 1]), axis=0)
    feature2_n21xc = np.concatenate((feature2_n2xc[i2:], feature2_n2xc[:i2], feature2_n2xc[i2:i2 + 1]), axis=0)
    dist_n11xn21xc = np.expand_dims(feature1_n11xc, axis=1) - np.expand_dims(feature2_n21xc, axis=0)
    dist_n11xn21 = np.exp(-np.sqrt(np.sum(dist_n11xn21xc ** 2, axis=2)) / 10.0)

    # diagnal line
    jmatch = np.arange( n1 +1, dtype=np.float32) / n1 * n2
    jbor = 7

    # begin from 0, 0
    score = -10000 * np.ones(shape=(n1 + 1, n2 + 1))
    score[0, 0] = dist_n11xn21[0, 0]

    strategy = np.zeros(shape=(n1 + 1, n2 + 1))

    for i in range(n1 + 1):
        for j in range(n2 + 1):

            if i == 0 and j == 0:
                continue

            if jbor > 0:
                if np.abs(j - jmatch[i]) > jbor:
                    continue

            s1 = 0
            s2 = 0
            s3 = 0

            # prefer go diagnal
            if i > 0 and j > 0:
                s1 = score[i - 1, j - 1] + dist_n11xn21[i, j]

            # if we go from top or from left, no reward
            if j > 0:
                s2 = score[i, j - 1] + dist_n11xn21[i, j- 1]
            if i > 0:
                s3 = score[i - 1, j] + dist_n11xn21[i - 1, j]

            # 1 is from ortho
            # 2 is from left
            # 3 is from up
            s = np.array([s1, s2, s3])
            smax = np.max(s)
            sid = np.argmax(s)
            score[i, j] = smax
            strategy[i, j] = sid + 1
    pointsId1 = []
    pointsId2 = []

    ptr1 = n1;
    ptr2 = n2;

    while True:
        sid = strategy[ptr1, ptr2]
        assert sid != 0
        if sid == 1:
            ptr1 -= 1
            ptr2 -= 1
        if sid == 2:
            ptr2 -= 1
        if sid == 3:
            ptr1 -= 1

        pointsId1.append(ptr1)
        pointsId2.append(ptr2)
        if ptr1 == 0 and ptr2 == 0:
            break


    pointsId1 = (np.array(pointsId1[::-1]) + i1) % n1;
    pointsId2 = (np.array(pointsId2[::-1]) + i2) % n2;

    return pointsId1, pointsId2


######################################
# 1 load image

if __name__ == '__main__':
    im1 = cv2.imread('im.png', cv2.IMREAD_UNCHANGED);
    im1, alpha1 = cutallpha(im1, (width, height))

    im2 = cv2.imread('shape.png', cv2.IMREAD_UNCHANGED);
    im2, alpha2 = cutallpha(im2, (width, height))

    cv2.imshow('alpha1', alpha1);
    cv2.imshow('alpha2', alpha2);

    # 3 get countaur, they must be ccw!!!
    contour1_n1x2 = im2countaur(alpha1)
    contour2_n2x2 = im2countaur(alpha2)

    contour1_n1x2 = cv2.approxPolyDP(np.expand_dims(np.round(contour1_n1x2 * height - 0.5).astype(np.int32), 1), 1,
                                     False)[:, 0, :]
    contour2_n2x2 = cv2.approxPolyDP(np.expand_dims(np.round(contour2_n2x2 * height - 0.5).astype(np.int32), 1), 1,
                                     False)[:, 0, :]

    contour1_n1x2 = (contour1_n1x2.astype(np.float32) + 0.5) / height
    contour2_n2x2 = (contour2_n2x2.astype(np.float32) + 0.5) / height

    contour1_n1x2 = pointresamplingnp(contour1_n1x2, 70)
    contour2_n2x2 = pointresamplingnp(contour2_n2x2, 30)
    print(contour1_n1x2.shape, contour2_n2x2.shape)

    imct1 = imdrawcontour2(contour1_n1x2, linecolor=(0, 255, 0), pointcolor=(0, 0, 255))
    imct2 = imdrawcontour2(contour2_n2x2, linecolor=(0, 255, 0), pointcolor=(0, 0, 255))
    cv2.imshow('ct1', imct1);
    cv2.imshow('ct2', imct2);
    cv2.waitKey()

    pointsId1, pointsId2 = matchpoints(contour1_n1x2, contour2_n2x2)

    im = imdrawcontour3(contour1_n1x2, contour2_n2x2, pointsId1, pointsId2, linecolor1=(0, 255, 0),
                        linecolor2=(0, 0, 255), linecolor3=(0, 255, 255))

    cv2.imshow('match', im)
    cv2.waitKey()
