from random import sample

import numpy as np


def pt_cloud_to_pillars(pt_cloud, config):
    x_min = config.x_range[0]
    x_max = config.x_range[1]
    y_min = config.y_range[0]
    y_max = config.y_range[1]
    z_min = config.z_range[0]
    z_max = config.z_range[1]
    resolution = config.resolution

    # mask out points out of range
    x_mask = np.logical_and(pt_cloud[:, 0] >= x_min, pt_cloud[:, 0] <= x_max)
    y_mask = np.logical_and(pt_cloud[:, 1] >= y_min, pt_cloud[:, 1] <= y_max)
    z_mask = np.logical_and(pt_cloud[:, 2] >= z_min, pt_cloud[:, 2] <= z_max)
    mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
    pt_cloud = pt_cloud[mask]

    # collect points at each unique index
    pointpillar_map = dict()
    # for x, y, z, r in pt_cloud:
    #     x_idx = int(np.floor((x - x_min) / resolution))
    #     y_idx = int(np.floor((y - y_min) / resolution))

    #     if (x_idx, y_idx) in pointpillar_map.keys():
    #         pointpillar_map[x_idx, y_idx].append([x, y, z, r])
    #     else:
    #         pointpillar_map[x_idx, y_idx] = [[x, y, z, r]]

    x_bins = np.arange(x_min, x_max, resolution)
    x_idx = np.digitize(pt_cloud[:, 0], x_bins) - 1
    y_bins = np.arange(y_min, y_max, resolution)
    y_idx = np.digitize(pt_cloud[:, 1], y_bins) - 1

    for k, (x, y) in enumerate(zip(x_idx, y_idx)):
        if (x, y) in pointpillar_map.keys():
            pointpillar_map[x, y].append(pt_cloud[k, :])
        else:
            pointpillar_map[x, y] = [pt_cloud[k, :]]

    # P = 12k (non-empty pillars per sample)
    # N = 100 (number of points per pillar)

    D, P, N = 4, 12000, 100

    stacked_pillars = np.zeros((D, P, N), dtype=np.float32)
    indices = np.zeros((P, 2), dtype=np.int64)

    # randomly sample P pillars
    if len(pointpillar_map.keys()) > P:
        keys_to_delete = sample(pointpillar_map.keys(), len(pointpillar_map.keys()) - P)
        for k in keys_to_delete:
            pointpillar_map.pop(k)

    for p, (idx, pts) in enumerate(pointpillar_map.items()):
        indices[p, :] = idx

        if len(pts) > N:    # randomly sample N points
            pts = sample(pts, N)

        for n, pt in enumerate(pts):
            stacked_pillars[:, p, n] = pt

    return stacked_pillars, indices


if __name__ == "__main__":
    from argparse import Namespace


    config = Namespace()
    config.x_min = 0.0
    config.x_max = 70.4
    config.y_min = -40.0
    config.y_max = 40.0
    config.z_min = -3.0
    config.z_max = 1.0
    config.resolution = 0.16

    file = "/Users/luisgonzales/Downloads/data_object_velodyne/training/velodyne/000002.bin"
    pt_cloud = np.fromfile(file, dtype=np.float32).reshape((-1, 4))

    stacked_pillars, pillar_indices = pt_cloud_to_pillars(pt_cloud, config)
    print("stacked_pillars:", stacked_pillars.shape)
    print("pillar_indices:", pillar_indices.shape)
