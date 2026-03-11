import os
import argparse
import yaml

import numpy as np
import open3d as o3d
import cv2
from sklearn.decomposition import PCA
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from toolBox import imagePrc
from toolBox import io

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def compute_planarity(points):
    """
    Standard geometric planarity metric.
    Returns value in [0, 1].
    """
    if points.shape[0] < 3:
        return 0.0

    # Center
    pts = points - points.mean(axis=0)

    # Covariance matrix
    cov = np.dot(pts.T, pts) / pts.shape[0]

    # Eigenvalues (ascending order)
    eigvals = np.linalg.eigvalsh(cov)

    # Sort descending: λ1 >= λ2 >= λ3
    l1, l2, l3 = eigvals[::-1]

    if l1 <= 1e-12:
        return 0.0

    planarity = (l2 - l3) / l1
    return planarity

def plane_sigma0_approx(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    return np.linalg.svd(centered, compute_uv=False)[-1] / np.sqrt(len(points) - 3)

def extract_plane(points, distance_thresh=0.01, min_points=100):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    planes = []
    remaining = cloud

    while len(remaining.points) > min_points:
        model, inliers = remaining.segment_plane(
            distance_threshold=distance_thresh,
            ransac_n=3,
            num_iterations=1000
        )

        if len(inliers) < min_points:
            break

        planes.append(np.asarray(remaining.points)[inliers])
        remaining = remaining.select_by_index(inliers, invert=True)

    return planes

def ransac_plane(points, dist_thresh):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    model, inliers = cloud.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=4,
        num_iterations=1000
    )

    return model, np.array(inliers)

def project_to_plane(points, plane):
    n = np.array(plane[:3])
    n /= np.linalg.norm(n)
    d = plane[3]

    dist = np.dot(points, n) + d
    return points - dist[:, None] * n

def region_grow(points, voxel_size=0.05, min_pts=50):
    vox = np.floor(points / voxel_size).astype(np.int32)
    voxel_map = {}

    for i, v in enumerate(map(tuple, vox)):
        voxel_map.setdefault(v, []).append(i)

    visited = set()
    clusters = []

    neighbors = [(i,j,k) for i in (-1,0,1)
                        for j in (-1,0,1)
                        for k in (-1,0,1)]

    for v in voxel_map:
        if v in visited:
            continue

        stack = [v]
        visited.add(v)
        idxs = []

        while stack:
            cur = stack.pop()
            idxs.extend(voxel_map[cur])

            for dv in neighbors:
                nb = (cur[0]+dv[0], cur[1]+dv[1], cur[2]+dv[2])
                if nb in voxel_map and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if len(idxs) >= min_pts:
            clusters.append(points[idxs])

    return clusters

def extract_homogeneous_planes(points, config):
    remaining = points.copy()
    planes = []

    while remaining.shape[0] > config["min_seg_size"]:
        model, inliers = ransac_plane(
            remaining, config["max_plane_dist"]
        )

        if len(inliers) < config["min_seg_size"]:
            break

        inlier_pts = remaining[inliers]
        projected = project_to_plane(inlier_pts, model)

        sub_planes = region_grow(
            projected,
            voxel_size=config["voxel_size"],
            min_pts=config["min_seg_size"]
        )

        planes.extend(sub_planes)

        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]

    return planes

def project_to_image(points, resolution=0.01):
    pca = PCA(n_components=2)
    uv = pca.fit_transform(points)

    uv -= uv.min(axis=0)
    img_size = np.ceil(uv.max(axis=0) / resolution).astype(int) + 1

    img = np.zeros(img_size[::-1], dtype=np.uint8)
    pix = (uv / resolution).astype(int)
    img[pix[:,1], pix[:,0]] = 255

    return img, pix

def filter_points_by_mask(points, pix, mask):
    """
    Keeps 3D points whose 2D projection falls inside mask.
    """
    inside = mask[pix[:,1], pix[:,0]] > 0
    return points[inside.flatten()], inside

def area_mbr_ratio(binary_img):
    ys, xs = np.where(binary_img > 0)
    if len(xs) == 0:
        return 0.0

    area = len(xs)
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    return area / (w * h)

def detect_sub_linear_segments(binary_img, pix, points):
    kernel = np.ones((5,5),np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    lines = imagePrc.getLineSegments(binary_img) # get oriented lines of beam sides (left of line is beam surface)

    if lines is not None and len(lines)> 4:

        img_lines = cv2.merge((binary_img,binary_img,binary_img))
        for line in lines:
            x1, y1, x2, y2 = line[0]
            img_lines = cv2.arrowedLine(img_lines, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)

        rect_mpts, boxes = imagePrc.linesToRects(binary_img, lines)
        
        sub_segments = []
        for box in boxes:
            mask = np.zeros((binary_img.shape[0],binary_img.shape[1],1), np.uint8)
            box = box.astype(np.int32)
            cv2.drawContours(mask, contours=[box], 
                         contourIdx = 0,
                         color=(255), thickness=-1)
        
            pts, inside = filter_points_by_mask(points, pix, mask)
            sub_segments.append(pts)
        
        return sub_segments

def process_segment(points, seg_id, new_id_start, config):
    new_segments = []
    planarity = compute_planarity(points)
    sigma0 = plane_sigma0_approx(points)

    if sigma0 < config["max_plane_dist"] and planarity < config["surface_variation"]:
        img, pix = project_to_image(points, config["resolution"])
        #ratio = area_mbr_ratio(img)
        img_stats = imagePrc.getMBRStats(img)
        ratio = img_stats["fArea"]

        if ratio > config["linear_ratio"]:
            return [(points, seg_id)], new_id_start
        else:
            if np.min(img.shape) > 10:
                sub_segments = detect_sub_linear_segments(img, pix, points)
                if sub_segments:
                    for pts in sub_segments:
                        if len(pts) > config["min_seg_size"]:
                            new_segments.append((pts, new_id_start))
                            new_id_start += 1
                else:
                    new_segments.append((points, new_id_start))
                    new_id_start += 1        
    else:
        planes = extract_homogeneous_planes(points, config)       
        for plane_pts in planes:
            img, pix = project_to_image(plane_pts, config["resolution"])
            #ratio = area_mbr_ratio(img)
            img_stats = imagePrc.getMBRStats(img)
            ratio = img_stats["fArea"]
        
            if ratio > config["linear_ratio"]:
                new_segments.append((plane_pts, new_id_start))
                new_id_start += 1
            else:
                if np.min(img.shape) > 15: # Img min size should be more than 15px
                    sub_segments = detect_sub_linear_segments(img, pix, plane_pts)
                    if sub_segments:
                        for points in sub_segments:
                            if len(points) > config["min_seg_size"]:# Maybe also linearity check!
                                new_segments.append((points, new_id_start))
                                new_id_start += 1
                    else:
                        new_segments.append((plane_pts, new_id_start))
                        new_id_start += 1              
    return new_segments, new_id_start

def refine_segments(points, segment_ids, config):
    result_points = []
    result_ids = []

    next_id = segment_ids.max() + 1

    for seg_id in tqdm(np.unique(segment_ids)):
        mask = segment_ids == seg_id
        seg_points = points[mask]

        segments, next_id = process_segment(
            seg_points, seg_id, next_id, config
        )

        #segments, next_id = process_segment(
        #    points, normals, adj_indices, mask, seg_id, next_id, config
        #)

        for pts, sid in segments:
            result_points.append(pts)
            result_ids.append(np.full(len(pts), sid))

    return (
        np.vstack(result_points),
        np.concatenate(result_ids)
    )

def main():
    start_total = datetime.now()

    # --- Load Config params ---
    parser = argparse.ArgumentParser(description="Split Segments")
    parser.add_argument('cfg', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()
    config_data = yaml.safe_load(args.cfg)
    
    #Internal parameters:
    #input
    #segmented_pcd_path = args.cfg.name.replace("config.yml", "02_segmented_pcd.ply")
    segmented_pcd_path = args.cfg.name.replace("config.yml", config_data['splitSegments']['point_cloud'])
    surface_variation = config_data['splitSegments']['surface_variation']
    max_plane_dist = config_data['splitSegments']['max_plane_dist']
    min_seg_size = config_data['splitSegments']['min_seg_size']
    linear_ratio = config_data['splitSegments']['linear_ratio']
    #output
    #split_pcd_path = args.cfg.name.replace("config.yml", "02_split_segments_pcd.ply")
    split_pcd_path = args.cfg.name.replace("config.yml", config_data['splitSegments']['split_segments_pcd'])

    config = {
    "surface_variation": surface_variation,
    "max_plane_dist": max_plane_dist,
    "min_seg_size": min_seg_size,
    "linear_ratio": linear_ratio,
    "resolution": 0.01,
    "voxel_size": 0.05
    }

    # --- 1. Load Data ---
    print("Loading point cloud...")
    try:
        pcd_data = io.readPLY(segmented_pcd_path)
        print(f"Total points: {len(pcd_data['points'])}")
    except:
        print("Error: Could not find %s"%segmented_pcd_path)
        return

    points = pcd_data['points']
    normals = pcd_data['normals']
    segments = pcd_data['segmentid']

    mask_seg = np.where(segments != -1)

    points = points[mask_seg]
    normals = normals[mask_seg]
    segments = segments[mask_seg]

    # --- 2. Split Non-linear Segments to Linear Parts ---
    print("Split  segments...")
    ref_pts, ref_seg = refine_segments(points, segments, config)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(ref_pts)
    pcd_out.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    ref_nrm = np.asarray(pcd_out.normals)

    max_label = ref_seg.max()
    num_clusters = max_label + 1
    palette = np.random.uniform(0.1, 1.0, (num_clusters + 1, 3))
    palette[-1] = [0, 0, 0]  # Set noise to Black
    colors = palette[ref_seg] * 255

    io.writePLY(split_pcd_path, ref_pts, ref_nrm, colors, ref_seg)

    unique_found = len(np.unique(ref_seg[ref_seg != -1]))   
    print("-" * 40)
    print(f"Segments found: {unique_found}")
    print(f"Total Process Time: {str(datetime.now() - start_total)[:-4]}")
    print("-" * 40)

if __name__ == "__main__":
    main()