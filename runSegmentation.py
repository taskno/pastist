# =============================================================================
"""
Author  : Taşkın Özkan <taskinozkn@gmail.com>
Version : 0.2.0
Date    : March 2026

Description:
High performed region-growing segmentation application for point clouds.

Usage:
python runSegmentation.py config.yml

Template content for config.yml:
Segmentation:
 max_angle: 5.
 search_radius: 0.05
 min_seg_size: 300
 nr_neighbors: 30
     
"""
# =============================================================================

import os
import argparse
import yaml

import open3d as o3d
import numpy as np
from datetime import datetime
from numba import njit
from scipy.spatial import cKDTree

from toolBox import io

@njit(cache=True)
def compute_labels_numba(points, normals, adj_indices, max_angle_rad, dist_thresh_sq, min_size):
    """
    Compiled BFS Region Growing. 
    Uses machine-code loops and zero Python object overhead.
    """
    n = points.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)
    label_id = 0
    
    # Pre-allocate queue to avoid dynamic resizing
    queue = np.empty(n, dtype=np.int32)
    
    for i in range(n):
        if visited[i]:
            continue
            
        head = 0
        tail = 0
        queue[tail] = i
        tail += 1
        visited[i] = True
        
        # Region Growing BFS
        while head < tail:
            curr = queue[head]
            head += 1
            
            # Neighbors are pre-calculated by Scipy
            for j in range(adj_indices.shape[1]):
                neigh = adj_indices[curr, j]
                
                # -1 check for padding, visited check for performance
                if neigh == -1 or visited[neigh]:
                    continue
                
                # 1. Optimized Angle Check (Dot Product)
                dot = 0.0
                for k in range(3):
                    dot += normals[curr, k] * normals[neigh, k]
                
                # Clamp for stability and compare angle
                abs_dot = abs(dot)
                if abs_dot > 1.0: abs_dot = 1.0
                angle = np.arccos(abs_dot)
                
                if angle < max_angle_rad:
                    # 2. Optimized Distance Check (Squared)
                    d2 = 0.0
                    for k in range(3):
                        d2 += (points[curr, k] - points[neigh, k])**2
                    
                    if d2 < dist_thresh_sq:
                        visited[neigh] = True
                        queue[tail] = neigh
                        tail += 1
        
        # If the discovered cluster meets size requirements, label it
        if tail >= min_size:
            for idx in range(tail):
                labels[queue[idx]] = label_id
            label_id += 1
            
    return labels

def main():
    start_total = datetime.now()

    # --- Load Config params ---
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument('cfg', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()
    config_data = yaml.safe_load(args.cfg)
    
    #Internal parameters:
    #input
    #inner_pcd_path = args.cfg.name.replace("config.yml", "01_inner_pcd.ply")
    inner_pcd_path = args.cfg.name.replace("config.yml", config_data['Segmentation']['point_cloud'])
    max_angle = config_data['Segmentation']['max_angle']
    search_radius = config_data['Segmentation']['search_radius']
    nr_neighbors = config_data['Segmentation']['nr_neighbors']
    min_seg_size = config_data['Segmentation']['min_seg_size']
    #output
    #segmented_pcd_path = args.cfg.name.replace("config.yml", "02_segmented_pcd.ply")
    segmented_pcd_path = args.cfg.name.replace("config.yml", config_data['Segmentation']['segments_pcd'])

    # --- 1. Load Data ---
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(inner_pcd_path)
    if pcd.is_empty():
        print("Error: Could not find %s"%inner_pcd_path)
        return
    else:
        print(f"Total points: {len(pcd.points)}")

    # --- 2. Normal Estimation ---
    print("Estimating normals...")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # --- 3. Parallel Neighbor Search ---
    print("Querying Neighbors via KDTree ...")
    tree_start = datetime.now()
    tree = cKDTree(points)
    # k=30 to match your original search range
    _, adj_indices = tree.query(points, k=nr_neighbors, workers=-1)
    print(f"KDTree Query took: {str(datetime.now() - tree_start)[:-4]}")

    # --- 4. Numba Compiled Clustering ---
    print("Running Region Growing...")
    cluster_start = datetime.now()
    labels = compute_labels_numba(
        points, 
        normals, 
        adj_indices.astype(np.int32), 
        max_angle_rad=np.deg2rad(max_angle), 
        dist_thresh_sq=search_radius**2, # Squared threshold for speed
        min_size=min_seg_size
    )
    print(f"Clustering took: {str(datetime.now() - cluster_start)[:-4]}")

    # --- 5. Optimized Vectorized Coloring ---
    print("Assigning colors...")
    max_label = labels.max()
    num_clusters = max_label + 1
    
    # Generate random colors for each cluster + 1 for noise (-1)
    # The -1 label will automatically index the last row in this array
    palette = np.random.uniform(0.1, 1.0, (num_clusters + 1, 3))
    palette[-1] = [0, 0, 0]  # Set noise to Black 
    pcd.colors = o3d.utility.Vector3dVector(palette[labels])

    # --- 6. Output ---
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    colors = palette[labels] * 255
    io.writePLY(segmented_pcd_path, points, normals, colors, labels)
    
    unique_found = len(np.unique(labels[labels != -1]))
    print("-" * 40)
    print(f"Segments found: {unique_found}")
    print(f"Total Process Time: {str(datetime.now() - start_total)[:-4]}")
    print("-" * 40)

if __name__ == "__main__":
    main()