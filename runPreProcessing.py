import os
import argparse
import yaml
from pathlib import Path

import open3d as o3d
import numpy as np
from datetime import datetime
from numba import njit
from scipy.spatial import cKDTree

from toolBox import geometry
from toolBox import io

def getRoofCover(pcd_full, voxel_size, view_positions = ["+z"]):  
    pcd = pcd_full.voxel_down_sample(voxel_size=voxel_size)#0.2   
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))    
    #radius = diameter * 1000
    obb_ = pcd.get_oriented_bounding_box()
    # For parametric camera location
    #Surrounding camera positions
    cameras = []
    if len(view_positions) == 0:
        cameras.append([obb_.center[0], obb_.center[1], obb_.center[2] + diameter]) # +z direction only by default
    else:
        if "-x" in view_positions:
            cameras.append([obb_.center[0]- diameter, obb_.center[1], obb_.center[2]])  #-x direction
        if "+x" in view_positions:
            cameras.append([obb_.center[0]+ diameter, obb_.center[1], obb_.center[2]])  #+x direction
        if "-y" in view_positions:
            cameras.append([obb_.center[0], obb_.center[1] - diameter, obb_.center[2]]) #-y direction
        if "+y" in view_positions:
            cameras.append([obb_.center[0], obb_.center[1] + diameter, obb_.center[2]]) #+y direction
        if "-z" in view_positions:
            cameras.append([obb_.center[0], obb_.center[1], obb_.center[2] - diameter]) #-z direction
        if "+z" in view_positions:
            cameras.append([obb_.center[0], obb_.center[1], obb_.center[2] + diameter]) #+z direction

    pcd_cover = o3d.geometry.PointCloud()

    for cam in cameras:
        _, pt_map = pcd.hidden_point_removal(cam, diameter * 200)
        pcd_cover += pcd.select_by_index(pt_map, invert=False)
         
    #Normal estimation    
    pcd_cover.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=20)) 
    pcd_cover.orient_normals_towards_camera_location(camera_location= pcd_cover.get_center())
    mesh_cover, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_cover, depth=8)
    return pcd_cover, mesh_cover

def getInnerCloud(pcd_full, cover_mesh, dist_thresh=0.15):
    # Convert to tensor-based geometry for raycasting
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(cover_mesh)
    pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_full)

    # Create raycasting scene and add the mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    # Compute unsigned distances from points to the mesh
    distances = scene.compute_distance(pcd_t.point.positions).numpy()

    # Filter points farther than 5cm (0.05m)
    indices = np.where(distances > dist_thresh)[0]

    # Create new point cloud with filtered points
    outlier_pcd = pcd_full.select_by_index(indices)

    return outlier_pcd

def main():
    start_total = datetime.now()
    #project_dir = os.path.abspath(os.getcwd())
    #cwd = os.getcwd()
    #results_dir = os.path.join(cwd, "results")
    
    # --- Load Config params ---
    parser = argparse.ArgumentParser(description="Pre-Processing")
    parser.add_argument('cfg', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()
    config_data = yaml.safe_load(args.cfg)


    #Internal parameters:
    #input
    input_point_cloud = config_data['preProcessing']['point_cloud']
    input_point_cloud = Path(input_point_cloud).resolve()
    roof_cover_voxel_size = config_data['preProcessing']['roof_cover_voxel_size']
    view_positions = config_data['preProcessing']['view_positions']
    inner_point_sampling_size = config_data['preProcessing']['inner_point_sampling_size']
    cover_inner_dist_thresh = config_data['preProcessing']['cover_inner_dist_thresh']

    #output
    pcd_cover_path = args.cfg.name.replace("config.yml", "01_cover_pcd.ply")
    mesh_cover_path = args.cfg.name.replace("config.yml", "01_cover_mesh.ply")
    pcd_inner_path = args.cfg.name.replace("config.yml", "01_inner_pcd.ply")

    # --- 1. Extract Roof Cover ---
    print("Extracting roof cover...")
    start_1 = datetime.now()
    print("Loading point cloud...")
    in_pcd = io.readPointCloud(input_point_cloud)
    if in_pcd.is_empty():
        print("Error: Could not find %s"%input_point_cloud)
        return
    else:
        print(f"Total points: {len(in_pcd.points)}")


    pcd_cover, mesh_cover = getRoofCover(in_pcd, roof_cover_voxel_size, view_positions)
    #pcd_cover_path = os.path.join(results_dir, "cover_pcd.ply")
    #mesh_cover_path = os.path.join(results_dir, "cover_mesh.ply")
    o3d.io.write_point_cloud(pcd_cover_path,pcd_cover)
    o3d.io.write_triangle_mesh(mesh_cover_path, mesh_cover)

    #print("Cover point cloud: ", pcd_cover_path)
    #print("Cover mesh: ", mesh_cover_path)
    print(f"Cover extraction took: {str(datetime.now() - start_1)[:-4]}")

    # --- 2. Extract Inner Points ---
    print("Extracting inner point cloud...")
    start_2 = datetime.now()
    in_pcd = in_pcd.voxel_down_sample(voxel_size=inner_point_sampling_size)
    inner_pcd = getInnerCloud(in_pcd, mesh_cover, cover_inner_dist_thresh)
    #pcd_inner_path = os.path.join(results_dir, "inner_points.ply")
    o3d.io.write_point_cloud(pcd_inner_path, inner_pcd)

    #print("Inner point cloud: ", pcd_inner_path)
    print(f"Cover extraction took: {str(datetime.now() - start_2)[:-4]}")
    print("-" * 40)
    print(f"Total Process Time: {str(datetime.now() - start_total)[:-4]}")
    print("-" * 40)
if __name__ == "__main__":
    main()