import os
import sys
import argparse
import yaml
from pathlib import Path
import open3d as o3d
import numpy as np
from datetime import datetime

def showPreprocessingResult(config_path):  
    #Paths of corresponding outputs
    cover_pcd_path = config_path.replace("config.yml", "01_cover_pcd.ply")
    cover_mesh_path = config_path.replace("config.yml", "01_cover_mesh.ply")
    inner_pcd_path = config_path.replace("config.yml", "01_inner_pcd.ply")

    cover_pcd = o3d.io.read_point_cloud(cover_pcd_path)
    cover_mesh = o3d.io.read_triangle_mesh(cover_mesh_path)
    inner_pcd = o3d.io.read_point_cloud(inner_pcd_path)

    if cover_pcd.is_empty():
        print(f"Error: Could not find {cover_pcd}")
        return
    else:
        print(f"Visualization: Cover point cloud")
        o3d.visualization.draw_geometries([cover_pcd],window_name=cover_pcd_path,width=800,height=800)
    
    if cover_mesh.is_empty():
        print(f"Error: Could not find {cover_mesh}")
        return
    else:
        print("Visualization: Cover mesh")
        o3d.visualization.draw_geometries([cover_mesh],window_name=cover_mesh_path,width=800,height=800,mesh_show_wireframe=True, mesh_show_back_face=True)
    
    if inner_pcd.is_empty():
        print(f"Error: Could not find {inner_pcd}")
        return
    else:
        print("Visualization: Inner point cloud")
        inner_pcd.paint_uniform_color([1, 1, 0])
        o3d.visualization.draw_geometries([inner_pcd],window_name=cover_pcd_path,width=800,height=800)

    print("Visualization: PreProcessing Results")
    o3d.visualization.draw_geometries([cover_pcd, cover_mesh, inner_pcd],window_name="PreProcessing Results",width=800, height=800,mesh_show_wireframe=True,mesh_show_back_face=True)

def showSegmentationResult(config_path):
    #Path of corresponding output
    config_data = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    #segmented_pcd_path = config_path.replace("config.yml", "02_segmented_pcd.ply")
    segmented_pcd_path = config_path.replace("config.yml", config_data['Segmentation']['segments_pcd'])
    segmented_pcd = o3d.io.read_point_cloud(segmented_pcd_path)

    if segmented_pcd.is_empty():
        print(f"Error: Could not find {segmented_pcd_path}")
        return
    else:
        print("Visualization: Segmented point cloud")
        o3d.visualization.draw_geometries([segmented_pcd],window_name=segmented_pcd_path,width=800,height=800)

    #split_pcd_path = config_path.replace("config.yml", "02_split_segments_pcd.ply")
    split_pcd_path = config_path.replace("config.yml", config_data['splitSegments']['split_segments_pcd'])    
    split_pcd = o3d.io.read_point_cloud(split_pcd_path)

    if split_pcd.is_empty():
        print(f"Error: Could not find {split_pcd_path}")
        return
    else:
        print("Visualization: After split point cloud")
        o3d.visualization.draw_geometries([split_pcd],window_name=split_pcd_path,width=800,height=800)


def showBeamModelingResult(config_path):
    #Paths of corresponding outputs
    config_data = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    #beams_pcd_path = config_path.replace("config.yml", "03_beams_pcd.ply")
    #beams_mesh_path = config_path.replace("config.yml", "03_beams_mesh.ply")
    beams_pcd_path = config_path.replace("config.yml", config_data['beamModeling']['beams_pcd'])
    beams_mesh_path = config_path.replace("config.yml", config_data['beamModeling']['beams_mesh'])

    beams_pcd = o3d.io.read_point_cloud(beams_pcd_path)
    beams_mesh = o3d.io.read_triangle_mesh(beams_mesh_path)

    if beams_pcd.is_empty():
        print(f"Error: Could not find {beams_pcd_path}")
        return
    else:
        print(f"Visualization: Beams point cloud")
        o3d.visualization.draw_geometries([beams_pcd],window_name=beams_pcd_path,width=800,height=800)
    
    if beams_mesh.is_empty():
        print(f"Error: Could not find {beams_mesh_path}")
        return
    else:
        print("Visualization: Beams")
        o3d.visualization.draw_geometries([beams_mesh],window_name=beams_mesh_path,width=800,height=800,mesh_show_wireframe=True, mesh_show_back_face=True)
        
    if not beams_pcd.is_empty() and not beams_mesh.is_empty():
        print("Visualization: Beam Modeling Results")
        o3d.visualization.draw_geometries([beams_pcd, beams_mesh],window_name="Beam Modeling Results",width=800, height=800,mesh_show_wireframe=True,mesh_show_back_face=True)

def main():
    #List of Processing stages
    processing_stages = ["PreProcessing", "Segmentation", "BeamModeling"]
    config_path = sys.argv[1]
    picked_stage = sys.argv[2]

    if picked_stage not in processing_stages:
        print(f"Processing stage: {picked_stage} is not valid")
    else:
        for stage in processing_stages:
            if picked_stage == processing_stages[0]:
                showPreprocessingResult(config_path)
                break
            if picked_stage == processing_stages[1]:
                showSegmentationResult(config_path)
                break
            if picked_stage == processing_stages[2]:
                showBeamModelingResult(config_path)
                break

if __name__ == "__main__":
    main()