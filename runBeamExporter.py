import sys
import argparse
import yaml
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import copy
import open3d as o3d
from sklearn.cluster import MeanShift
import toolBox.exchange as exchange
import components.Beam as Beam

def clusterCrossSections(beam_obbs, all_cross_sections = False, rotate = True, inch_unit=True):
    #Cross section clustering
    if inch_unit:
        inch_factor = 39.3700787 /1000 #inch conversion and 1000 scale factor
    else:
        inch_factor = 1.

    dims = np.array([np.sort(obb.extent) for obb in beam_obbs])[:,:2]
    if rotate:
        dims = np.vstack((dims[:,1], dims[:,0])).transpose()

    if all_cross_sections:
        labels = [i for i,d in enumerate(dims)]
        cross_sections = dims * inch_factor
    else:
        ms = MeanShift(bandwidth=0.04, bin_seeding=True)
        ms.fit(dims)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        cross_sections = cluster_centers * inch_factor
        print("Cross-sections clustered:")
        print(f"{cross_sections}")

    return labels, cross_sections

def rayBasedBeamExtension(beams, cover_hull, full_mesh = None, max_dist= 0.4):
    #Extension of each beam to the convex hull + Mesh
    scene = o3d.t.geometry.RaycastingScene()

    #Include full mesh if exists
    if full_mesh is not None:
        full_mesh = o3d.t.geometry.TriangleMesh.from_legacy(full_mesh)

    #Add the convex hull to the scene
    cover_hull = o3d.t.geometry.TriangleMesh.from_legacy(cover_hull)
    
    hull_id = scene.add_triangles(cover_hull)
    if full_mesh is not None:
        mesh_id = scene.add_triangles(full_mesh)

    #Add all beams to the scene
    for b in beams:
        b_hull = b.getConvexHull3D()
        b_hull.scale(0.99, center=b_hull.get_center())
        b_hull = o3d.t.geometry.TriangleMesh.from_legacy(b_hull)
        hull_id = scene.add_triangles(b_hull)

    #Create rays to cast:
    #Ray (x,y,z,nx,ny,nz) 
    #-> for top :    b.axis[1], b.unit_vector
    #-> for bottom : b.axis[0], -b.unit_vector
    top_ray_array = [np.hstack((b.axis[1],b.unit_vector)) for b in beams]
    bot_ray_array = [np.hstack((b.axis[0],-b.unit_vector)) for b in beams]

    top_rays = o3d.core.Tensor(top_ray_array, dtype=o3d.core.Dtype.Float32)
    bot_rays = o3d.core.Tensor(bot_ray_array, dtype=o3d.core.Dtype.Float32)

    ans_top = scene.cast_rays(top_rays)
    dist_top = ans_top['t_hit'].numpy()
    ans_bot = scene.cast_rays(bot_rays)
    dist_bot = ans_bot['t_hit'].numpy()

    for i,b in enumerate(beams):
        if  dist_top[i] != float('inf') and dist_top[i] < max_dist:
            p1 = b.axis[1] + dist_top[i] * b.unit_vector
        else:
            p1= None
        if dist_bot[i] != float('inf') and dist_bot[i] < max_dist:
            p2 = b.axis[0] + dist_bot[i] * -b.unit_vector
        else:
            p2 = None
        if not (p1 is None and p2 is None):
            b.extendAlongLongitudinalAxis(p1, p2)
            b.obb.color = [1,1,0]
            b_hull = b.getConvexHull3D()
            b_hull.scale(0.99, center=b_hull.get_center())
            b_hull = o3d.t.geometry.TriangleMesh.from_legacy(b_hull)
            hull_id = scene.add_triangles(b_hull)
    
    ext_obbs = [b.obb for b in beams] 
    o3d.visualization.draw([cover_hull, full_mesh, *ext_obbs])

    return beams

def getPCD2MeshDist(pcd, mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh) 
    query_points = o3d.core.Tensor(np.array(pcd.points), dtype=o3d.core.Dtype.Float32) 
    unsigned_distance = scene.compute_distance(query_points)
    return unsigned_distance.numpy()

def getPointsInBox(pcd, obb):
    idx = obb.get_point_indices_within_bounding_box(pcd.points)   
    pts_in_box = pcd.select_by_index(idx, invert=False)
    return pts_in_box

def fineICP(pcd, beam):
     #####  
     tmp_obb = copy.deepcopy(beam.obb)
     tmp_sample = template.sampleOBB(tmp_obb, np.max(beam.obb.extent) * 1000)
     tmp_sbox = copy.deepcopy(tmp_obb)
     search_box = tmp_sbox.scale(2., tmp_sbox.center)
     pcd_sub = getPointsInBox(pcd, search_box)

     if len(pcd_sub.points) < 100:
         return None

     pcd_sub.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=8))
     
     init = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
     icp_transform = o3d.pipelines.registration.registration_icp(
         tmp_sample, pcd_sub, 0.2,init,
         o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))  
     tmp_sample2 = copy.deepcopy(tmp_sample)
     tmp_sample2 = tmp_sample2.transform(icp_transform.transformation)
     tmp_sample2.paint_uniform_color([1,0,0])
     box_pts = o3d.geometry.PointCloud()
     box_pts.points = o3d.utility.Vector3dVector(tmp_obb.get_box_points())                    
     box_pts = box_pts.transform(icp_transform.transformation)
     tmp_obb2 = box_pts.get_oriented_bounding_box() #this is transformed obb
     o3d.visualization.draw([pcd_sub, tmp_sample, tmp_sample2, tmp_obb2])

def main():
    start_total = datetime.now()

    # --- Load Config params ---
    parser = argparse.ArgumentParser(description="Beam Modeling")
    parser.add_argument('cfg', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()
    config_data = yaml.safe_load(args.cfg)
    
    #Internal parameters:
    #input
    input_mesh_path = args.cfg.name.replace("config.yml", config_data['beamExporter']['mesh'])
    formats = config_data['beamExporter']['formats']
    joint_detection = config_data['beamExporter']['joint_detection']
    max_joint_size = config_data['beamExporter']['max_joint_size']
    cross_sections_as_inch = config_data['beamExporter']['cross_sections_as_inch']
    cluster_cross_sections = config_data['beamExporter']['cluster_cross_sections']
    material = config_data['beamExporter']['material']
    
    #outputs
    out_file_name = args.cfg.name.replace("config.yml", config_data['beamExporter']['out_file_name'])

    if "stp" not in formats and "dxf" not in formats:
        print("At least one of the valid formats required")
        print("Valid formats: dxf, stp")
        return
    if "dxf" in formats:
        out_file_name_dxf = out_file_name + ".dxf"
        if joint_detection:
            out_file_name_jo_dxf = out_file_name +"_joints.dxf"
    if "stp" in formats:
        out_file_name_stp = out_file_name + ".stp"
        if joint_detection:
            out_file_name_jo_stp = out_file_name + "_joints.stp"
           
    # --- 1. Load Data ---
    print("Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    print(f"Mesh has {len(mesh.triangles)} triangles")


    beam_obbs = exchange.mesh2OBBs(mesh)
    #beams = [Beam.obb2Beam(b) for b in beam_obbs]
    #refined_beams_obbs = [b.obb for b in beams]

    # --- 2. Exporting Beams ---

    print(f"Exporting {len(beam_obbs)} beams...")
    export_starts = datetime.now()
    all_cs = not cluster_cross_sections
    labels, cross_sections = clusterCrossSections(beam_obbs, all_cross_sections=all_cs, rotate=True, inch_unit=cross_sections_as_inch)
    pbsBeams = [exchange.obb2PbsBeam(obb) for obb in beam_obbs if min(obb.extent)> 0.02]

    for i,beam in enumerate(pbsBeams):
        dim_a = cross_sections[labels[i]][0]
        dim_b = cross_sections[labels[i]][1]
        beam.cross_section_class = {'id': labels[i], 'a': dim_a, 'b': dim_b}

    processor = exchange.getBeamProcessor(pbsBeams)
    processor.PBS_GUI._MATERIALS = [material]
    processor.PBS_GUI._MAX_JOINT_LEN = max_joint_size

    if "dxf" in formats:
        processor.export_beams_dxf(out_file_name_dxf)
        print("dxf file with beams saved.")
    if "stp" in formats:
        processor.export_beams_stp(out_file_name_stp, cross_sections)
        print("stp file with beams saved.")

    print(f"Beam export took: {str(datetime.now() - export_starts)[:-4]}")

    if joint_detection:
        print("Joint detection...")
        joint_detection_starts = datetime.now()
        processor.automatic_joint_detection()
        print(f"Joint detection took: {str(datetime.now() - joint_detection_starts)[:-4]}")   

        if "dxf" in formats:
            processor.export_beams_dxf(out_file_name_jo_dxf)
            print("dxf file with beams and joints saved.")
        if "stp" in formats:
            processor.export_beams_stp(out_file_name_jo_stp, cross_sections)
            print("stp file with beams and joints saved.")
   
    print("-" * 40)
    print(f"Total Process Time: {str(datetime.now() - start_total)[:-4]}")
    print("-" * 40)


if __name__ == '__main__':
    main()