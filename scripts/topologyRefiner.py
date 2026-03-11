
import sys
import argparse
import yaml
from datetime import datetime
import os
from pathlib import Path
import numpy as np

import toolBox.database as database
import toolBox.exchange as exchange
import toolBox.geometry as geometry
import toolBox.template as template
import toolBox.imagePrc as imagePrc

import roof.RoofTile as RoofTile
import roof.Beam as Beam
import roof.BeamGroup as BeamGroup
import roof.Joint as Joint
import roof.BeamGroup as BeamGroup

from wf2_roofCoverBeams import iterativeBeamModeling

import open3d as o3d
import ezdxf
import alphashape
from shapely import wkb
from shapely.geometry import Point
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import cv2

def main(config_path):

    #Modified version of mesh2Beams.py
    start = datetime.now()
    #Read config parameters
    config_data = exchange.readConfig(config_path)

    #Read point cloud
    #points, normals, segments = exchange.readODM(config_data['db_odm1']) # Read segmented point cloud (before split)
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(points)
    #pcd.normals = o3d.utility.Vector3dVector(normals)

    #Fetch data from beam_new table
    roof_db = database.modelDatabase(config_path)
    beam_records = roof_db.getNewBeams(["id", "roof_tile_id","rafter_id","group_id", "cluster_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"])#, "roof_tile_id is not null")

    beams_new_db = [Beam.Beam(id=r['id'], roof_tile_id = r['roof_tile_id'], rafter_id = r['rafter_id'], group_id=r['group_id'], cluster_id = r['cluster_id'],
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]]) for r in beam_records]


    beam_records_old = roof_db.getBeams(["id", "roof_tile_id","rafter_id","group_id", "cluster_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"])#, "roof_tile_id is not null")

    beams_db_old = [Beam.Beam(id=r['id'], roof_tile_id = r['roof_tile_id'], rafter_id = r['rafter_id'], group_id=r['group_id'], cluster_id = r['cluster_id'],
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]]) for r in beam_records_old]


    #Visualize the CAD beams
    necessary_beams = [b for b in beams_db_old]# if b.roof_tile_id is not None] ## All beams / only covering beams!!
    necessary_beams_obbs = [b.setOBB() for b in necessary_beams]
    for obb in necessary_beams_obbs:
        obb.color = [1,0,0]
    o3d.visualization.draw(necessary_beams_obbs)

    input_mesh = o3d.geometry.TriangleMesh()
    for b in necessary_beams:
        input_mesh += b.getConvexHull3D()
    o3d.io.write_triangle_mesh("beforeRayExtend_tower.ply", input_mesh)

    #Ray Casting part
    cover_hull = o3d.io.read_triangle_mesh(config_data['mesh_of_cover'])
    full_mesh = o3d.io.read_triangle_mesh(config_data['mesh_of_pcd'])

    refined_beams = rayBasedRafterBeamExtension(necessary_beams, cover_hull, full_mesh, 2) 
    refined_beams_obbs = [b.obb for b in refined_beams]
    o3d.visualization.draw(refined_beams_obbs)

    #Save as Mesh
    result_mesh = o3d.geometry.TriangleMesh()
    for b in refined_beams:
        result_mesh += b.getConvexHull3D()
    o3d.io.write_triangle_mesh("afterRayExtend_tower.ply", result_mesh)



    #Step file generation
    labels, cross_sections = clusterCrossSections(refined_beams_obbs)
    pbsBeams = [exchange.obb2PbsBeam(obb) for obb in refined_beams_obbs]

    for i,beam in enumerate(pbsBeams):
        dim_a = cross_sections[labels[i]][0]
        dim_b = cross_sections[labels[i]][1]
        beam.cross_section_class = {'id': labels[i], 'a': dim_a, 'b': dim_b}

    processor = exchange.getBeamProcessor(pbsBeams)
    #processor.automatic_joint_detection() #Postpone it to the end of manual addings
    #processor.export_beams_dxf("hk_roof1_nojo.dxf")#("mk_roof_wing_north_before.dxf")          
    #cs_model = [[7.8740/1000,7.8740/1000]] #20cm as inches
    processor.export_beams_stp("tower_Rayextend.stp",cross_sections)#("mk_roof_wing_north_before.stp", cross_sections)

    end = datetime.now()
    print("Analysis end :\t", (end - start))

def clusterCrossSections(beam_obbs, all_cross_sections = False, rotate = True):
    #Cross section clustering
    from sklearn.cluster import MeanShift
    dims = np.array([np.sort(obb.extent) for obb in beam_obbs])[:,:2]
    if rotate:
        dims = np.vstack((dims[:,1], dims[:,0])).transpose()

    if all_cross_sections:
        labels = [i for i,d in enumerate(dims)]
        cross_sections = dims * 39.3700787 /1000
    else:
        ms = MeanShift(bandwidth=0.04, bin_seeding=True)
        ms.fit(dims)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        cross_sections = cluster_centers * 39.3700787 /1000 #inch conversion and 1000 scale factor

    return labels, cross_sections

def rayBasedRafterBeamExtension(beams, cover_hull, full_mesh = None, max_dist= 0.4):
    #Extension of each beam to the convex hull + Mesh
    scene = o3d.t.geometry.RaycastingScene()

    #Include full mesh if exists
    if full_mesh is not None:
        full_mesh = o3d.t.geometry.TriangleMesh.from_legacy(full_mesh)

    #Add the convex hull to the scene
    cover_hull = o3d.t.geometry.TriangleMesh.from_legacy(cover_hull)
    
    hull_id = scene.add_triangles(cover_hull)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ray based extension")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)

