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

def createNewBeam(roof_db, beam1):
    roof_db.fillBeamNewTable([beam1])
    roof_db.connect(True)
    update_str = "update beam_new set old_id = -1, cluster_id = " + str(beam1.cluster_id) + " where comment = 'manuel_create' and cluster_id is null"
    roof_db.cursor.execute(update_str)
    roof_db.closeSession()

def updateExistingBeam(roof_db, beam):
    roof_db.connect(True)
    values = (             "'"+ beam.comment + "'",
                           beam.unit_vector[0], beam.unit_vector[1], beam.unit_vector[2],
                           beam.width, beam.height, beam.length,
                    "'POINT Z (" + str(beam.vertices[0][0]) + " " + str(beam.vertices[0][1]) + " " + str(beam.vertices[0][2]) + ")'::geometry",
                    "'POINT Z (" + str(beam.vertices[1][0]) + " " + str(beam.vertices[1][1]) + " " + str(beam.vertices[1][2]) + ")'::geometry",
                    "'POINT Z (" + str(beam.vertices[2][0]) + " " + str(beam.vertices[2][1]) + " " + str(beam.vertices[2][2]) + ")'::geometry", 
                    "'POINT Z (" + str(beam.vertices[3][0]) + " " + str(beam.vertices[3][1]) + " " + str(beam.vertices[3][2]) + ")'::geometry", 
                    "'POINT Z (" + str(beam.vertices[4][0]) + " " + str(beam.vertices[4][1]) + " " + str(beam.vertices[4][2]) + ")'::geometry",
                    "'POINT Z (" + str(beam.vertices[5][0]) + " " + str(beam.vertices[5][1]) + " " + str(beam.vertices[5][2]) + ")'::geometry",
                    "'POINT Z (" + str(beam.vertices[6][0]) + " " + str(beam.vertices[6][1]) + " " + str(beam.vertices[6][2]) + ")'::geometry",
                    "'POINT Z (" + str(beam.vertices[7][0]) + " " + str(beam.vertices[7][1]) + " " + str(beam.vertices[7][2]) + ")'::geometry",
                    "'POINT Z (" + str(beam.axis[0][0]) + " " + str(beam.axis[0][1]) + " " + str(beam.axis[0][2]) + ")'::geometry",
                    "'POINT Z (" + str(beam.axis[1][0]) + " " + str(beam.axis[1][1]) + " " + str(beam.axis[1][2]) + ")'::geometry"
                    )

    update_str = "update beam_new set comment = %s, nx = %s, ny = %s, nz = %s, width = %s, height = %s, length = %s, p0 = %s, p1 = %s, p2 = %s, p3 = %s, p4 = %s, p5 = %s, p6 = %s, p7 = %s, axis_start = %s, axis_end = %s " % values
    update_str += " where id = " + str(beam.id)
    roof_db.cursor.execute(update_str)

def main(config_path):
    start = datetime.now()
    #Read config parameters
    config_data = exchange.readConfig(config_path)

    pcd = exchange.getPCD(config_data['missing_beam'])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=12))

    #Fetch data from beam_new table and roof_tile table
    roof_db = database.modelDatabase(config_path)
    beam_records = roof_db.getNewBeams(["id", "roof_tile_id","group_id", "cluster_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"], "roof_tile_id is not null")

    beams_new_db = [Beam.Beam(id=r['id'], roof_tile_id = r['roof_tile_id'], group_id=r['group_id'], cluster_id = r['cluster_id'],
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]]) for r in beam_records]

    #Search for missing rafter connections
    roof_tile_records = roof_db.getRoofTiles()
    roof_tiles_db =  [RoofTile.RoofTile(id=r['id'], plane=[float(r['plane_a']), float(r['plane_b']), float(r['plane_c']), float(r['plane_d'])], alpha_shape2d = None) for r in roof_tile_records]

    beam_group_records = roof_db.getBeamGroups()
      
    #Beam detection
    cuboid, obb1 = iterativeBeamModeling(pcd,voxel_size=0.05)
    beam1 = Beam.obb2Beam(obb1)
    o3d.visualization.draw([pcd, obb1])

    angles = np.array([geometry.getAngleBetweenVectors(beam1.unit_vector,
                                              (float(rec['nx_avg']),
                                               float(rec['ny_avg']),
                                               float(rec['nz_avg'])))
              for rec in  beam_group_records])

    distances = np.array([abs(geometry.getPoint2PlaneDistance(obb1.center, 
                                                 (float(rec['plane_a']),
                                                  float(rec['plane_b']),
                                                  float(rec['plane_c']),
                                                  float(rec['plane_d']))))
                 for rec in  beam_group_records])


    gr_candidate1 = beam_group_records[np.argmin(angles)]['id']
    gr_candidate2 = beam_group_records[np.argmin(distances)]['id']

    tmp_beam = None
    tmp_w = None
    tmp_h = None
    if gr_candidate1 == gr_candidate1:
        beam1.group_id = gr_candidate1
        for g in beam_group_records:
            if g['id'] == gr_candidate1 and g['name'] == "roof_tile":
                beam1.roof_tile_id = gr_candidate1
                tmp_w = g['width_avg']
                tmp_h = g['height_avg']
                break

        for b in beams_new_db:
            if b.group_id == gr_candidate1:
                beam1.cluster_id = b.cluster_id
                tmp_beam = b.setOBB()
                break
    else:
        print ("TODO: investigate this case!")
        sys.exit(0)
    
    beam1.comment = "manual_create"
    beam1.old_id = -1

    #Refine beam1 using the template size of the group
    #box = pcd.get_oriented_bounding_box()
    #obb2 = template.getBeamInSearchBox(pcd, box, tmp_beam, template_len= beam1.obb.extent[0], 
    #                                  target_dims=(tmp_w, tmp_h), vis=False)  
    #obb2 = o3d.geometry.OrientedBoundingBox(beam1.obb.center, beam1.obb.R, 
    #                                                (beam1.obb.extent[0], tmp_beam.extent[1], tmp_beam.extent[2]))
    #o3d.visualization.draw([pcd, obb1, obb2])


    # New beam -> group beams comparison (if beam exists)
    # mesh intersection check
    beam1.obb.color = [1.,0.,0.]
    beams_gr = [b for b in beams_new_db if b.group_id == gr_candidate1]    
    beams_gr_obb = [b.setOBB() for b in beams_gr]

    if config_data['show_missing']:
        o3d.visualization.draw([beam1.obb,*beams_gr_obb])
        print("Keep going ? (y/n)")
        keep_going = str(input())
        if keep_going != "y":
            print("Process stops.")
            sys.exit(0)
        else:
            print("Red colored beam goes to database")

    beams_gr_mesh = [exchange.obb2Mesh(b) for b in beams_gr_obb]
    beam1_mesh = exchange.obb2Mesh(beam1.obb)

    intersecting_beam = None
    for i,b in enumerate(beams_gr_mesh):
        if beam1_mesh.is_intersecting(b):
            intersecting_beam = beams_gr[i]
            break
    if intersecting_beam is None:
        createNewBeam(roof_db, beam1)
        print("Beam inserted to the database.\n")
    else:
        # Extend or re-modeling case
        beam1.id = intersecting_beam.id
        intersecting_beam.extendAlongLongitudinalAxis(beam1.axis[0], beam1.axis[1], True)
        intersecting_beam.comment = 'manual_update'
        intersecting_beam.obb.color = [1.,1.,0.] # Yellow
        #updateExistingBeam(roof_db, intersecting_beam)
        
        o3d.visualization.draw([pcd, beam1.obb, intersecting_beam.obb])
        print("New beam(red) or extended beam(yellow) ? (n/e)")
        answer = str(input())
        if answer == "n":
            print("New beam replaced to database.\n")
            updateExistingBeam(roof_db, beam1)
            sys.exit(0)
        elif answer == "e":
            print("Extended beam updated on database.\n")
            updateExistingBeam(roof_db, intersecting_beam)
        else:
            print("No change on database.\n")
            sys.exit(0)

    #Write beam to beam_new table
    end = datetime.now()
    print("Rafters defined :\t", (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rafter Definition")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)