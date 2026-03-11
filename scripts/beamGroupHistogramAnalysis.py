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
    start = datetime.now()
    #Read config parameters
    config_data = exchange.readConfig(config_path)

    #Read point cloud
    points, normals, segments = exchange.readODM(config_data['db_odm1']) # Read segmented point cloud (before split)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    #Fetch data from beam_new table
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


    beam_records_old = roof_db.getBeams(["id", "roof_tile_id","group_id", "cluster_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"])#, "roof_tile_id is not null")

    beams_db_old = [Beam.Beam(id=r['id'], roof_tile_id = r['roof_tile_id'], group_id=r['group_id'], cluster_id = r['cluster_id'],
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]]) for r in beam_records_old]


    #Visualize the CAD beams
    roof_tile_beams = [b for b in beams_db_old if b.roof_tile_id is not None]
    roof_tile_obbs = [b.setOBB() for b in roof_tile_beams]
    for obb in roof_tile_obbs:
        obb.color = [1,0,0]
    not_roof_tile_beams = [b for b in beams_db_old if b.roof_tile_id is None]
    not_roof_tile_obbs = [b.setOBB() for b in not_roof_tile_beams]

    roof_tile_beams_new = [b for b in beams_new_db]
    roof_tile_obbs_new = [b.setOBB() for b in roof_tile_beams_new]
    for obb in roof_tile_obbs_new:
        obb.color = [0,0,1]

    o3d.visualization.draw([*roof_tile_obbs, *not_roof_tile_obbs, *roof_tile_obbs_new])
    #o3d.visualization.draw(not_roof_tile_obbs)

    #Search for missing rafter connections
    roof_tile_records = roof_db.getRoofTiles()
    roof_tiles_db =  [RoofTile.RoofTile(id=r['id'], plane=[float(r['plane_a']), float(r['plane_b']), float(r['plane_c']), float(r['plane_d'])], alpha_shape2d = None) for r in roof_tile_records]
    roofTile = [t for t in roof_tiles_db if t.id == 3][0]


    old_beams = [b for b in beams_db_old if b.roof_tile_id == roofTile.id]
    old_obbs = [b.setOBB() for b in old_beams]

    for obb in old_obbs:
        obb.color = [1,0,0]

    group_beams = [b for b in beams_new_db if b.roof_tile_id == roofTile.id]


    group_obbs = [b.setOBB() for b in group_beams]

    for obb in group_obbs:
        obb.color = [1,1,0]

    vert1 = [b.vertices for b in beams_new_db if b.roof_tile_id == roofTile.id]
    pts1 = np.vstack(vert1)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    group_obb =pcd1.get_oriented_bounding_box()

    group_pcd = template.getPointsInBox(pcd, group_obb)

    o3d.visualization.draw([group_pcd, *old_obbs, *group_obbs])

    pts_2d= geometry.project3DPointsToPlane2D(np.array(group_pcd.points), roofTile.plane)
    group_rects = [geometry.project3DPointsToPlane2D(b.vertices ,roofTile.plane) for b in group_beams] # This is multi point object including 8 vertices (to get rectangle MBR computation is necessary)
    group_lines = [geometry.project3DPointsToPlane2D(b.axis ,roofTile.plane) for b in group_beams]
    img, img_size, img_ext = imagePrc.getImageFromPoints(pts_2d, scale = 1)

    #Modeled beams as rectangles on image space
    group_rects_image_cs = []
    for rect in group_rects:
        image_coors = []
        for p in rect:
            image_coors.append(imagePrc.cartesian2ImageCoordinates(p, img_ext[0], img_ext[3], img_size))
        group_rects_image_cs.append(image_coors)

    group_lines_image_cs = []
    for line in group_lines:
        image_coors = []
        for p in line:
            image_coors.append(imagePrc.cartesian2ImageCoordinates(p, img_ext[0], img_ext[3], img_size))
        group_lines_image_cs.append(image_coors)

    binary0 = img <= 0
    binary0 = binary0.astype(np.uint8)  #convert to an unsigned byte
    binary0*=255   
    img = cv2.bitwise_not(binary0) # White pixels are beams

    plt.imshow(img)
    plt.show()

    plt.hist(img.flatten(), bins=None, ec="k")
    #plt.xticks((0,1))
    plt.show()

    #Search for beams on raw image
    lines = imagePrc.getLineSegments(img) # get oriented lines of beam sides (left of line is beam surface)
    rect_mpts, boxes = imagePrc.linesToRects(img, lines) # line matches to beam surface rectangles

    
    #Show results for extracted line / rectangles
    test_image = cv2.merge((img,img,img))
    for pts in rect_mpts:
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(test_image, [box],0,(0,255,0),2)
        #hull = cv2.convexHull(np.array(pts,dtype='float32'))
        #hull = [np.array(hull).reshape((-1,1,2)).astype(np.int32)]       
        #cv2.drawContours(test_image, contours=hull, 
        #                 contourIdx = 0,                     
        #                 color=(55,55,255), thickness=2)
    
    for line in group_lines_image_cs:
        x1,y1 = line[0]
        x2,y2 = line[1]
        cv2.line(test_image, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.imshow("TestRes", test_image)
    cv2.waitKey()

    end = datetime.now()
    print("Analysis end :\t", (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Histogram analysis")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)
