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
import toolBox.imagePrc as imagePrc
import toolBox.optimize as optimize
import toolBox.template as template

import roof.RoofTile as RoofTile
import roof.Beam as Beam
import roof.BeamGroup as BeamGroup

import open3d as o3d
import ezdxf
from ezdxf.addons import r12writer
import alphashape

from shapely import wkb
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import linemerge
from sklearn.cluster import KMeans, DBSCAN, MeanShift

import external.pbs_beam
from enum import Enum

import math
from CGAL.CGAL_Kernel import Point_2, Point_3, Plane_3, Vector_3
from CGAL.CGAL_Kernel import Weighted_point_2
from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2
from CGAL.CGAL_Alpha_shape_2 import Weighted_alpha_shape_2
from CGAL.CGAL_Alpha_shape_2 import Weighted_alpha_shape_2_Face_handle
from CGAL.CGAL_Alpha_shape_2 import GENERAL, EXTERIOR, SINGULAR, REGULAR, INTERIOR
from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2_Vertex_handle
from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2_Face_handle
from CGAL.CGAL_Alpha_shape_2 import Face_Interval_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Shape_detection import *

from matplotlib import pyplot as plt


class BEAM_SIDE(Enum):
    BASE = 0
    RIGHT = 1
    LEFT = 2
    OPPOSITE = 3

def getOBBAxis(obb):
    all_pts = np.array(obb.get_box_points())
    d = [geometry.getDistance(all_pts[0], p) for p in all_pts]
    idx = np.argsort(d)
    v1 = all_pts[idx[0:4]]
    v2 = all_pts[idx[4:]]
    return np.array((np.mean(v1, axis=0), np.mean(v2, axis=0)))

def shapeClassification(fElong, fArea, fBorder, rmse):
    #Code from pbs_auto
    #Shape classification decision tree
    type = None
    if rmse > 0.04:
        # Planar sub -segmentation case
        #print("RANSAC/ planeSearch here")
        type = "d" # non-Planar segment! needs to be separated (type in (a, b, c) after segmentation!!)

    else:
        if fElong > 5 and fArea > 0.5:
            #if fArea > 0.5:
            type = "a" #Linear segment
            #else:
            #   if fBorder < 0.1:
            #        type = "a" #Linear segment
        elif fElong > 5 and fArea <= 0.5 and fBorder < 0.1:
            type = "a"
        elif fElong < 4.5 and fArea > 0.8:
            type = "c" #Compact segment
        else:
            type = "b" #Splittable segment
    return type

def getAlphaShapeVertices(alphaShapePoly):   
    #Code from pbs_auto
    x, y = alphaShapePoly.exterior.coords.xy

    if len(alphaShapePoly.interiors) > 0:
        for inner in alphaShapePoly.interiors:
            xi, yi = inner.coords.xy
            x = np.hstack((x, xi))
            y = np.hstack((y, yi))
    else:
        x = np.asarray(x)
        y= np.asarray(y)

    return (x, y)

def findAlphaShape(segmentPoints, mode = "CGAL"):
    #Code from pbs_auto
    plane_fit,ev,ew = geometry.getPlaneLS(np.array(segmentPoints))
    #plane_fit = fitPlaneLS(segmentPoints)
    
    segPlane = Plane_3(plane_fit[0], plane_fit[1], plane_fit[2], plane_fit[3]) # CGAL plane object handle
    if mode == "geometryProcess":
        pts2D = geometry.project3DPointsToPlane2D(np.array(segmentPoints),plane_fit)
        cgalPoints2dP = [Point_2(p[0], p[1]) for p in pts2D]

    else:
        #Conversion optimisation
        cgalPoints3d = [Point_3(item[0],item[1], item[2]) for item in segmentPoints]
        cgalPoints3dP = [segPlane.projection(item) for item in cgalPoints3d]
        cgalPoints2dP = [segPlane.to_2d(item) for item in cgalPoints3dP]
        pts2D = [(float(pt.x()), float(pt.y())) for pt in cgalPoints2dP]

    # Alpha Shape Extraction
    a = Alpha_shape_2()
    a.set_mode(REGULAR)
    a.make_alpha_shape(cgalPoints2dP)
    optimal_alpha = a.find_optimal_alpha(1)#(len(cgalPoints2dP))
    alpha_val = optimal_alpha.next()
    if alpha_val < 0.001:
        alpha_val = 0.001
    a.set_alpha(alpha_val) #(0.001) TODO: alpha value can be calculated automatically, Development comes    
    #it = a.find_optimal_alpha(1)
    #optimal_alpha = it.next()    
    #a.set_alpha(optimal_alpha)

    geometry_valid = False
    max_iter = 5
    iter = 0

    while not geometry_valid and iter < max_iter:
        iter+=1
        lines = []
        alpha_shape_edges = [] # Line list comes here
        alpha_shape_vertices = [] # Ordered point list comes here
        simplifedPts = []

        a_lines =[LineString([[a.segment(it).point(0).x(), a.segment(it).point(0).y()], [a.segment(it).point(1).x(), a.segment(it).point(1).y()]]) for it in  a.alpha_shape_edges()]
        multi_line = MultiLineString(a_lines) # Lines to multi-line(s)
        merged_line = linemerge(multi_line) # handle merged multi-line
        
        line_str = []
        #line_str_len = []
        if not merged_line.geom_type == 'LineString':
            #for line in merged_line:
                #line_str.append(line)
                #line_str_len.append(line.length)
            line_str = [(line, line.length) for line in merged_line]
                    
        wkt = merged_line.wkt
        if len(line_str) > 1:
            #poly = Polygon(line_str[4]) # 106
            #poly = Polygon(line_str[2]) # 6
            #indices = heapq.nlargest(2, range(len(line_str_len)), key=line_str_len.__getitem__)
            
            idMax = np.asarray(list(zip(*line_str))[1]).argmax()
            outer = Polygon(line_str[idMax][0])
            #hole = Polygon(line_str[indices[1]])
            line_str.pop(idMax)
            holes = [Polygon(h[0]) for h in line_str if h[0].convex_hull.geom_type == "Polygon"]
        
            poly = Polygon(outer.exterior.coords, [hole.exterior.coords for hole in holes])
        else:
            poly = Polygon(merged_line) # Multi polygon development should come here!
            hole = None
            holes = None
        geometry_valid = poly.is_valid
        if not poly.is_valid:
            alpha_val *= 2
            a.set_alpha(alpha_val)
    

    #mpoly = MultiPolygon(merged_line) # TODO multipolygon and alpha value optimization
    rect = poly.minimum_rotated_rectangle

    # get coordinates of polygon vertices
    x_, y_ = rect.exterior.coords.xy
    
    # get length of bounding box edges
    edge_length = (Point(x_[0], y_[0]).distance(Point(x_[1], y_[1])), Point(x_[1], y_[1]).distance(Point(x_[2], y_[2])))
    
    # get length of polygon as the longest edge of the bounding box
    length = max(edge_length)
    
    # get width of polygon as the shortest edge of the bounding box
    width = min(edge_length)

    polyArea = poly.area
    mbrArea = rect.area
    
    #Compute shape factors
    fArea = polyArea / mbrArea
    fElong = math.sqrt(ew[2] / ew[1])
    fBorder = 0

    if fElong > 5 and fArea <= 0.5 and width > 0.09:
        innerRect = Polygon(rect.exterior.parallel_offset(0.04, "left", 0, 2, 10))  
        x_a,y_a = getAlphaShapeVertices(poly)
        alphaPts = np.vstack((x_a,y_a)).transpose()
        multiPts = MultiPoint(alphaPts)
        innerRectPts = innerRect.intersection(multiPts)
        if len(multiPts) > 0:
            if innerRectPts.geom_type == 'Point':
                fBorder = 1. / len(multiPts) #TODO chech lengts here!!
            else:
                fBorder = len(innerRectPts)/len(multiPts) #TODO chech lengts here!!
    
    shape = {}
    shape["pts2D"] = pts2D
    shape["poly"] = poly
    shape["plane"] = plane_fit
    shape["mbrArea"] = mbrArea
    shape["fArea"] = fArea
    shape["fElong"] = fElong
    shape["fBorder"] = fBorder
    shape["rmse"] = plane_fit[4]
    shape["eigenVecs"] = ev
    shape["shape_class"] = shapeClassification(fElong, fArea, fBorder, plane_fit[4])

    return shape

def beamDetectionSingle(pts, plane_fit= 0.015):
    #Beam detection from points inside a search box
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    obb = pcd.get_oriented_bounding_box()

    #RANSAC Planes
    planar_points = []
    plane_params = []
    planes_pcd, planes_eq = getNPlanesRANSAC(pcd, plane_fit, 3, 1000, 300) # 0.015
    if len (planar_points) == 0:
        planar_points = planes_pcd
        plane_params = planes_eq
    else:
        planar_points = [*planar_points, *planes_pcd]
        plane_params = [*plane_params, *planes_eq]

    #o3d.visualization.draw(planar_points)

    #Segment classification
    #o3d.visualization.draw([pcd,*planar_points])
    alpha_shapes = [ findAlphaShape(np.array(pts.points), mode="geometryProcess") for i, pts in enumerate(planar_points) if i < 4] # Consider only first 4 planar points

    obb_axes = getOBBAxis(obb)
    #Project obb axes to the planes of segments
    lines_ref = []
    for a_s in alpha_shapes:
        axis_2d = geometry.project3DPointsToPlane2D(obb_axes,a_s["plane"][:4])      
        ref_line = geometry.getLineEquation2D(axis_2d[0], axis_2d[1])
        lines_ref.append(ref_line)   
    #lines_border = [imagePrc.getHoughLinesFrom2DPts(np.array(a_s["pts2D"])) for a_s in alpha_shapes] -CGAL vs geometryProcess pts2D differs!!!!
    lines_border = [imagePrc.getHoughLinesFrom2DPts(geometry.project3DPointsToPlane2D(np.array(pts3d.points),alpha_shapes[i]["plane"])) for i, pts3d in enumerate(planar_points) if i < 4] 

    angles_to_ref = []
    for i, l in enumerate(lines_ref):
        a = []
        for l2 in lines_border[i]:
            a.append(np.abs(geometry.getAngleBetweenLines2D(l, l2)))
        angles_to_ref.append(a)

    cog2D_on_planes = [geometry.project3DPointToPlane2D(np.array(obb.get_center()),alpha_shapes[i]["plane"]) for i in range(len(lines_border))]

    line_matches = []
    for i,lines in enumerate(lines_border):
        lines = np.array(lines)
        lines_a = lines[np.where(np.array(angles_to_ref[i]) < 3.)]
        if len(lines_a) < 2:
            line_matches.append(None)
        elif len(lines_a) == 2:
            p_on_l0 = geometry.getPointOnLine2D(lines[0], cog2D_on_planes[i])
            dist = geometry.getPoint2LineDistance(lines[1], p_on_l0)
            if dist > 0.08 and dist < 0.4:
                line_matches.append(lines_a)
            else:
                line_matches.append(None)
        elif len(lines_a) > 2:
            a = np.array(angles_to_ref[i])[np.where(np.array(angles_to_ref[i]) < 3.)]

            kmeans = KMeans(n_clusters=2)
            kmeans = kmeans.fit(a.reshape(-1, 1) )
            
            match_id = None
            if len(a[kmeans.labels_==0]) == 1:
                match_id = 1
                lines_b = lines_a[kmeans.labels_==match_id]
            elif len(a[kmeans.labels_==1]) == 1:
                match_id = 0
                lines_b = lines_a[kmeans.labels_==match_id]
            else:
                match_id = np.argmin((np.std(a[kmeans.labels_==0]), np.std(a[kmeans.labels_==1])))
                lines_b = lines_a[kmeans.labels_==match_id]
            
            lines_c = lines_b [np.argsort(a[kmeans.labels_==match_id])]

            if len(lines_c) > 0 :
                p_on_l0 = geometry.getPointOnLine2D(lines_c[0], cog2D_on_planes[i])
                dist = geometry.getPoint2LineDistance(lines_c[1], p_on_l0)
                if dist > 0.08 and dist < 0.4:
                    line_matches.append(lines_c[:2])
                else:
                    line_matches.append(None)
            else:
                line_matches.append(None)

    # Get point between matching lines
    new_segment_pcd = []
    for i,line_match in enumerate(line_matches):
        if line_match is not None:
            id1 = np.argmin(alpha_shapes[i]["pts2D"][:,0])
            id2 = np.argmax(alpha_shapes[i]["pts2D"][:,0])
            id3 = np.argmin(alpha_shapes[i]["pts2D"][:,1])
            id4 = np.argmax(alpha_shapes[i]["pts2D"][:,1])
            v1 = geometry.projectPointToLine2D(line_match[0], alpha_shapes[i]["pts2D"][id1])
            v2 = geometry.projectPointToLine2D(line_match[0], alpha_shapes[i]["pts2D"][id2])
            v3 = geometry.projectPointToLine2D(line_match[0], alpha_shapes[i]["pts2D"][id3])
            v4 = geometry.projectPointToLine2D(line_match[0], alpha_shapes[i]["pts2D"][id4])
            v5 = geometry.projectPointToLine2D(line_match[1], alpha_shapes[i]["pts2D"][id1])
            v6 = geometry.projectPointToLine2D(line_match[1], alpha_shapes[i]["pts2D"][id2])
            v7 = geometry.projectPointToLine2D(line_match[1], alpha_shapes[i]["pts2D"][id3])
            v8 = geometry.projectPointToLine2D(line_match[1], alpha_shapes[i]["pts2D"][id4])
        
            multi_pts_2D = MultiPoint(alpha_shapes[i]["pts2D"])
            cut_poly = MultiPoint((v1,v2,v3,v4,v5,v6,v7,v8)).convex_hull
            ref_poly = multi_pts_2D.convex_hull
            inter_poly = ref_poly.intersection(cut_poly)
        
            inter_idx = [id for id, point in enumerate(multi_pts_2D) if inter_poly.intersects(point)]
        
            cut_pcd = planar_points[i].select_by_index(inter_idx)            
        
            new_segment_pcd.append(cut_pcd)
        else:
            new_segment_pcd.append(None)

   
    #if True:
        #o3d.visualization.draw([pcd,*new_segment_pcd])


    candidate_segments_pcd = [seg for seg in new_segment_pcd if seg is not None and len(seg.points) >100]
    
    candidate_alpha_shapes = [ findAlphaShape(np.array(pts.points), mode="geometryProcess") for i, pts in enumerate(candidate_segments_pcd)]
    
    angles = [ geometry.getAngleBetweenVectors(a_s["plane"][:3], candidate_alpha_shapes[0]["plane"][:3]) for a_s in candidate_alpha_shapes]
    
    #Side selection and adjacent beam surface extraction
    sides = [None, None, None, None] #[Parallel=0, Orthogonal=1]
    if len(angles) >1:
        sides[0] = 0
        for i,a in enumerate(angles):
            if i >0:
                if a >80. and a<100.:
                    #Orthogonal
                    sides[i] = 1
                if a < 10. or (a > 170. and a <= 180):
                    #Parallel
                    sides[i] = 0
    
        faces = [0, None, None, None] # [Base, Oppo., Right, Left]
        #v1 = None # First vec of Right or Left
        for i, a_s in enumerate(candidate_alpha_shapes):
            if i > 0:
                plane_base = candidate_alpha_shapes[0]["plane"]
                p_a = geometry.project3DPointToPlane2D(np.array(candidate_segments_pcd[0].get_center()), plane_base)
                p_b = geometry.project3DPointToPlane2D(np.array(candidate_segments_pcd[0].get_center()) + candidate_alpha_shapes[0]["eigenVecs"][2], plane_base) # Longitudinal direction on plane_base
                p_i = geometry.project3DPointToPlane2D(np.array(candidate_segments_pcd[i].get_center()), plane_base)
                d = (p_i[0] - p_a[0]) * (p_b[1] - p_a[1]) - (p_i[1] - p_a[1]) * (p_b[1] - p_a[1])
                
                if sides[i] == 0: #Parallel
                    p0 = np.array(candidate_segments_pcd[0].get_center())    
                    dist1 = np.abs(d)
                    dist2 = np.abs(geometry.getPoint2PlaneDistance(p0,a_s["plane"]))
                    if dist1 < 0.25 and dist2 > 0.08 and dist2 < 0.4:
                        if faces[1] is None:
                            faces[1] = i # Opposite to the base segment, ignore segments very closed to the base!!
                if sides[i] == 1: #Orthogonal
                    #if d > 0:
                    #    if faces[2] is None:
                    #        faces[2] = i
                    #else:
                    #    if faces[3] is None and d <0.:
                    #        faces[3] = i
                    if faces[2] is None:
                        faces[2] = i
                    else:
                        pt = np.array(candidate_segments_pcd[i].get_center())
                        pl = candidate_alpha_shapes[faces[2]]["plane"]
                        dist = np.abs(geometry.getPoint2PlaneDistance(pt,pl))
                        if dist > 0.08:
                            faces[3] = i
    
        #o3d.visualization.draw(candidate_segments_pcd)
        #Check if Right & Left are opposite to each other
        if faces[2] is not None and faces[3] is not None:
             base_id = np.argmax((len(candidate_segments_pcd[faces[2]].points) , len(candidate_segments_pcd[faces[3]].points))) + 2
             oppo_id = 2 if base_id == 3 else 3

             p0 = np.array(candidate_segments_pcd[faces[base_id]].get_center())
             plane_base = candidate_alpha_shapes[faces[base_id]]["plane"]
             p_a = geometry.project3DPointToPlane2D(np.array(candidate_segments_pcd[faces[base_id]].get_center()), plane_base)
             p_b = geometry.project3DPointToPlane2D(np.array(candidate_segments_pcd[faces[base_id]].get_center()) + candidate_alpha_shapes[faces[base_id]]["eigenVecs"][2], plane_base) # Longitudinal direction on plane_base
             p_i = geometry.project3DPointToPlane2D(np.array(candidate_segments_pcd[faces[oppo_id]].get_center()), plane_base)
             d = (p_i[0] - p_a[0]) * (p_b[1] - p_a[1]) - (p_i[1] - p_a[1]) * (p_b[1] - p_a[1])
             
             dist1 = np.abs(d)
             dist2 = np.abs(geometry.getPoint2PlaneDistance(p0,a_s["plane"]))
             if not (dist1 < 0.25 and dist2 > 0.08 and dist2 < 0.4):
                 faces[oppo_id] = None # Left & Right doesnt opposite to each other


        faces = np.array(faces)
        face_count = len([i for i in faces if i is not None])

        if face_count > 1:
            oriented_beam_faces = [np.empty(shape=(0,3), dtype=float), np.empty(shape=(0,3), dtype=float),
                                   np.empty(shape=(0,3), dtype=float), np.empty(shape=(0,3), dtype=float)]
            
            for i, face in enumerate(faces):
                if face is not None:
                    if i == 0: #Base
                        oriented_beam_faces[BEAM_SIDE.BASE.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.BASE.value], 
                                                                                        np.asarray(candidate_segments_pcd[face].points)))
                    elif i == 1:#Oppo
                        oriented_beam_faces[BEAM_SIDE.OPPOSITE.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.OPPOSITE.value], 
                                                                                        np.asarray(candidate_segments_pcd[face].points)))
                    elif i == 2:#Right
                        oriented_beam_faces[BEAM_SIDE.RIGHT.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.RIGHT.value], 
                                                                                        np.asarray(candidate_segments_pcd[face].points)))
                    elif i == 3:#Left
                        oriented_beam_faces[BEAM_SIDE.LEFT.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.LEFT.value], 
                                                                                        np.asarray(candidate_segments_pcd[face].points)))
            
            all_pts = o3d.geometry.PointCloud()
            for id in faces:
                if id is not None:
                    if all_pts is None:
                        all_pts.points = candidate_segments_pcd[id].points
                    else:
                        all_pts += candidate_segments_pcd[id]
            
            obb_rob = all_pts.get_oriented_bounding_box(robust = True)
            obb_norm = all_pts.get_oriented_bounding_box()
            #o3d.visualization.draw([all_pts, obb_rob, obb_norm, *candidate_segments_pcd])
            
            
            tmp_beam = external.pbs_beam.Beam(oriented_beam_faces)       
            cuboid_corners = tmp_beam.get_corner_points()
            
            if cuboid_corners is None:
                return None, None
            else:
                cuboid_corners_x = np.array(cuboid_corners)[:,0].flatten()
                cuboid_corners_y = np.array(cuboid_corners)[:,1].flatten()
                cuboid_corners_z = np.array(cuboid_corners)[:,2].flatten()
                cuboid_corners_xyz = np.transpose(np.vstack((cuboid_corners_x, cuboid_corners_y, cuboid_corners_z)))
                                         
                obb = o3d.geometry.OrientedBoundingBox()
                obb =  obb.create_from_points(points=o3d.utility.Vector3dVector(cuboid_corners_xyz))
                #o3d.visualization.draw([obb, *candidate_segments_pcd])
                
                return tmp_beam, obb
        else:
            return None, None
    else:
        return None, None
   
def getNPlanesRANSAC(pcd, ransac_th, ransac_n, ransac_iter, min_pts):
    
   inlier_clouds = []
   inlier_planes = []

   colors = [[1.0, 1.0, 0], # Yellow
            [0.0, 1.0, 1.0], # Turquoise
            [0.0, 1.0, 0.0], # Green
            [1.0, 0.0, 0.0] # Red
            ]


   import random
   for i in range(500):
       random.uniform(0, 1)
       colors.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])


   outlier_cloud = pcd
   for i in range(500):
       if len(outlier_cloud.points) > min_pts:
           plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=ransac_th,
                                                    ransac_n=ransac_n,
                                                    num_iterations=ransac_iter)
           
           [a, b, c, d] = plane_model
           if len(inliers) > min_pts:
               inlier_cloud = outlier_cloud.select_by_index(inliers)
               inlier_cloud.paint_uniform_color(colors[i])
               inlier_clouds.append(inlier_cloud)
               inlier_planes.append(plane_model)
               outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
           else:
               break

   return inlier_clouds, inlier_planes

def getColors(clusters):
    colors = [[1.0, 1.0, 0], # Yellow
        [0.0, 1.0, 1.0], # Turquoise
        [0.0, 1.0, 0.0], # Green
        [1.0, 0.0, 0.0] # Red
        ]

    if len(clusters) > 4:
        import random
        for i in range(len(clusters)):
            random.uniform(0, 1)
            colors.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    return colors

def beamDetection(pcd, box, box_pts, nr_planes=4, plot_sub_results=False):
      
    #Phase 1: DB Search on 2D projection
    pt_bottom = np.mean(np.vstack((box_pts[0],box_pts[2],box_pts[6],box_pts[4])), axis = 0)
    pt_top = np.mean(np.vstack((box_pts[1],box_pts[3],box_pts[7],box_pts[5])), axis = 0)

    nrm = np.array(geometry.getUnitVector(pt_top - pt_bottom))
    pt_center = np.mean(np.array(box_pts), axis = 0)
    d = - np.dot(nrm, pt_center)
    plane_ref = np.append(nrm,d)

    pts_2D = geometry.project3DPointsToPlane2D(np.array(pcd.points), plane_ref)


    dbClustering = DBSCAN(eps=0.01, min_samples=100, algorithm="kd_tree").fit(pts_2D)
    dbLabels = dbClustering.labels_
    dbNames = np.unique(dbLabels)
    colors = getColors(dbNames)

    tmp_names, counts = np.unique(dbLabels[np.where(dbLabels >-1)], return_counts=True)
    if len(counts):
        id_max_count = tmp_names[np.argmax(counts)]
        
        pcd_phase1 = None
        idx = np.where(dbLabels == id_max_count)[0]
        if len(idx) > 300:
            pcd_phase1 = pcd.select_by_index(idx)
            pcd_phase1.paint_uniform_color(colors[id_max_count])
              
        #for i, x in enumerate(dbNames):
        #    if x > -1:
        #        idx = np.where(dbLabels == x)[0]
        #        if len(idx) > 300:
        #            if pcd_phase1 == None:
        #                pcd_phase1 = pcd.select_by_index(idx)
        #                pcd_phase1.paint_uniform_color(colors[i])
        #            else:
        #                pcd_phase1 += pcd.select_by_index(idx)
        
        if plot_sub_results:
            o3d.visualization.draw([pcd, pcd_phase1])
        
        # Phase 2: 3D DBSCAN to get connected largest point groups
        if pcd_phase1 is not None:
            pcd_phase2 = []
            db_clustering_3d = DBSCAN(eps=0.1, min_samples=20, algorithm="kd_tree").fit(pcd_phase1.points)
            db_labels_3d = db_clustering_3d.labels_
            db_names_3d = np.unique(db_labels_3d)
            colors = getColors(db_names_3d)
            for n in db_names_3d:
                if n > -1:
                    idx = np.where(db_labels_3d == n)[0]
                    if len(idx) > 500:
                        seg_pcd = pcd_phase1.select_by_index(idx)
                        seg_pcd.paint_uniform_color(colors[n])
                        pcd_phase2.append(seg_pcd)
            
            if plot_sub_results:
                o3d.visualization.draw([pcd,*pcd_phase2])
            
            #Phase 3: Plane fitting RANSAC
            planar_points = []
            plane_params = []
            for pts in pcd_phase2:
                planes_pcd, planes_eq = getNPlanesRANSAC(pts, 0.015, 3, 1000, 300)
                if len (planar_points) == 0:
                    planar_points = planes_pcd
                    plane_params = planes_eq
                else:
                    planar_points = [*planar_points, *planes_pcd]
                    plane_params = [*plane_params, *planes_eq]
                   
            if plot_sub_results:
                o3d.visualization.draw([pcd,*planar_points])
        
            #Phase 4: Connected components on planar points(2D)
            pcd_phase4 = []
            pcd_phase4_2D = []
            planes_phase4 = []
            if len(planar_points) > 0:
                for i, plane_pts in enumerate(planar_points):           
                    """
                    #Reference plane re-estimation
                    pt_bottom_3d = geometry.reproject2DPointToPlane3D(pt_bottom, plane_ref)
                    pt_top_3d = geometry.reproject2DPointToPlane3D(pt_top, plane_ref)
                    
                    pt_bottom_2d = geometry.project3DPointToPlane2D(pt_bottom_3d, plane_params[i])
                    pt_top_2d = geometry.project3DPointToPlane2D(pt_top_3d, plane_params[i])
                    
                    pt_bottom_3d = geometry.reproject2DPointToPlane3D(pt_bottom, plane_params[i])
                    pt_top_3d = geometry.reproject2DPointToPlane3D(pt_top, plane_params[i])
                
                    nrm = np.array(geometry.getUnitVector(pt_top_3d - pt_bottom_3d))
                    #pt_center =  np.mean(np.array(box_pts), axis = 0)
                    
                    d =  - np.dot(nrm, pt_center)                
                    plane_ref_update = np.append(nrm,d)
                    """
                            
                    pts_2D = geometry.project3DPointsToPlane2D(np.array(plane_pts.points), plane_params[i])     
                    
                    dbClustering = DBSCAN(eps=0.1, min_samples=20, algorithm="kd_tree").fit(pts_2D)
                    dbLabels = dbClustering.labels_
                    dbNames = np.unique(dbLabels)                
                    colors = getColors(dbNames)
                                    
                    for j, x in enumerate(dbNames):
                        if x > -1:
                            idx = np.where(dbLabels == x)[0]
                            if len(idx) > 300:
                                res_pcd = plane_pts.select_by_index(idx)
                                res_pcd.paint_uniform_color(colors[j])
        
                                pcd_phase4_2D.append(pts_2D[idx])
                                pcd_phase4.append(res_pcd)
                                planes_phase4.append(plane_params[i])
                           
                if plot_sub_results:
                    o3d.visualization.draw([pcd,*pcd_phase4])
        

        else:
            print ("No beam face found in the search box: Level1")
            return None
        #Phase 5 Beam side detection
        if len(pcd_phase4_2D):
            pcd_phase5 = []
            planes_phase5 = []
            for i, pts_2D in enumerate(pcd_phase4_2D):
                ref_plane = planes_phase4[i]
                pt_bottom_2d = geometry.project3DPointToPlane2D(pt_bottom, ref_plane)
                pt_top_2d = geometry.project3DPointToPlane2D(pt_top, ref_plane)
            
                ref_line = geometry.getLineEquation2D(pt_bottom_2d, pt_top_2d)
            
                lines = imagePrc.getHoughLinesFrom2DPts(pts_2D)
                cog_2D = np.mean(pts_2D, axis = 0)
            
                useful_lines = []
                for l in lines:
                    a = abs(geometry.getAngleBetweenLines2D(ref_line, l))
                    if a < 5.:
                        useful_lines.append(l)
            
                if len(useful_lines) > 1:
            
                     if len(useful_lines) > 2:
                        useful_lines = np.array(useful_lines)
                        useful_lines = useful_lines[useful_lines[:, 0].argsort()]
                        intervals = np.array([ m - useful_lines[i - 1][0]  for i,m in enumerate(useful_lines[:, 0]) if i>0])
                        id = np.argmin(intervals) # TODO: if there is more than one,  the best one need to be used not the first
                        l1 = useful_lines[id]
                        l2 = useful_lines[id + 1]
                        useful_lines = [l1, l2]
                    
                     if len(useful_lines) == 2:
                        p1 = geometry.projectPointToLine2D(useful_lines[0], cog_2D)
                        p2 = geometry.projectPointToLine2D(useful_lines[1], cog_2D)               
                        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        if dist > 0.4:
                            useful_lines = []
            
                # Get point between lines
                if len(useful_lines) ==2:
                    id1 = np.argmin(pts_2D[:,0])
                    id2 = np.argmax(pts_2D[:,0])
                    id3 = np.argmin(pts_2D[:,1])
                    id4 = np.argmax(pts_2D[:,1])
                    v1 = geometry.projectPointToLine2D(useful_lines[0], pts_2D[id1])
                    v2 = geometry.projectPointToLine2D(useful_lines[0], pts_2D[id2])
                    v3 = geometry.projectPointToLine2D(useful_lines[0], pts_2D[id3])
                    v4 = geometry.projectPointToLine2D(useful_lines[0], pts_2D[id4])
                    v5 = geometry.projectPointToLine2D(useful_lines[1], pts_2D[id1])
                    v6 = geometry.projectPointToLine2D(useful_lines[1], pts_2D[id2])
                    v7 = geometry.projectPointToLine2D(useful_lines[1], pts_2D[id3])
                    v8 = geometry.projectPointToLine2D(useful_lines[1], pts_2D[id4])
            
                    multi_pts_2D = MultiPoint(pts_2D)
                    cut_poly = MultiPoint((v1,v2,v3,v4,v5,v6,v7,v8)).convex_hull
                    ref_poly = multi_pts_2D.convex_hull
                    inter_poly = ref_poly.intersection(cut_poly)
            
                    inter_idx = [id for id, point in enumerate(multi_pts_2D) if inter_poly.intersects(point)]
            
                    cut_pcd = pcd_phase4[i].select_by_index(inter_idx)            
            
                    pcd_phase5.append(cut_pcd)
                    planes_phase5.append(planes_phase4[i])
            
            if plot_sub_results:
                o3d.visualization.draw([pcd,*pcd_phase5])
                #o3d.visualization.draw([pcd_phase5[0]])
            return {"face_points": pcd_phase5, "face_planes": planes_phase5}
        else:
            print ("No beam face found in the search box: Level2")
            return None
    else:
        print ("No beam face found in the search box: Level3")
        return None
    
def getBeamModel(grup_ref_plane, ref_lines_2D, beam_faces, plot_sub_results=False):
    #Phase 6: Beam side identification
    all_pts = o3d.geometry.PointCloud()
    for pcd in beam_faces["face_points"]:
        if all_pts is None:
            all_pts.points = pcd.points
        else:
            all_pts += pcd
    cog = all_pts.get_center()
   
    #o3d.visualization.draw(beam_faces["face_points"])

    oriented_planes = []
    for i, plane in enumerate(beam_faces["face_planes"]):
        p =  beam_faces["face_points"][i].get_center()
        n = plane[:3]
        n_or= geometry.orientNormalVector(p,n,cog,False)
        d = - np.dot(p, n_or)
        oriented_planes.append(np.append(n_or,d))

    oriented_beam_faces = [np.empty(shape=(0,3), dtype=float), np.empty(shape=(0,3), dtype=float),
                               np.empty(shape=(0,3), dtype=float), np.empty(shape=(0,3), dtype=float)]
    beam_face_found = [False, False, False, False]

    #ref_lines top, center, bottom
    cog_2D = geometry.project3DPointToPlane2D(cog,grup_ref_plane)
    p_bottom = geometry.projectPointToLine2D(ref_lines_2D[2], cog_2D)
    p_top = geometry.projectPointToLine2D(ref_lines_2D[0], cog_2D)

    
    #Excluding of irrelevant beam face : _|‾ // START
    #First plane is reference
    valid_beam_faces = np.full(len(oriented_planes), True, dtype =bool)
    if len(oriented_planes)>2:
        ref_plane1 = oriented_planes[0]
        ref_plane2 = None
        
        #Define the slice plane
        ref_pts1 = beam_faces["face_points"][0].points
        ref_pts1_prj2D = geometry.project3DPointsToPlane2D(ref_pts1, ref_plane1)
        
        p_bottom3D = geometry.reproject2DPointToPlane3D(p_bottom, grup_ref_plane)
        p_top3D = geometry.reproject2DPointToPlane3D(p_top, grup_ref_plane)
        
        x_mbr1, y_mbr1 = MultiPoint(ref_pts1_prj2D).minimum_rotated_rectangle.exterior.coords.xy
        box_pts1_2D = np.transpose(np.vstack((x_mbr1,y_mbr1)))[:4]
        #box_pts1_3D = [geometry.reproject2DPointToPlane3D(p,ref_plane1) for p in box_pts1_2D]
        
        #Get side lines of the box through longitudinal axis
        edge_length = (Point(box_pts1_2D[0]).distance(Point(box_pts1_2D[1])), Point(box_pts1_2D[1]).distance(Point(box_pts1_2D[2])))
        axis_id = np.argmax(edge_length)
        if axis_id == 0:  
            side_line11 = geometry.getLineEquation2D(box_pts1_2D[0], box_pts1_2D[1])
            side_line12 = geometry.getLineEquation2D(box_pts1_2D[3], box_pts1_2D[2])
        elif axis_id == 1:
            side_line11 = geometry.getLineEquation2D(box_pts1_2D[1], box_pts1_2D[2])
            side_line12 = geometry.getLineEquation2D(box_pts1_2D[0], box_pts1_2D[3])
        
        for i, pl in enumerate(oriented_planes):
            if i> 0:
                angle = geometry.getAngleBetweenVectors(ref_plane1[:3], pl[:3])
                if angle < 20. or angle > 160:
        
                    target_pts = beam_faces["face_points"][i].points
                    #target_pts_2D = geometry.project3DPointsToPlane2D(target_pts, pl_slice)
                    target_center = np.mean(target_pts, axis = 0)
                    target_center_2D = geometry.project3DPointToPlane2D(target_center, ref_plane1)
        
                    po_1 = geometry.projectPointToLine2D(side_line11,target_center_2D)
                    po_2 = geometry.projectPointToLine2D(side_line12,target_center_2D)
        
                    check_rule = geometry.isPointOnLineSegment(target_center_2D, (po_1,po_2))
                    if check_rule==False:
                        valid_beam_faces[i] = False
                else:
                    if ref_plane2 is None:
                        ref_plane2 = pl
                        ref_pts2 = beam_faces["face_points"][i].points
                        ref_pts2_2D = geometry.project3DPointsToPlane2D(ref_pts2, ref_plane2)
                        x_mbr2, y_mbr2 = MultiPoint(ref_pts2_2D).minimum_rotated_rectangle.exterior.coords.xy
                        box_pts2_2D = np.transpose(np.vstack((x_mbr2,y_mbr2)))[:4]
        
                        #Get side lines of the box through longitudinal axis
                        edge_length2 = (Point(box_pts2_2D[0]).distance(Point(box_pts2_2D[1])), Point(box_pts2_2D[1]).distance(Point(box_pts2_2D[2])))
                        axis_id2 = np.argmax(edge_length2)
                        if axis_id == 0:  
                            side_line21 = geometry.getLineEquation2D(box_pts2_2D[0], box_pts2_2D[1])
                            side_line22 = geometry.getLineEquation2D(box_pts2_2D[3], box_pts2_2D[2])
                        elif axis_id == 1:
                            side_line21 = geometry.getLineEquation2D(box_pts2_2D[1], box_pts2_2D[2])
                            side_line22 = geometry.getLineEquation2D(box_pts2_2D[0], box_pts2_2D[3])
                        
        
                    else:
                        angle = geometry.getAngleBetweenVectors(ref_plane2[:3], pl[:3])
                        if angle < 20. or angle > 160:
                        
                            target_pts = beam_faces["face_points"][i].points
                            #target_pts_2D = geometry.project3DPointsToPlane2D(target_pts, pl_slice)
                            target_center = np.mean(target_pts, axis = 0)
                            target_center_2D = geometry.project3DPointToPlane2D(target_center, ref_plane2)
                            
                            po_1 = geometry.projectPointToLine2D(side_line21,target_center_2D)
                            po_2 = geometry.projectPointToLine2D(side_line22,target_center_2D)
                            
                            check_rule = geometry.isPointOnLineSegment(target_center_2D, (po_1,po_2))
                            if check_rule==False:
                                valid_beam_faces[i] = False
    #Excluding of irrelevant beam face : _|‾ // END

    #a = oriented_beam_faces[BEAM_SIDE.BASE.value]
    #b = np.asarray(beam_faces["face_points"][i].points)
    #c= np.concatenate((oriented_beam_faces[BEAM_SIDE.BASE.value],b))

    for i, plane in enumerate(oriented_planes):
        if valid_beam_faces[i]:
            angle = geometry.getAngleBetweenVectors(plane[:3], grup_ref_plane[:3])
            if angle < 20. or angle > 160:
                #Parallel
                if np.dot(plane[:3], grup_ref_plane[:3]) > 0: #TODO: Check if grup_ref_plane orientation is fixed!!!
                    #Opposite
                    oriented_beam_faces[BEAM_SIDE.OPPOSITE.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.OPPOSITE.value], 
                                                                                    np.asarray(beam_faces["face_points"][i].points)))
                    beam_face_found[BEAM_SIDE.OPPOSITE.value] = True
                else:
                    #Base
                    oriented_beam_faces[BEAM_SIDE.BASE.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.BASE.value], 
                                                                                np.asarray(beam_faces["face_points"][i].points)))
                    beam_face_found[BEAM_SIDE.BASE.value] = True
            else:
                #Perpendicular
                p_3D = beam_faces["face_points"][i].get_center()
                p_2D = geometry.project3DPointToPlane2D(p_3D, grup_ref_plane)
            
                v1 = np.array(p_top) - np.array(p_bottom)
                v2 = np.array(p_2D) - np.array(p_bottom)
                
                if np.cross(v1, v2) > 0: # Check with clarification of grup_ref_plane direction!!!
                    #Right
                    oriented_beam_faces[BEAM_SIDE.RIGHT.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.RIGHT.value], 
                                                                                 np.asarray(beam_faces["face_points"][i].points)))
                    beam_face_found[BEAM_SIDE.RIGHT.value] = True
                else:
                    #Left
                    oriented_beam_faces[BEAM_SIDE.LEFT.value] = np.concatenate((oriented_beam_faces[BEAM_SIDE.LEFT.value], 
                                                                                np.asarray(beam_faces["face_points"][i].points)))
                    beam_face_found[BEAM_SIDE.LEFT.value] = True
        
    #If accidentally base side is not found!
    if len(oriented_beam_faces[BEAM_SIDE.BASE.value]) == 0 and len(oriented_beam_faces[BEAM_SIDE.OPPOSITE.value]) > 0:
        oriented_beam_faces = [oriented_beam_faces[BEAM_SIDE.OPPOSITE.value],
                               oriented_beam_faces[BEAM_SIDE.LEFT.value],
                               oriented_beam_faces[BEAM_SIDE.RIGHT.value],
                               oriented_beam_faces[BEAM_SIDE.BASE.value]]
        beam_face_found =     [beam_face_found[BEAM_SIDE.OPPOSITE.value],
                               beam_face_found[BEAM_SIDE.LEFT.value],
                               beam_face_found[BEAM_SIDE.RIGHT.value],
                               beam_face_found[BEAM_SIDE.BASE.value]]
              
    nr_sides_found = np.count_nonzero(beam_face_found)

    #o3d.visualization.draw([beam_faces["face_points"][0]])
    if oriented_beam_faces is not None:
        if nr_sides_found == 1:
            oriented_beam_faces = getBeamSingleFace(oriented_beam_faces, beam_face_found)
            if oriented_beam_faces is not None:
                #Here we have at least 2 faces
                cuboid_pcd = o3d.geometry.PointCloud()
                cuboid_pts = np.vstack(oriented_beam_faces)
                cuboid_pcd.points = o3d.utility.Vector3dVector(cuboid_pts) 
                obb = cuboid_pcd.get_oriented_bounding_box()
                
                if plot_sub_results:
                    o3d.visualization.draw([cuboid_pcd, obb])
                #exportOBB2Mesh(obb, "data/exp/cuboids.ply")
                #exportOBB2Dxf(obb, "data/exp/new_beams.dxf", color = 5)
                return obb

            else:
                print("No beam face in the search box: Level5")
                return None       
        else:        
            try:        
                tmp_beam = external.pbs_beam.Beam(oriented_beam_faces)       
                cuboid_corners = tmp_beam.get_corner_points()
                            
                cuboid_corners_x = np.array(cuboid_corners)[:,0].flatten()
                cuboid_corners_y = np.array(cuboid_corners)[:,1].flatten()
                cuboid_corners_z = np.array(cuboid_corners)[:,2].flatten()
                cuboid_corners_xyz = np.transpose(np.vstack((cuboid_corners_x, cuboid_corners_y, cuboid_corners_z)))
                                         
                obb = o3d.geometry.OrientedBoundingBox()
                obb =  obb.create_from_points(points=o3d.utility.Vector3dVector(cuboid_corners_xyz))

                ##OBB->Beam Conversion test
                #beam_ = external.pbs_beam.Beam(None)
                ##beam_.R = np.array([-obb.R[:,2], obb.R[:,1], obb.R[:,0]])
                ##beam_.dimensions = np.array([obb.extend[2], obb.extend[1], obb.extend[0]])
                ##beam_.basepoint = None
                #beam_.R = np.array([obb.R[:,2], obb.R[:,1], -obb.R[:,0]]).transpose()
                #beam_.dimensions = np.array([obb.extent[2], obb.extent[1], obb.extent[0]])
                #
                #
                ##Define the basepoint
                #beam_box_pts = np.array(obb.get_box_points())
                #diff_list = []
                #for p in beam_box_pts:
                #    candidate = np.dot(beam_.R, beam_.dimensions) + p
                #    if candidate in beam_box_pts:
                #        beam_.basepoint = np.array(([p[0]],[p[1]],[p[2]]))
                #    diff_list.append(candidate)
                #
                #from dxfwrite import DXFEngine as dxf
                #beams_drawing = dxf.drawing(name="data/test/tmpBeam.dxf")
                #beams_drawing.add_layer(name="Beams", color=1)
                #beams_drawing.add_layer(name="Beam-Axes", color=7)
                #beams_drawing.add_layer(name="Joints", color=2)
                #
                #tmp_dxf_cuboid = tmp_beam.get_dxfwrite_cuboid(color=None,layer="Beams")
                #if tmp_dxf_cuboid is not None:
                #    beams_drawing.add(tmp_dxf_cuboid)
                # 
                #beams_drawing_ = dxf.drawing(name="data/test/obbBeam.dxf")
                #beams_drawing_.add_layer(name="Beams", color=1)
                #beams_drawing_.add_layer(name="Beam-Axes", color=7)
                #beams_drawing_.add_layer(name="Joints", color=2)
                #
                #tmp_dxf_cuboid_ = beam_.get_dxfwrite_cuboid(color=None,layer="Beams")
                #if tmp_dxf_cuboid_ is not None:
                #    beams_drawing_.add(tmp_dxf_cuboid_)
                #beams_drawing.save()
                #beams_drawing_.save()
                #
                #beam_.basepoint = None

                
                #all_pts = np.vstack(oriented_beam_faces)
                #all_pcd = o3d.geometry.PointCloud()
                #all_pcd.points = o3d.utility.Vector3dVector(all_pts)                    
                #o3d.visualization.draw([obb,cuboid_pcd, all_pcd])
        
                #exportCuboid2Dxf(tmp_beam, "data/exp/new_beams.dxf", color = 5)
                #exportCuboid2Mesh(tmp_beam, "data/cuboids.ply")
                return obb
        
            except:
              print("Cuboid fitting failed, single-face solution is being performed instead.")
              
              nr_pts = [ np.size(face) for face in oriented_beam_faces]
              
              beam_face_found = [False, False, False, False]
              beam_face_found[np.argmax(nr_pts)] = True
              oriented_beam_faces = getBeamSingleFace(oriented_beam_faces, beam_face_found)
              if oriented_beam_faces is not None:
                  cuboid_pcd = o3d.geometry.PointCloud()
                  cuboid_pts = np.vstack(oriented_beam_faces)
                  cuboid_pcd.points = o3d.utility.Vector3dVector(cuboid_pts) 
                  obb = cuboid_pcd.get_oriented_bounding_box()
                  
                  if plot_sub_results:
                      o3d.visualization.draw([cuboid_pcd, obb])

                  #OBB->Beam Conversion test
                  beam_ = external.pbs_beam.Beam(None)
                  beam_.R = np.array([-obb.R[:,2], obb.R[:,1], obb.R[:,0]]).transpose()
                  beam_.dimensions = np.array([obb.extent[2], obb.extent[1], obb.extent[0]])
                  beam_.basepoint = None
                  
                  return obb
                  #exportOBB2Mesh(obb, "data/exp/cuboids.ply")
                  #exportOBB2Dxf(obb, "data/exp/new_beams.dxf", color = 5)
              else:
                  print("No beam face in the search box: Level6")
                  return None
    else:
        print("No beam face in the search box: Level7")
        return None

def getBeamSingleFace(oriented_beam_faces, beam_face_found, width = None):
    side = np.where(np.array(beam_face_found) == True)[0][0]    
    #side = 0 # Test case for base + opposite
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(oriented_beam_faces[side])
    #For a better in/out estimation, plane fitting re-applied with smaller distance threshold
    planes_pcd, planes_eq = getNPlanesRANSAC(pcd, 0.01, 40, 1000, 300)

    if len(planes_pcd):

        #o3d.visualization.draw([pcd])
        
        cog = pcd.get_center()
        #Use first largest planar points as plane reference
        ref_point = planes_pcd[0].get_center()
        n = geometry.orientNormalVector(ref_point, planes_eq[0][:3], cog, False)
        d = - np.dot(n, ref_point)
        plane = np.append(n, d)
        
        #Prepare reference shape vertices
        pts_2D = geometry.project3DPointsToPlane2D(oriented_beam_faces[side],plane)
        mbr_2D = MultiPoint(pts_2D).minimum_rotated_rectangle
        centroid_2D = (mbr_2D.centroid.x, mbr_2D.centroid.y)
        
        mbr_x, mbr_y = mbr_2D.exterior.coords.xy
        
        if width == None:
            edge_length = (Point(mbr_x[0], mbr_y[0]).distance(Point(mbr_x[1], mbr_y[1])), Point(mbr_x[1], mbr_y[1]).distance(Point(mbr_x[2], mbr_y[2])))
            width = min(edge_length)
        mbr_vertices_2D = np.transpose(np.vstack((mbr_x, mbr_y)))[:4]
        
        #width calculation should come here!!!
        plane1, plane2 = geometry.getParallelPlanes(plane, width)
        
        po1 = geometry.project3DPointToPlane(ref_point, plane1)
        #po2 = geometry.project3DPointToPlane(ref_point, plane2)
        
        v1 = po1 - ref_point
        #v2 = po2 - ref_point
        
        #Negative signed distance gives the correct plane
        signed_dist = np.dot(v1, n)
        #sig_dist2 = np.dot(v2, n)
        
        if signed_dist < 0:
            #Use plane 1       
            #oppo_3D = [ geometry.reproject2DPointToPlane3D(p, plane1) for  p in pts_2D]
            #pcd_artificial = o3d.geometry.PointCloud()
            oppo_3D = [ geometry.reproject2DPointToPlane3D(p, plane1) for  p in mbr_vertices_2D]
            oppo_3D.append(geometry.reproject2DPointToPlane3D(centroid_2D, plane1))
            [oppo_3D.append(geometry.reproject2DPointToPlane3D(np.mean((np.array(p), np.array(centroid_2D)), axis = 0), plane1)) for  p in mbr_vertices_2D]
            #pcd_artificial.points = o3d.utility.Vector3dVector(oppo_3D)
            
        else:
            #Use plane 2
            #oppo_3D = [ geometry.reproject2DPointToPlane3D(p, plane2) for  p in pts_2D]
            #pcd_artificial = o3d.geometry.PointCloud()
            oppo_3D = [ geometry.reproject2DPointToPlane3D(p, plane2) for  p in mbr_vertices_2D]
            oppo_3D.append(geometry.reproject2DPointToPlane3D(centroid_2D, plane2))
            [oppo_3D.append(geometry.reproject2DPointToPlane3D(np.mean((np.array(p), np.array(centroid_2D)), axis = 0), plane2)) for  p in mbr_vertices_2D]
            #pcd_artificial.points = o3d.utility.Vector3dVector(oppo_3D)
        
        #base_3D = [ geometry.reproject2DPointToPlane3D(p, plane) for  p in mbr_vertices_2D]
        #base_3D.append(geometry.reproject2DPointToPlane3D(centroid_2D, plane))
        #[base_3D.append(geometry.reproject2DPointToPlane3D(np.mean((np.array(p), np.array(centroid_2D)), axis = 0), plane)) for  p in mbr_vertices_2D]
        
        opposite_side = None
        for s in BEAM_SIDE:
            if s.value == side:
                if side == BEAM_SIDE.BASE.value:
                    opposite_side = BEAM_SIDE.OPPOSITE
                    break
                elif side == BEAM_SIDE.RIGHT.value:
                    opposite_side = BEAM_SIDE.LEFT
                    break
                elif side == BEAM_SIDE.LEFT.value:
                    opposite_side = BEAM_SIDE.RIGHT
                    break
                elif side == BEAM_SIDE.OPPOSITE.value:
                    opposite_side = BEAM_SIDE.BASE
                    break
        
        #oriented_beam_faces[side] = np.asarray(base_3D)
        oriented_beam_faces[opposite_side.value] = np.asarray(oppo_3D)
        
        #pcd_artificial2 = o3d.geometry.PointCloud()
        #pcd_artificial2.points = o3d.utility.Vector3dVector(base_3D)
        #o3d.visualization.draw([pcd,pcd_artificial, pcd_artificial2])
        return oriented_beam_faces
    else:
        print("No beam face fouund in the search box: Level4")
        return None

def getPointsInBox(pcd, obb):
    #print(2)
    idx = obb.get_point_indices_within_bounding_box(pcd.points)   
    pts_in_box = pcd.select_by_index(idx, invert=False)
    return pts_in_box

def getSearchBox3D(beam_2D, refPlane, offset, coordinates_xy=True):
    #Get parallel lines to define buffer
    line_eq = geometry.getLineEquation2D(beam_2D[0], beam_2D[1])
    l1, l2 = geometry.getParallelLines(line_eq, offset)

    p1_1 = geometry.projectPointToLine2D(l1,beam_2D[0])
    p1_2 = geometry.projectPointToLine2D(l1,beam_2D[1])
    p2_1 = geometry.projectPointToLine2D(l2,beam_2D[0])
    p2_2 = geometry.projectPointToLine2D(l2,beam_2D[1])

    if coordinates_xy is False:
        p1_1 = (p1_1[1], p1_1[0])
        p1_2 = (p1_2[1], p1_2[0])
        p2_1 = (p2_1[1], p2_1[0])
        p2_2 = (p2_2[1], p2_2[0])

    #Get parallel planes of reference plane (Ref plane is assumed as central plane)
    pl1, pl2 = geometry.getParallelPlanes(refPlane, offset)

    #Reproject defined 2D point to 3D planes
    p1 = geometry.reproject2DPointToPlane3D(p1_1, pl1)
    p2 = geometry.reproject2DPointToPlane3D(p1_2, pl1)
    p3 = geometry.reproject2DPointToPlane3D(p2_1, pl1)
    p4 = geometry.reproject2DPointToPlane3D(p2_2, pl1)

    p5 = geometry.reproject2DPointToPlane3D(p1_1, pl2)
    p6 = geometry.reproject2DPointToPlane3D(p1_2, pl2)
    p7 = geometry.reproject2DPointToPlane3D(p2_1, pl2)
    p8 = geometry.reproject2DPointToPlane3D(p2_2, pl2)

    pts = [p1,p2,p3,p4,p5,p6,p7,p8]

    obb = o3d.geometry.OrientedBoundingBox()
    obb =  obb.create_from_points(points=o3d.utility.Vector3dVector(pts))
    return obb, pts

def missingBeams(new_beams_2D, beam_gr, point_cloud, ref_planes):
    new_beams = []
    not_found_boxes = []
    search_boxes = []
    for i, b in enumerate(new_beams_2D):
        if True:#i == 1:
            box, box_pts = getSearchBox3D(b, ref_planes[i], 0.13, coordinates_xy=beam_gr.coordinates_xy)#offset:0.25
            search_boxes.append(box)
            points_in_box = getPointsInBox(point_cloud, box)
            points_in_box.paint_uniform_color([1.0, 0.0, 0])        
            beam_faces= beamDetection(points_in_box, box, box_pts,4, False)
            if beam_faces:
                model = getBeamModel(ref_planes[i], beam_gr.optimal_reference_lines, beam_faces, False)
                if model:
                    new_beams.append(model)
                else:
                    not_found_boxes.append(box)
            else:
                not_found_boxes.append(box)
    return new_beams, not_found_boxes, search_boxes

def exportOBBList2Dxf(obb_list, file_name, color=0):
    beams = []
    for obb in obb_list:
        obb_vertices = np.asarray(obb.get_box_points())
        obb_faces = []
        obb_faces.append([obb_vertices[0], obb_vertices[1], obb_vertices[7], obb_vertices[2]]) #Front
        obb_faces.append([obb_vertices[0], obb_vertices[1], obb_vertices[6], obb_vertices[3]]) #Right
        obb_faces.append([obb_vertices[3], obb_vertices[5], obb_vertices[4], obb_vertices[6]]) #Back
        obb_faces.append([obb_vertices[4], obb_vertices[5], obb_vertices[2], obb_vertices[7]]) #Left
        obb_faces.append([obb_vertices[4], obb_vertices[6], obb_vertices[1], obb_vertices[7]]) #Top
        obb_faces.append([obb_vertices[0], obb_vertices[2], obb_vertices[5], obb_vertices[3]]) #Bottom
        beams.append(obb_faces)
        
    with r12writer(file_name) as dxf:
        for beam_faces in beams:
            for i, face in enumerate(beam_faces):
                dxf.add_polyface(vertices= face, faces=[(0, 1, 2, 3)], color=color, layer = "Beams")
            dxf.add_line(np.mean(beam_faces[5], axis=0),np.mean(beam_faces[4], axis=0), color=color, layer = "Beam-Axes")

def saveLines2DXF(lines, outputPath, color, id):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for l in lines:
        msp.add_line(l.dxf.start, l.dxf.end)
    
    linesW = msp.query('LINE')
    
    for k,lW in enumerate(linesW):
        lW.dxf.color = color
    doc.saveas(outputPath + "/lines_" + str(id) + ".dxf")

def iterativeBeamModeling(pcd, voxel_size=0.05, plane_sigma=0.005, nr_iter=5):
    if len(pcd.points) > 100:
        if voxel_size > 0.02:
            downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
            beam_candidate = None, None
            beam_list = []
            # iterative cuboid fitting
            for i in range(nr_iter):
                cuboid, bbx = beamDetectionSingle(np.array(downpcd.points), plane_sigma)
                
                if cuboid is not None and cuboid.sigma0 < plane_sigma / 3.7:
                    beam_candidate = cuboid, bbx
                    return beam_candidate
                elif cuboid is not None:
                    beam_list.append((cuboid,bbx))
        
            if len(beam_list) >0:
                sigma_list = np.array([c[0] .sigma0 for c in beam_list])
                beam_candidate = beam_list[np.argmin(sigma_list)]
                return beam_candidate
                    
            if beam_candidate[0] is None:
                #Change the voxel size and try again
                downpcd = pcd.voxel_down_sample(voxel_size=0.02)
                beam_list = []
                for i in range(nr_iter):
                    cuboid, bbx = beamDetectionSingle(np.array(downpcd.points), plane_sigma)
        
                    if cuboid is not None and cuboid.sigma0 < plane_sigma / 3.7:
                        beam_candidate = cuboid, bbx
                        return beam_candidate
                    elif cuboid is not None:
                        beam_list.append((cuboid,bbx))
                if len(beam_list) >0:
                    sigma_list = np.array([c[0] .sigma0 for c in beam_list])
                    beam_candidate = beam_list[np.argmin(sigma_list)]
                    return beam_candidate
            return beam_candidate
        else:
            downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            beam_candidate = None, None
            beam_list = []
            for i in range(nr_iter):
                cuboid, bbx = beamDetectionSingle(np.array(downpcd.points), plane_sigma)
        
                if cuboid is not None and cuboid.sigma0 < plane_sigma / 3.7:
                     beam_candidate = cuboid, bbx
                     return beam_candidate
                elif cuboid is not None:
                        beam_list.append((cuboid,bbx))
            if len(beam_list) >0:
                sigma_list = np.array([c[0] .sigma0 for c in beam_list])
                beam_candidate = beam_list[np.argmin(sigma_list)]
                return beam_candidate
            return beam_candidate
    else:
        return None, None

def processBeamGroup(beams, roof_tile, pcd, template_dims = [12., 0.2, 0.201]):

    #Get beams of corresponding roof_tile
    beams_db_gr = [b for b in beams if b.roof_tile_id == roof_tile.id]

    for b in beams_db_gr:
        b.old_id = b.id
        b.group_id = roof_tile.id

    beams_dxf_gr = exchange.getBeamAxesAsDXF(beams_db_gr)
    
    #beam_gr_obb = [b.setOBB() for b in beams_db_gr]
    #exportOBBList2Dxf(beam_gr_obb, "data/wing_north/beamgr1.dxf", 1) 

    beamGr = BeamGroup.BeamGroup(beam_axes = beams_dxf_gr, roof_plane= roof_tile.plane)
    beamGr.id = roof_tile.id
    beamGr.name = "roof_tile"
    
    #set avg_dimensions based on reliable beams
    rel_beams = np.array([(b.width, b.height,b.length) for i,b in enumerate(beams) if i in beamGr.reliable_beams])
    avg_dimesions = np.mean(rel_beams, axis = 0)
    beamGr.avg_width = avg_dimesions[0]
    beamGr.avg_height = avg_dimesions[1]
    beamGr.avg_length = avg_dimesions[2]

    new_beams_2D, ref_planes = beamGr.getAdditionalBeams(align_to_neighbours=True)
    merge_beams_2D = beamGr.getMergableBeams(full_extend=True)
    extend_beams_2D = beamGr.getExtendableBeams(threshold=0.5)

    #Deciding on reference template: check w/h of reliable beams
    #beam_dims = np.array([(beams_db_gr[i].width, beams_db_gr[i].height) for i in beamGr.reliable_beams])
    temp_obb = beams_db_gr[beamGr.reliable_beams[0]].setOBB() # Get the first reliable beam as temp (just for a reference orientation!)
    temp_obb = None ##### For testing of tower data
    beamGr.template_obb = temp_obb

    #Ignore beams
    ignore_beams = []
    if len(beamGr.ignore_beams) > 0:
        ignore_beams = [beams_db_gr[id] for id in beamGr.ignore_beams]
        for i,b in enumerate(ignore_beams):
            ignore_beams[i].comment = "stage2_ignore"

    #Create (if) beams found
    created_beams = []
    created_beams_obb = []
    for i, b in enumerate(new_beams_2D):
        box, box_pts = getSearchBox3D(b, ref_planes[i], 0.13, coordinates_xy=beamGr.coordinates_xy)

        #o3d.visualization.draw([pcd, box])

        points_in_box = getPointsInBox(pcd, box)
        if temp_obb:
            obb = template.getBeamInSearchBox(pcd, box, temp_obb, template_len=template_dims[0], target_dims=(template_dims[1], template_dims[2]))
        else:
            cuboid, obb = iterativeBeamModeling(points_in_box,voxel_size=0.05)

        if obb is not None:
            created_beams_obb.append(obb)
    created_beams = [Beam.obb2Beam(b,comment = "stage2_create") for b in created_beams_obb]

    for cb in created_beams:
        angle = geometry.getAngleBetweenVectors(cb.unit_vector, beamGr.optimal_uvec)
        if angle < 5.:
            cb.group_id = roof_tile.id
            cb.roof_tile_id = roof_tile.id
            cb.old_id = -1
        else:
            cb.group_id = -1
            cb.roof_tile_id = roof_tile.id
            cb.old_id = -1
            cb.comment = "stage2_create_no_group"

    #Extend(able) beams loop over reliable beams
    extend_beams = []
    keep_beams = []
    extend_beams_obb = []
    for i, b in enumerate(extend_beams_2D):
        if b is None:
            extend_beams_obb.append(None)
        else:
            box, box_pts = getSearchBox3D(b, beamGr.optimal_plane, 0.13, coordinates_xy=beamGr.coordinates_xy)
            points_in_box = getPointsInBox(pcd, box)
            if temp_obb:
                obb = template.getBeamInSearchBox(pcd, box, temp_obb, template_len=template_dims[0], target_dims=(template_dims[1], template_dims[2]))
            else:
                cuboid, obb = iterativeBeamModeling(points_in_box,voxel_size=0.05)
            #o3d.visualization.draw([pcd, box, points_in_box, obb])
            extend_beams_obb.append(obb)

    #Compare extend / old versions
    for i,ext_beam in enumerate(extend_beams_obb):
        if ext_beam is None:
            b = beams_db_gr[beamGr.reliable_beams[i]]
            b.comment = "stage2_keep" # This beam has reliable size in term on extend thresholding
            b.group_id = roof_tile.id
            b.roof_tile_id = roof_tile.id

            if temp_obb is not None:
                box = b.setOBB()
                box = box.scale(1.1, box.center)
                obb = template.getBeamInSearchBox(pcd, box, temp_obb, template_len=template_dims[0], target_dims=(template_dims[1], template_dims[2]))

                if obb is None:
                        keep_beams.append(b)
                        continue

                b_k = Beam.obb2Beam(obb)

                b.axis = b_k.axis
                b.vertices = b_k.vertices
                b.unit_vector = b_k.unit_vector
                b.obb  = obb
                b.height = b_k.height
                b.width = b_k.width
                b.length = b_k.length

                keep_beams.append(b)

            else:
                keep_beams.append(b)
        else:
            # Compare old / new
            #ext_beam vs beamGr.beam_axes_2D[beamGr.reliable_beams[i]]
            current_beam_2D = beamGr.beam_axes_2D[beamGr.reliable_beams[i]]
            ext_beam_obj = Beam.obb2Beam(ext_beam)
            ext_beam_2D =  np.array( (geometry.project3DPointToPlane2D(ext_beam_obj.axis[0], beamGr.optimal_plane), geometry.project3DPointToPlane2D(ext_beam_obj.axis[1], beamGr.optimal_plane)) )

            d_ext_bottom = geometry.getPoint2LineDistance(beamGr.optimal_reference_lines[2],ext_beam_2D[0])
            d_ext_top = geometry.getPoint2LineDistance(beamGr.optimal_reference_lines[0],ext_beam_2D[1])

            d_current_bottom = geometry.getPoint2LineDistance(beamGr.optimal_reference_lines[2],current_beam_2D[0])
            d_current_top = geometry.getPoint2LineDistance(beamGr.optimal_reference_lines[0],current_beam_2D[1])

            if d_current_top < d_ext_top and d_current_bottom < d_ext_bottom:
                #Keep current beam
                b = beams_db_gr[beamGr.reliable_beams[i]]
                b.comment = "stage2_keep" # This beam is (slightly) better than the extend version
                b.group_id = roof_tile.id
                b.roof_tile_id = roof_tile.id

                if temp_obb is not None:
                    box = b.setOBB()
                    box = box.scale(1.1, box.center)
                    obb = template.getBeamInSearchBox(pcd, box, temp_obb, template_len=template_dims[0], target_dims=(template_dims[1], template_dims[2]))

                    if obb is None:
                        keep_beams.append(b)
                        continue

                    b_k = Beam.obb2Beam(obb)
                    
                    b.axis = b_k.axis
                    b.vertices = b_k.vertices
                    b.unit_vector = b_k.unit_vector
                    b.obb  = obb
                    b.height = b_k.height
                    b.width = b_k.width
                    b.length = b_k.length
                    
                    keep_beams.append(b)

                else:
                    keep_beams.append(b)

            else:
                #Define top& bottom extends of extended beam in 2D space
                b_curr = beams_db_gr[beamGr.reliable_beams[i]]
                ref_pt_tops = np.array((d_current_top, d_ext_top))
                ref_pt_bots = np.array((d_current_bottom, d_ext_bottom))
                if np.argmin(ref_pt_tops) == 0:
                    # keep top vertices of current beam
                    vertices_top = b_curr.vertices[4:]
                else:
                    # Compute top vertices based on extended beam
                    d_t = - np.dot(ext_beam_obj.axis[1],b_curr.unit_vector)
                    plane_ref_top = np.append(b_curr.unit_vector, d_t)

                    vertices_ref = b_curr.vertices[:4]
                    vertices_top = [geometry.project3DPointToPlane(v, plane_ref_top) for v in vertices_ref]

                if np.argmin(ref_pt_bots) == 0:
                    # keep bottom vertices of current beam
                    vertices_bottom = b_curr.vertices[:4]
                else:
                    # Compute bottom vertices based on extended beam
                    d_b = - np.dot(ext_beam_obj.axis[0],b_curr.unit_vector)
                    plane_ref_bot = np.append(b_curr.unit_vector, d_b)

                    vertices_ref = b_curr.vertices[:4]
                    vertices_bottom = [geometry.project3DPointToPlane(v, plane_ref_bot) for v in vertices_ref]

                beam_ext_pts = [*vertices_bottom, *vertices_top] 
                obb_ext = o3d.geometry.OrientedBoundingBox()
                obb_ext =  obb_ext.create_from_points(points=o3d.utility.Vector3dVector(beam_ext_pts))

                b_ext = Beam.obb2Beam(obb_ext)
                b_ext.comment = "stage2_extend"
                b_ext.group_id = roof_tile.id
                b_ext.roof_tile_id = roof_tile.id
                b_ext.old_id = b_curr.id
                extend_beams.append(b_ext)

                #o3d.visualization.draw([ext_beam, obb_ext])

    #Merge(able) beams
    merge_beams = []
    merge_beams_obb = []
    for i, b in enumerate(merge_beams_2D):
        box, box_pts = getSearchBox3D(b, beamGr.optimal_plane, 0.13, coordinates_xy=beamGr.coordinates_xy)
        points_in_box = getPointsInBox(pcd, box)
        if temp_obb:
            obb = template.getBeamInSearchBox(pcd, box, temp_obb, template_len=template_dims[0], target_dims=(template_dims[1], template_dims[2]), vis=False)
        else:
            cuboid, obb = iterativeBeamModeling(points_in_box,voxel_size=0.05)
        if obb is not None:
            merge_beams_obb.append(obb)
            b_merge = Beam.obb2Beam(obb)
            b_merge.comment = "stage2_merge"
            b_merge.old_id = np.min(np.array(beamGr.merge_beams[i]))
            b_merge.group_id = roof_tile.id
            b_merge.roof_tile_id= roof_tile.id
            merge_beams.append(b_merge)

        else:
            #Statistically decide on beam cross-section -> apply it from start to end of all merge-members

            merge_members_idx = beamGr.merge_beams[i]
            merge_members = [beams_db_gr[id] for id in merge_members_idx]

            #Basically choose largest one as reference
            beaam_lengths = np.array([m.length for m in merge_members])
            ref_beam_id = np.argmax(beaam_lengths)
            ref_vertices = merge_members[ref_beam_id].vertices[:4]
            ref_uvec = merge_members[ref_beam_id].unit_vector

            merge_beam_vertices = []
            for member in merge_members:
                #bottom ref plane
                d = - np.dot(ref_uvec, member.axis[0])
                plane_b = np.append(ref_uvec, d)
                for v in ref_vertices:
                    merge_beam_vertices.append(geometry.project3DPointToPlane(v, plane_b))

                #Top ref plane
                d2 = -np.dot(ref_uvec, member.axis[1])
                plane_t = np.append(ref_uvec, d2)
                for v in ref_vertices:
                    merge_beam_vertices.append(geometry.project3DPointToPlane(v, plane_t))
            
            obb_merge = o3d.geometry.OrientedBoundingBox()
            obb_merge =  obb_merge.create_from_points(points=o3d.utility.Vector3dVector(merge_beam_vertices))

            b_merge = Beam.obb2Beam(obb_merge)
            b_merge.comment = "stage2_merge"
            b_merge.old_id = np.min(np.array(merge_members_idx))
            b_merge.group_id = roof_tile.id
            b_merge.roof_tile_id = roof_tile.id
            merge_beams.append(b_merge)

    return {"beam_group":beamGr,"ignore_beams": ignore_beams, "created_beams":created_beams, "extend_beams":extend_beams, "keep_beams":keep_beams, "merge_beams":merge_beams}

def setProcessResultsOnDB(processResult, roof_db):
    # dict from processBeamGroup function is the input

    if roof_db.conn.closed == 1:
        roof_db.connect(True)

    #Update beam table
    if len(processResult['ignore_beams'])> 0:
        for b in processResult['ignore_beams']:
            sql_str = "update beam set roof_tile_id = null, comment = 'stage2_ignore' where id = " + str(b.id)
            roof_db.cursor.execute(sql_str)

    if len(processResult['merge_beams']) > 0:
        for merge_ids in processResult['beam_group'].merge_beams:
            min_id = np.min(merge_ids)
            for id in merge_ids:
                sql_str = "update beam set comment = 'stage2_merge', merge_id = " + str(min_id) + " where id = " + str(id)
                roof_db.cursor.execute(sql_str)

    #Insert to beam_new table
    if len(processResult['created_beams']) > 0:
        beams = []
        for b in processResult['created_beams']:
            if b is not None:
                beams.append(b)
        roof_db.fillBeamNewTable(beams)
    if len(processResult['keep_beams']) > 0:
        beams = []
        for b in processResult['keep_beams']:
            if b is not None:
                beams.append(b)
        roof_db.fillBeamNewTable(beams)
    if len(processResult['extend_beams']) > 0:
        beams = []
        for b in processResult['extend_beams']:
            if b is not None:
                beams.append(b)
        roof_db.fillBeamNewTable(beams)
    if len(processResult['merge_beams']) > 0:
        beams = []
        for b in processResult['merge_beams']:
            if b is not None:
                beams.append(b)
        roof_db.fillBeamNewTable(beams)

    #Insert to beam_group table
    roof_db.addBeamGroupTable(processResult['beam_group'])

def showProcessingResults(pcd, process_dict):
    obbs_keep = [b.setOBB() for b in process_dict['keep_beams']]
    obbs_merge = [b.setOBB() for b in process_dict['merge_beams']]
    obbs_create = [b.setOBB() for b in process_dict['created_beams']]
    obbs_extend = [b.setOBB() for b in process_dict['extend_beams']]

    for b in obbs_keep:
        b.color = [1,0,0]
    for b in obbs_merge:
        b.color = [0,1,0]
    for b in obbs_create:
        b.color = [0,0,1]
    for b in obbs_extend:
        b.color = [1,1,0]

    o3d.visualization.draw([pcd, *obbs_keep, *obbs_merge, *obbs_create, *obbs_extend])

    return [*obbs_keep, *obbs_merge, *obbs_create, *obbs_extend]

def beamGroupVsImage(dct):
    obbs_keep = [b for b in dct['keep_beams']]
    obbs_merge = [b for b in dct['merge_beams']]
    obbs_create = [b for b in dct['created_beams']]
    obbs_extend = [b for b in dct['extend_beams']]

    group_beams = [*obbs_keep, *obbs_merge, *obbs_create, *obbs_extend]
    
    vertices, group_obb = Beam.getOBBofBeamList(group_beams)
    group_pcd = getPointsInBox(pcd, group_obb)

    o3d.visualization.draw([pcd, group_obb, group_pcd, *gr_obbs])
   
    #2d space conversion
    pts_2d= geometry.project3DPointsToPlane2D(np.array(group_pcd.points),dct['beam_group'].optimal_plane)
    group_rects = [geometry.project3DPointsToPlane2D(b.vertices ,dct['beam_group'].optimal_plane) for b in group_beams]

    #Image conversion
    img, img_size, img_ext = imagePrc.getImageFromPoints(pts_2d)

    #Modeled beams as rectangles on image space
    group_rects_image_cs = []
    for rect in group_rects:
        image_coors = []
        for p in rect:
            image_coors.append(imagePrc.cartesian2ImageCoordinates(p, img_ext[0], img_ext[3], img_size))
        group_rects_image_cs.append(image_coors)

    binary0 = img <= 0
    binary0 = binary0.astype(np.uint8)  #convert to an unsigned byte
    binary0*=255

    plt.imshow(binary0)
    plt.show()

    imagePrc.cvLines(binary0, rects = group_rects_image_cs)


def main(config_path):
    start = datetime.now()
    config_data = exchange.readConfig(config_path)

    #Read point cloud
    points, normals, segments = exchange.readODM(config_data['db_odm1']) # Read segmented point cloud (before split)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    #Fetch beams from db
    roof_db = database.modelDatabase(config_path)
    beam_records = roof_db.getBeams(["id", "roof_tile_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"], "true")
                                    #"roof_tile_id is not null and(comment is null or comment != 'stage1_outlier')")

    beams_db_all = [Beam.Beam(id=r['id'], roof_tile_id = r['roof_tile_id'], 
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]]) for r in beam_records]

    # Cluster all beams : Kmeans
    roof_db.connect(True)
    #beams_c, clusters = Beam.kmeansClusterBeams(beams_db_all, 7,vis=True)
    beams_c, clusters = Beam.meanShiftClusterBeams(beams_db_all,vis=True)
    roof_db.cursor.execute("delete from cluster;")
    roof_db.fillClusterTable(clusters)
    roof_db.connect(True)
    for b in beams_c:
        sql_str = "update beam set cluster_id = " + str(b.cluster_id) + " where id = " + str(b.id)
        roof_db.cursor.execute(sql_str)

    #Beams only related to roof cover
    beams_db = [b for b in beams_db_all if b.roof_tile_id is not None and b.comment!= 'stage1_outlier']

    #Fetch roofTiles from db
    roof_tile_records = roof_db.getRoofTiles()
    roof_tiles_db =  [RoofTile.RoofTile(id=r['id'], plane=[float(r['plane_a']), float(r['plane_b']), float(r['plane_c']), float(r['plane_d'])], alpha_shape2d = None) for r in roof_tile_records]

    #Test code : Write a list of beams to dxf
    #obbs = [b.setOBB() for b in beams_db if b.roof_tile_id == 1]
    #exportOBBList2Dxf(obbs, "data/wing_north/obbs_rtile3.dxf", 11) 

    #Test code : To see the result of process
    #dct = processBeamGroup(beams_db, roof_tiles_db[1], pcd) # tmp_beam
    #gr_obbs = showProcessingResults(pcd, dct)
    #setProcessResultsOnDB(dct,roof_db)

    #Process all roof-tile beams
    begin = datetime.now()
    
    roof_db.connect(True)
    roof_db.cursor.execute("delete from beam_new where roof_tile_id is not null;")
    roof_db.cursor.execute("delete from beam_group where name = 'roof_tile';")
    
    for roof_tile in roof_tiles_db:
        process_gr_dict = processBeamGroup(beams_db, roof_tile, pcd)   
        gr_obbs = showProcessingResults(pcd, process_gr_dict)
        setProcessResultsOnDB(process_gr_dict,roof_db)
    end = datetime.now()
    print("Rooftiles processed:\t", (end - begin))

    #roof_tiles_idx = np.array([b.roof_tile_id for b in beams_db])
    #roof_tiles_ids = np.unique(roof_tiles_idx) # These refer to BeamGroup objects to analysis

    end = datetime.now()
    print("CAD to Database [Beam, Joint]:\t", (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Roof Cover Beams to BeamGroup's")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)