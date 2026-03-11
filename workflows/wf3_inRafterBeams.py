import sys
import argparse
import yaml
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import copy

import toolBox.database as database
import toolBox.exchange as exchange
import toolBox.geometry as geometry
import toolBox.imagePrc as imagePrc
import toolBox.template as template

from wf2_roofCoverBeams import iterativeBeamModeling

import roof.RoofTile as RoofTile
import roof.Beam as Beam
import roof.Joint as Joint
import roof.Rafter as Rafter
import roof.BeamGroup as BeamGroup

import open3d as o3d
import ezdxf
import alphashape
from shapely import wkb
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster  

from matplotlib import pyplot as plt

import cv2

def getSearchBox3D(rect_vertices_2d, refPlane, plane_offset):

    #Get parallel planes of reference plane (Ref plane is assumed as central plane)
    pl1, pl2 = geometry.getParallelPlanes(refPlane, plane_offset)

    #Reproject defined 2D point to 3D planes
    p1 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[0], pl1)
    p2 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[1], pl1)
    p3 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[2], pl1)
    p4 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[3], pl1)

    p5 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[0], pl2)
    p6 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[1], pl2)
    p7 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[2], pl2)
    p8 = geometry.reproject2DPointToPlane3D(rect_vertices_2d[3], pl2)

    pts = [p1,p2,p3,p4,p5,p6,p7,p8]

    obb = o3d.geometry.OrientedBoundingBox()
    obb =  obb.create_from_points(points=o3d.utility.Vector3dVector(pts))
    return obb, pts

def setRafterIds(beams, rafters):
    for b in beams:
        b_hull  = b.getConvexHull3D()
        for r in rafters:
            #r.setBeamObjects(beams)
            r_hull = r.getConvexHull3D()

            r_buff2d = r.convex_hull.buffer(0.5, cap_style = 2, join_style= 2)
            vertices_2d = np.dstack(r_buff2d.boundary.xy).tolist()[0][:-1]
            vertices_3d = [geometry.reproject2DPointToPlane3D(v, r.plane) for v in vertices_2d]

            r_pcd = o3d.geometry.PointCloud()
            r_pcd.points = o3d.utility.Vector3dVector(np.vstack((np.array(r_hull.vertices), vertices_3d))) 
            r_hull2, _ = r_pcd.compute_convex_hull()

            if b_hull.is_intersecting(r_hull2):
                #Check if rafter plane is perpendicular to beam
                angle = np.abs(geometry.getAngleBetweenVectors(r.plane[:3], b.unit_vector))
                
                if  (angle > 84. and angle < 96.):
                    b.rafter_id = r.id
                else:
                    b.truss_id = -1 # means this is a truss candidate!

def processRafterBeams(pcd, group_beams, rafter, show_results=True):

    vertices1, group_obb1 = Beam.getOBBofBeamList(group_beams)
    vertices = np.vstack((np.array(rafter.convex_hull_3d.vertices),vertices1)) #Extend the obb using rafter convex_hull
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(vertices)
    group_obb = tmp_pcd.get_oriented_bounding_box()
    group_obb.scale(1.2, group_obb.get_center())
    group_pcd = template.getPointsInBox(pcd, group_obb)

    o3d.visualization.draw([pcd, group_obb, group_pcd])
   
    #2d space conversion
    pts_2d= geometry.project3DPointsToPlane2D(np.array(group_pcd.points),rafter.plane)
    group_rects = [geometry.project3DPointsToPlane2D(b.vertices ,rafter.plane) for b in group_beams] # This is multi point object including 8 vertices (to get rectangle MBR computation is necessary)
    group_lines = [geometry.project3DPointsToPlane2D(b.axis ,rafter.plane) for b in group_beams]

    #Image conversion
    img, img_size, img_ext = imagePrc.getImageFromPoints(pts_2d,scale = 1)

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

    plt.imshow(binary0)
    plt.show()

    plt.hist(img.flatten(), bins=None, ec="k")
    #plt.xticks((0,1))
    plt.show()



    #Search for beams on raw image
    lines = imagePrc.getLineSegments(img) # get oriented lines of beam sides (left of line is beam surface)
    rect_mpts, boxes = imagePrc.linesToRects(img, lines) # line matches to beam surface rectangles

    if show_results:
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

    #Compare reference lines/rects to model lines/rects
    #Comparison on image coord sys
    ref_rects = [ MultiPoint(r).minimum_rotated_rectangle for r in rect_mpts]
    ref_lines = imagePrc.rectangles2Lines(boxes)
    mod_rects = [ MultiPoint(r).minimum_rotated_rectangle for r in group_rects_image_cs]
    #Model lines : group_lines_image_cs

    #1- Matching reference->model
    match_idx = []
    for i, m in enumerate(mod_rects):
        #first check if existing model valid!
        mask = np.zeros((img.shape[0],img.shape[1],1), np.uint8)
        r = cv2.minAreaRect(np.array(group_rects_image_cs[i]))
        b = cv2.boxPoints(r)
        b = np.int0(b)
        cv2.drawContours(mask, contours=[b], 
                     contourIdx = 0,
                     color=(255), thickness=-1)
        overlap_img = cv2.bitwise_and(img, mask)
        mod_cnt = cv2.countNonZero(mask)
        overlap_cnt = cv2.countNonZero(overlap_img)
        overlap_ratio = float(overlap_cnt) / float(mod_cnt)
        if overlap_ratio < 0.7:
            match_idx.append(-1) # Model is invalid
            continue
        else:
            candidates = []
            mod_vec = geometry.getUnitVector(np.array(group_lines_image_cs[i][1]) - np.array(group_lines_image_cs[i][0]))
            #center = Point(np.mean(group_lines_image_cs[i], axis =0))
            mod_line = LineString(group_lines_image_cs[i])
            
            for j, r in enumerate(ref_rects):
                if r.intersects(mod_line):
                    candidates.append(j)
            
            if len(candidates)>1:
                #Choose correct reference rect
                angles = []
                for c in candidates:
                    ref_vec = geometry.getUnitVector(np.array(ref_lines[c][1]) - np.array(ref_lines[c][0]))           
                    alpha = geometry.getAngleBetweenVectors(ref_vec, mod_vec)
                    alpha = 180 - alpha if alpha > 90 else alpha
                    #ref_eq = geometry.getLineEquation2D(ref_lines[c][0],ref_lines[c][1])
                    angles.append(alpha)
                    #equations.append(ref_eq)
                    #angles2.append(geometry.getAngleBetweenLines2D(ref_eq,mod_eq))
                angles = np.array(angles)            
                candidate = candidates[np.argmin(angles)] if np.min(angles) < 5. else None           
                match_idx.append(candidate)
            elif len(candidates) == 1:
                match_idx.append(candidates[0])
            elif len(candidates) == 0:
                match_idx.append(None)

    if show_results:
        #Show results of existing(model) beams to detected (ref) beams
        matching_img = cv2.merge((img,img,img))
        colors = imagePrc.getRandomColors(len(mod_rects), True)
        for i,line in enumerate(group_lines_image_cs):
            x1,y1 = line[0]
            x2,y2 = line[1]
            cv2.line(matching_img, (int(x1),int(y1)), (int(x2),int(y2)), (int(colors[i][0]), int(colors[i][1]), int(colors[i][2])), 2)        
            if match_idx[i] is not None and match_idx[i] != -1:
                box = np.int0(boxes[match_idx[i]])
                cv2.drawContours(matching_img, [box],0,(int(colors[i][0]), int(colors[i][1]), int(colors[i][2])),2)
        cv2.imshow("Matching", matching_img)
        cv2.waitKey()
    
    #Start comparison & decision on "keep", "ignore", "extend", "merge", "create"
    match_idx = np.array(match_idx)
    # Check existing beams to ignore, keep, extend or merge
    ignore_beams = []
    keep_beams = []
    extend_beams = []
    merge_beams = []
    create_beams = []
    for i,b in enumerate(group_beams):
        if match_idx[i] is None or match_idx[i] == -1:
            #Ignore case
            b.comment = "stage3_ignore"
            ignore_beams.append(b)
        else:
            ref_id = match_idx[i]
            merge_beam_ids = np.argwhere(match_idx == ref_id).flatten() # Potential merge beams corresponds to same reference rectangle

            if len(merge_beam_ids) == 1:
                # Keep or extend case
                #image cs to 2d object cs ref_lines
                p1,p2 = ref_lines[ref_id]
                x1, y1 = imagePrc.image2CartesianCoordinates((p1[0],p1[1]), img_ext[0], img_ext[3], img_size)
                x2, y2 = imagePrc.image2CartesianCoordinates((p2[0],p2[1]), img_ext[0], img_ext[3], img_size)
                ref_line_obj_cs_2d = np.array(((x1,y1),(x2,y2)))

                extended_rect = np.vstack((group_rects[i], ref_line_obj_cs_2d)) # model rect pts + ref line pts in obj cs
                extended_rect_pts_2d = np.dstack(MultiPoint(extended_rect).minimum_rotated_rectangle.boundary.xy).tolist()[0][:-1]
                extended_line_2d = imagePrc.rectangles2Lines([extended_rect_pts_2d])[0]

                ext_point1_2d = None
                ext_point2_2d = None
                #if ref_rects[ref_id].contains(Point(group_lines_image_cs[i][0])):
                #first point of model line in the rect
                d0 = geometry.getDistance(group_lines[i][0], extended_line_2d[0])# ref_line_obj_cs_2d[0])
                d1 = geometry.getDistance(group_lines[i][1], extended_line_2d[0])# ref_line_obj_cs_2d[0])
                if np.min((d0,d1)) > 0.05:
                    ext_point1_2d = extended_line_2d[0]

                #if ref_rects[ref_id].contains(Point(group_lines_image_cs[i][1])):
                #first point of model line in the rect
                d2 = geometry.getDistance(group_lines[i][0], extended_line_2d[1])# ref_line_obj_cs_2d[1])
                d3 = geometry.getDistance(group_lines[i][1], extended_line_2d[1])# ref_line_obj_cs_2d[1])
                if np.min((d2,d3)) > 0.05:
                    ext_point2_2d = extended_line_2d[1]

                if ext_point1_2d is None and ext_point2_2d is None:
                    #Keep this beam as it is
                    b.comment = "stage_3_keep"
                    b.obb.color = [0.5,0.5,0.5] #Gray for keep
                    keep_beams.append(b)

                else:
                    #Extend this beam
                    #extended_rect = np.vstack((group_rects[i], ref_line_obj_cs_2d)) # model rect pts + ref line pts in obj cs
                    #extended_rect_pts_2d = np.dstack(MultiPoint(extended_rect).minimum_rotated_rectangle.boundary.xy).tolist()[0][:-1]
                    #extended_line_2d = imagePrc.rectangles2Lines([extended_rect_pts_2d])
                    #ext_point1_2d = extended_line_2d[0][0]
                    #ext_point2_2d = extended_line_2d[0][1]

                    ext_point1_3d = None
                    ext_point2_3d = None
                    if ext_point1_2d is not None:
                        ext_point1_3d = geometry.reproject2DPointToPlane3D(ext_point1_2d,rafter.plane)
                    if ext_point2_2d is not None:
                        ext_point2_3d = geometry.reproject2DPointToPlane3D(ext_point2_2d,rafter.plane)
                    old_obb = copy.deepcopy(b.obb)
                    b.extendAlongLongitudinalAxis(ext_point1_3d, ext_point2_3d, False)
                    b.comment = "stage3_extend"
                    b.obb.color = [1.,1.,0.] #Yellow

                    #o3d.visualization.draw([pcd, b.obb, old_obb])

                    extend_beams.append(b)

            if len(merge_beam_ids) > 1:
                #Merge case
                b.comment = "stage3_merge"
                merge_beams_members = [group_beams[i] for i in merge_beam_ids if group_beams[i].comment != "stage3_merge"]

                if len(merge_beams_members):
                    #reference rect in object cs
                    p1,p2 = ref_lines[ref_id]
                    x1, y1 = imagePrc.image2CartesianCoordinates((p1[0],p1[1]), img_ext[0], img_ext[3], img_size)
                    x2, y2 = imagePrc.image2CartesianCoordinates((p2[0],p2[1]), img_ext[0], img_ext[3], img_size)
                    ref_line_obj_cs_2d = np.array(((x1,y1),(x2,y2)))
                    ref_line_obj_cs_3d = np.vstack([geometry.reproject2DPointToPlane3D(ref_line_obj_cs_2d[0],rafter.plane), geometry.reproject2DPointToPlane3D(ref_line_obj_cs_2d[1],rafter.plane)])
                    ref_len_3d = geometry.getDistance(ref_line_obj_cs_3d[0], ref_line_obj_cs_3d[1])

                    merge_beams_members.append(b)
                    for be in merge_beams_members:
                        be.comment = "stage3_merge"
                        be.old_id = b.id
                    
                    avg_ext = np.mean(np.array([(b.height, b.width) for b in merge_beams_members]), axis = 0)

                    # rotation of the first beam (b) is reference at the begining
                    tmp_obb = o3d.geometry.OrientedBoundingBox(np.mean(ref_line_obj_cs_3d, axis = 0), 
                                                               b.obb.R, (ref_len_3d, avg_ext[0], avg_ext[1]))

                    tmp_sample = template.sampleOBB(tmp_obb, ref_len_3d * 1000)

                    #Accurately locate the merged beam with ICP
                    init = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
                    icp_transform = o3d.pipelines.registration.registration_icp(
                        tmp_sample, pcd, 0.2,init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                    tmp_sample2 = copy.deepcopy(tmp_sample)
                    tmp_sample2 = tmp_sample2.transform(icp_transform.transformation)
                    box_pts = o3d.geometry.PointCloud()
                    box_pts.points = o3d.utility.Vector3dVector(tmp_obb.get_box_points())                    
                    box_pts = box_pts.transform(icp_transform.transformation)
                    tmp_obb2 = box_pts.get_oriented_bounding_box() #this is alignede obb
                    
                    merged_beam = Beam.obb2Beam(tmp_obb2)
                    merged_beam.comment = "stage3_merge"
                    merged_beam.old_id = b.id
                    merged_beam.cluster_id = b.cluster_id
                    merged_beam.rafter_id = b.rafter_id
                    merged_beam.truss_id = b.truss_id
                    merged_beam.obb.color = [0.,0.,1.] #Blue

                    merge_beams.append(merged_beam)
    
    #Create beams investigation
    #Do not forget to ignore roof-cover beams (already modeled&refined and define the rafters)

    b1_center_3d = np.mean(rafter.b1_obj.axis, axis=0)
    b1_center_2d = geometry.project3DPointToPlane2D(b1_center_3d, rafter.plane)
    b1_center_2d_img = imagePrc.cartesian2ImageCoordinates(b1_center_2d, img_ext[0], img_ext[3], img_size)
 
    b2_center_3d = np.mean(rafter.b2_obj.axis, axis=0)
    b2_center_2d = geometry.project3DPointToPlane2D(b2_center_3d, rafter.plane)
    b2_center_2d_img = imagePrc.cartesian2ImageCoordinates(b2_center_2d, img_ext[0], img_ext[3], img_size)

    for i, mpts in enumerate(rect_mpts):
        if i not in match_idx:
            if not (ref_rects[i].contains(Point(b1_center_2d_img)) or ref_rects[i].contains(Point(b2_center_2d_img))):
                #Here check integrity to rafter and ignore narrow candidates

                mpts_2d_obj = MultiPoint(np.array([imagePrc.image2CartesianCoordinates(pt, img_ext[0], img_ext[3], img_size) for pt in mpts]))

                r_buff2d = rafter.convex_hull.buffer(0.5, cap_style = 2, join_style= 2)

                if r_buff2d.convex_hull.contains(Point(mpts_2d_obj.centroid)):

                    rect_2d_obj = mpts_2d_obj.minimum_rotated_rectangle
                    poly_pts = np.dstack(rect_2d_obj.boundary.xy).tolist()[0][:-1]

                    dists = [geometry.getDistance(poly_pts[0], p) for j, p in enumerate(poly_pts) if j != 0]

                    if min(dists) >= 0.05:
                        #minimum thicknes of a beam side

                        #Prepare search box for beam search
                        rect_buff2d = rect_2d_obj.buffer(0.05, cap_style = 2, join_style= 2)
                        vrt_2d = np.dstack(rect_buff2d.boundary.xy).tolist()[0][:-1]

                        plane_offset = float(group_obb1.extent[2]) / 2.
                        search_box, search_pts = getSearchBox3D(vrt_2d, rafter.plane, plane_offset)

                        search_pcd = template.getPointsInBox(group_pcd, search_box)
                        
                        voxel_size =  0.05 if max(dists) > 2. else 0.015
                        cuboid, obb1 = iterativeBeamModeling(search_pcd,voxel_size = voxel_size)

                        if obb1 is not None:
                            #TODO recheck rafter membership?
                            beam1 = Beam.obb2Beam(obb1)
                            beam1.comment = "stage3_create"
                            beam1.rafter_id = rafter.id
                            beam1.obb.color = [1.,0.,0.] #Red                       
                            create_beams.append(beam1)
                        else:
                            vert_2d = np.dstack(rect_2d_obj.boundary.xy).tolist()[0][:-1]
                            dst = np.array([geometry.getDistance(v, vert_2d[0]) for j, v in enumerate(vert_2d) if j >0])
                            tmp_offset = (np.min(dst) / 2.) + 0.000001
                            tmp_box, tmp_pts = getSearchBox3D(vert_2d, rafter.plane, tmp_offset)
                            
                            obb2= template.getBeamInSearchBox(group_pcd, search_box, tmp_box, template_len=tmp_box.extent[0], 
                                                              target_dims=(tmp_box.extent[1], tmp_box.extent[2]))
                            if obb2 is not None:
                                beam2 = Beam.obb2Beam(obb2)
                                beam2.comment = "stage3_create"
                                beam2.rafter_id = rafter.id
                                beam2.obb.color = [0.,0.,0.] #Black
                                create_beams.append(beam2)

    return {"ignore_beams": ignore_beams, "created_beams":create_beams, "extend_beams":extend_beams, "keep_beams":keep_beams, "merge_beams":merge_beams, "group_pcd":group_pcd}

def refineRafterBeams(pcd, b1, b2, rafter):
    #TODO: Rafter cover beams needs to be connected to cover
    #but sometimes they are located a bit inside

    # Check the validity of rafter cover beams
    # Refine if it needs to be replaced/moved
    vertices1, group_obb1 = Beam.getOBBofBeamList([b1,b2])
    vertices = np.vstack((np.array(rafter.convex_hull_3d.vertices),vertices1)) #Extend the obb using rafter convex_hull
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(vertices)
    group_obb = tmp_pcd.get_oriented_bounding_box()
    group_obb.scale(1.2, group_obb.get_center())
    group_pcd = template.getPointsInBox(pcd, group_obb)

    b1_obb = b1.setOBB()
    b2_obb = b2.setOBB()
    o3d.visualization.draw([group_obb, group_pcd, b1_obb, b2_obb])


    b1_search = copy.deepcopy(b1_obb)
    b1_search.scale(3., b1_search.get_center())

    b1_ref= template.getBeamInSearchBox(group_pcd, b1_search, b1_obb, template_len=b1_obb.extent[0], 
                                                              target_dims=(b1_obb.extent[1], b1_obb.extent[2]))

    o3d.visualization.draw([group_obb, group_pcd, b1_obb, b2_obb, b1_search, b1_ref])

def templateAccumulate(pcd, beams_of_rafters, rafters_db, rafter_type = "primary", check_symmetry = True):

    refinedRafterMembers = []
    #--First loop is to singulariz e (merge) the rafter members
    #and to eliminate the invalid beams--
    for i,rafter in enumerate(rafters_db):
        if rafter.truss_type == rafter_type:
            refined_members = {
            "rafter_id"     : None,
            "rafter_pcd"    : None,
            "rafter_plane"  : None,
            "cluster_idx"   : [], # list of cluster idx
            "overlap_ratios": [], # list of beam->image overlaps 
            "widths"        : [], # list of beam widths
            "all_beams"     : [], # idx of all beal beams in correct order
            "keep_beams"    : [], # list of beam idx
            "merge_beams"   : [], # list of merge dictionary           
            "ignore_beams"  : [], # list of beam idx
            "create_beams"  : [], # list of create dictionary
            "beam_rects_img": [], # Beam rectangles in image cs
            "beam_lines_img": [], # Beam lines in image cs
            "image_info_2d" : [None, None, None] # img, img_size, img_ext   
            }
            refined_members["rafter_id"] = rafter.id
            #refined_members["rafter_plane"] = rafter.plane
            refined_members['all_beams'] = beams_of_rafters[i]

            #Re-compute the plane based on all rafter beams
            all_beam_axes = np.array([b.axis for b in refined_members['all_beams']])
            all_beam_axes_pts = np.vstack((all_beam_axes[:,0], all_beam_axes[:,1]))
            new_plane,_,_ = geometry.getPlaneLS(np.array(all_beam_axes_pts))
            refined_members["rafter_plane"] = new_plane[:4]

            obbs = [b.obb for b in beams_of_rafters[i]]
            #o3d.visualization.draw([*obbs, rafter.convex_hull_3d])
            
            vertices1, group_obb1 = Beam.getOBBofBeamList(beams_of_rafters[i])
            vertices = np.vstack((np.array(rafter.convex_hull_3d.vertices),vertices1))
            tmp_pcd = o3d.geometry.PointCloud()
            tmp_pcd.points = o3d.utility.Vector3dVector(vertices)
            group_obb = tmp_pcd.get_oriented_bounding_box()
            group_obb.scale(1.2, group_obb.get_center())
            group_pcd = template.getPointsInBox(pcd, group_obb) #Pcd of current rafter (a)
            
            refined_members["rafter_pcd"] = group_pcd

            #2d space conversion
            pts_2d= geometry.project3DPointsToPlane2D(np.array(group_pcd.points),refined_members["rafter_plane"])
            group_rects = [geometry.project3DPointsToPlane2D(b.vertices ,refined_members["rafter_plane"]) for b in beams_of_rafters[i]] # This is multi point object including 8 vertices (to get rectangle MBR computation is necessary)
            group_lines = [geometry.project3DPointsToPlane2D(b.axis ,refined_members["rafter_plane"]) for b in beams_of_rafters[i]]            
            
            #Image conversion
            img, img_size, img_ext = imagePrc.getImageFromPoints(pts_2d,scale = 2)
            binary0 = img <= 0
            binary0 = binary0.astype(np.uint8)  #convert to an unsigned byte
            binary0*=255   
            img = cv2.bitwise_not(binary0) # White pixels are beams
            refined_members['image_info_2d'] = [img, img_size, img_ext]

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

            refined_members['beam_rects_img'] = group_rects_image_cs
            refined_members['beam_lines_img'] = group_lines_image_cs

            #Compute model -> Point cloud overlap ratio in image space
            #to estimate model reliability
            overlap_ratios = [imagePrc.getImageRectListOverlapRatio(img,[r]) for r in group_rects_image_cs]
            beam_widths = [b.width for b in beams_of_rafters[i]]   
            reliable_beam_idx = np.where(np.bitwise_and((np.array(overlap_ratios)) > 0.7, (np.array(beam_widths) > 0.04)))
            ignore_beam_idx = [b.id for j,b in enumerate(beams_of_rafters[i]) if j not in reliable_beam_idx[0]]

            for j, beam in enumerate(beams_of_rafters[i]):
                beam.img_overlap = overlap_ratios[j]
            
            refined_members['ignore_beams'] = ignore_beam_idx
            refined_members['overlap_ratios'] = overlap_ratios
            refined_members['widths'] = beam_widths

            #ignored_beams = [b.obb for b in  refined_members['all_beams'] if b.id in ignore_beam_idx]
            #for b in ignored_beams:
            #    b.color = [1,0,0]
            #o3d.visualization.draw([*ignored_beams, *obbs])

            reliable_beams = [b for j,b in enumerate(beams_of_rafters[i]) if j in reliable_beam_idx[0]]
            keep_beam_idx = [b.id for j,b in enumerate(beams_of_rafters[i]) if j in reliable_beam_idx[0]] #Still has mergable beams

            # Cluster id based Merge beams searching
            beam_clusters = np.array([b.cluster_id for b in reliable_beams])
            cls_names, cls_counts = np.unique(beam_clusters, return_counts=True)
            repetitive_cls_names = cls_names[np.where(cls_counts>1)]

            for cluster in repetitive_cls_names:
                beams_of_cluster = [b for j,b in enumerate(beams_of_rafters[i]) if j in reliable_beam_idx[0] and b.cluster_id == cluster]

                #dist_map = [geometry.getPoint2VectorDistance3D(beams_of_cluster[0].axis[0], beams_of_cluster[0].axis[1], np.mean(b.axis, axis=0))
                #           for b in beams_of_cluster]
                merge_matches = Beam.getMergeMatches(beams_of_cluster)
                merge_overlap_scores = []
                merge_beams = []
                if len(merge_matches[0]):
                    for matches in merge_matches:
                        beams_to_merge = [b.id for j,b in enumerate(beams_of_cluster) if j in matches]                   
                        current_beam_idx = [k for k,bb in enumerate(beams_of_rafters[i]) if bb.id in beams_to_merge]
                        merge_beams.append(beams_to_merge)
                       
                        merged_vetices = [] # All vertices from merged rectangles
                        for id in current_beam_idx:
                            merged_vetices.append(group_rects_image_cs[id])
                    
                        merged_vetices = np.vstack(merged_vetices)
                        overlap_ratio = imagePrc.getImageRectListOverlapRatio(img,[merged_vetices])
                        merge_overlap_scores.append(overlap_ratio)
                        merge_status = True if overlap_ratio >0.7 else False
                        merge_dct = {'merge_idx': beams_to_merge, 'merge_score': overlap_ratio, 'merge_status': merge_status, 'beam': None}                                            
                        refined_members['merge_beams'].append(merge_dct)

            #Generate merged beams
            merged_beams_idx = []        
            if len (refined_members['merge_beams']):
                for merge_dict in refined_members['merge_beams']:
                    if merge_dict['merge_status'] is True:                       
                        ignore_beam_idx.extend(merge_dict['merge_idx'])
                        m_beams = [b for b in refined_members['all_beams'] if b.id in merge_dict['merge_idx']]
                        merged_beams_idx.extend([b.id for b in m_beams])
                        m_obb = Beam.mergeBeams(m_beams, refined_members['rafter_pcd'])
                        m_beam = Beam.obb2Beam(m_obb)
                        m_beam.comment = "stage3_merge"
                        m_beam.old_id = np.min(merge_dict['merge_idx'])
                        m_beam.cluster_id = m_beams[0].cluster_id
                        m_beam.rafter_id = m_beams[0].rafter_id
                        merge_dict['beam'] = m_beam

            keep_beams_idx2 = [id for id in keep_beam_idx if id not in merged_beams_idx] # here merged beams are excluded
            refined_members['keep_beams'] = keep_beams_idx2

            refinedRafterMembers.append(refined_members)
            #o3d.visualization.draw(obbs)
               
    # Symmetry check & create beams
    if rafter_type == "primary" and check_symmetry:
        for refined in refinedRafterMembers:
            ref_beams = [b for b in refined['all_beams'] if b.id in refined['keep_beams']]
            ref_beams.extend([m['beam'] for m in refined['merge_beams'] if m['merge_status'] is True ])
            checkRafterSymmetry(ref_beams, refined)

    #--Second loop is to create missing Beams looking to the other rafters
    #Assume the first rafter as template reference
    template_id = 0
    for stage in range(2):
        #First stage for template accumulation
        #Second stage is for re-check of each rafters

        if stage == 1:
            beam_counts = []
            for mem in refinedRafterMembers:
                beam_count = len(mem['keep_beams']) + len(mem['create_beams'])
                for b in mem['merge_beams']:
                    if b['merge_status'] is True:
                        beam_count += 1
                beam_counts.append(beam_count)

            template_id = np.argmax(np.array(beam_counts)).flatten()[0]

        if stage == 1 and rafter_type == "primary" and check_symmetry:
            #check if template is still symmetric
            tmp_beams = [b for b in refinedRafterMembers[template_id]['all_beams'] if b.id in refinedRafterMembers[template_id]['keep_beams']]
            tmp_beams.extend([m['beam'] for m in refinedRafterMembers[template_id]['merge_beams'] if m['merge_status'] is True ])
            tmp_beams.extend([c['create_beam'] for c in refinedRafterMembers[template_id]['create_beams']])
            for i,b in enumerate(tmp_beams):
                if b.id == -1:
                    b.id -= i
            checkRafterSymmetry(tmp_beams, refinedRafterMembers[template_id], force_detection = True)

        for mem_id,rafter_members in enumerate(refinedRafterMembers):     
            
            if mem_id != template_id:
                template_rafter = copy.deepcopy(refinedRafterMembers[template_id])
                template_beams = [b for b in template_rafter['all_beams'] if b.id in template_rafter['keep_beams'] and b.comment != 'stage1_outlier']
                template_beams.extend([m['beam'] for m in template_rafter['merge_beams'] if m['merge_status'] is True ])
                template_beams.extend([c['create_beam'] for c in template_rafter['create_beams']])
                template_beams_pcd_copy = getSampledPCDofBeamList(template_beams) # In correct position of the first (template rafter)
                template_beams_copy = copy.deepcopy(template_beams) # In correct position beams
                
                #Start comparison
                target_beams = [b for b in rafter_members['all_beams'] if b.id in rafter_members['keep_beams']]
                target_beams.extend([m['beam'] for m in rafter_members['merge_beams'] if m['merge_status'] is True])
                target_beams.extend([c['create_beam'] for c in rafter_members['create_beams']])
                target_obbs = [b.obb for b in target_beams]
                
                #For 1->1 comparison first move template beams to target position
                trans_vec = rafter_members["rafter_pcd"].get_center() - template_rafter['rafter_pcd'].get_center()
                trans_obbs = [b.obb.translate(trans_vec, True) for b in template_beams]       
                
                template_beams_pcd = getSampledPCDofBeamList(template_beams) # This will be ICP transformed to the target rafter position
                target_beams_pcd = getSampledPCDofBeamList(target_beams)
                
                #template_beams_pcd_copy = copy.deepcopy(template_beams_pcd)
                target_beams_pcd_copy = copy.deepcopy(target_beams_pcd)
                
                if rafter_type == "primary":
                    icp_transform = template.getICPTransform(template_beams_pcd,target_beams_pcd, 0.2, 50)
                if rafter_type == "secondary":
                    icp_transform = template.getICPTransform(template_beams_pcd,rafter_members["rafter_pcd"], 0.2, 50)
                icp_pcd = template_beams_pcd.transform(icp_transform.transformation)
                
                trans_obbs = [template.translateOBB(obb, icp_transform.transformation) for obb in trans_obbs] #Trans obbs are translated + ICP refined
                trans_beams = [Beam.obb2Beam(obb) for obb in trans_obbs] # trans_beams are just for comparison!!
                for o in trans_obbs:
                    o.color = [1,0,0]
                
                #o3d.visualization.draw([template_beams_pcd_copy, template_beams_pcd, target_beams_pcd_copy, target_beams_pcd])
                
                #Reverse transformation estimation
                current_rafter_obbs = copy.deepcopy(target_obbs)
                reverse_icp = template.getReverseTransformationMat(icp_transform.transformation)        
                reverse_obbs = [b.translate(-trans_vec, True) for b in current_rafter_obbs]
                target_beams_pcd_copy.translate(-trans_vec, True)
                #template_beams_pcd_copy.translate(-trans_vec, True)
                if rafter_type == "primary":
                    reverse_icp_check = template.getICPTransform(target_beams_pcd_copy,template_beams_pcd_copy, 0.2, 50)
                if rafter_type == "secondary":
                    reverse_icp_check = template.getICPTransform(target_beams_pcd_copy,template_rafter['rafter_pcd'], 0.2, 50)

                icp_pcd2 = target_beams_pcd_copy.transform(reverse_icp_check.transformation)
                reverse_obbs = [template.translateOBB(obb, reverse_icp_check.transformation) for obb in current_rafter_obbs]
                
                #o3d.visualization.draw([template_beams_pcd_copy, template_beams_pcd, target_beams_pcd_copy, target_beams_pcd])
                #o3d.visualization.draw([*target_obbs, *reverse_obbs, rafter_members["rafter_pcd"], template_rafter['rafter_pcd']])                               
                #o3d.visualization.draw([*target_obbs, *trans_obbs, template_beams_pcd, target_beams_pcd ])
                
                beam_matches = [] # tmpidx , targedidx
                for i,tmp_beam in enumerate(trans_beams):
                    for j, target_beam in enumerate(target_beams):
                        #Check angle match first
                        if geometry.getAngleBetweenVectors(tmp_beam.unit_vector, target_beam.unit_vector) <5.:
                            ch1 = tmp_beam.getConvexHull3D()
                            ch2 = target_beam.getConvexHull3D()
                            if ch1.is_intersecting(ch2):
                                beam_matches.append([i, j])
                
                #Case1: Template has beam but target has not
                tmp_create_candidates = []
                for id in np.arange(len(trans_beams)):
                    if  len(np.array(beam_matches).transpose()):
                        existing_idx = np.array(beam_matches).transpose()[0]
                    else:
                        existing_idx= []
                    if id not in existing_idx:
                        tmp_create_candidates.append(id)
                if len(tmp_create_candidates):
                    #Check &Create
                    #Create position is current rafter
                    for idx in tmp_create_candidates:
                        create_obb = template.getRegisteredOBB(trans_obbs[idx], rafter_members["rafter_pcd"], 0.2, 50)
                        create_beam = Beam.obb2Beam(create_obb)
                        img, img_size, img_ext = rafter_members['image_info_2d']
                        tmp_rafter_plane = rafter_members["rafter_plane"]
                        overlap_ratio = computeOverlapRatio(create_beam, tmp_rafter_plane,img, img_size, img_ext)

                        #o3d.visualization.draw([create_obb, *target_obbs])
                        if overlap_ratio > 0.7:
                            #o3d.visualization.draw([rafter_members["rafter_pcd"], create_beam.obb])      
                            create_beam.comment = "stage3_create_tmp"
                            create_beam.cluster_id = template_beams[idx].cluster_id
                            create_beam.rafter_id = rafter_members["rafter_id"]
                            create_dct = {'create_score': overlap_ratio, 'create_status': "template", 'create_beam': create_beam}
                            rafter_members['create_beams'].append(create_dct)
                
                #Case2: Target has beam but template has not
                trg_create_candidates = []
                for id in np.arange(len(target_beams)):
                    if len(np.array(beam_matches).transpose()):
                        existing_idx = np.array(beam_matches).transpose()[1]
                    else:
                        existing_idx = []
                    if id not in existing_idx:
                        trg_create_candidates.append(id)
                if len(trg_create_candidates):
                    #Check &Create
                    #Create position is template's original position: use reverse_obbs
                    for idx in trg_create_candidates:
                        create_obb = template.getRegisteredOBB(reverse_obbs[idx], refinedRafterMembers[template_id]["rafter_pcd"], 0.2, 50)
                        create_beam = Beam.obb2Beam(create_obb)#reverse_obbs[idx])
                        tmp_img,tmp_img_size,tmp_img_ext = refinedRafterMembers[template_id]['image_info_2d']
                        tmp_rafter_plane = refinedRafterMembers[template_id]["rafter_plane"]
                        overlap_ratio = computeOverlapRatio(create_beam, tmp_rafter_plane,tmp_img, tmp_img_size, tmp_img_ext)
                        if overlap_ratio > 0.7:
                            #o3d.visualization.draw([refinedRafterMembers[template_id]["rafter_pcd"], create_beam.obb])      
                            create_beam.comment = "stage3_create_tmp"
                            create_beam.cluster_id = target_beams[idx].cluster_id
                            create_beam.rafter_id = refinedRafterMembers[template_id]["rafter_id"]
                            create_dct = {'create_score': overlap_ratio, 'create_status': "template", 'create_beam': create_beam}
                            refinedRafterMembers[template_id]['create_beams'].append(create_dct)
                
                if stage == 1:
                    #Check dimension equality in the second stage
                    # 1-1 comparison of template and target
                    for tmp_idx, target_idx in beam_matches:
                        # Dimension equality check for extend/shorten decision
                        tmp_beam = trans_beams[tmp_idx] # BeamIds are always -1 , correct ids are on template_beams
                        trg_beam = target_beams[target_idx]
                        if abs(max(tmp_beam.obb.extent) - max(trg_beam.obb.extent)) > 0.5:
                            #o3d.visualization.draw([tmp_beam.obb, trg_beam.obb, template_beams_pcd, target_beams_pcd])
                        
                            tmp_beam_on_position = template_beams_copy[tmp_idx]
                            tmp_img,tmp_img_size,tmp_img_ext = refinedRafterMembers[template_id]['image_info_2d']
                            tmp_rafter_plane = refinedRafterMembers[template_id]["rafter_plane"]
                    
                            trg_beam_on_position = template_beams_copy[tmp_idx]
                            trg_img,trg_img_size,trg_img_ext = rafter_members['image_info_2d']
                            trg_rafter_plane = rafter_members["rafter_plane"]
                    
                            tmp_ratio = computeOverlapRatio(tmp_beam_on_position, tmp_rafter_plane,tmp_img, tmp_img_size, tmp_img_ext)
                            trg_ratio = computeOverlapRatio(trg_beam, trg_rafter_plane,trg_img, trg_img_size, trg_img_ext)
                    
                            if tmp_ratio >= trg_ratio:
                                #Target rafter beam will be re-extended
                     
                                print('TODO: Dimensional correction based on template')
                    
                            else:
                                #Template rafter beam will be updated
                                print('TODO: Dimensional correction on template based on reference rafter')
    return refinedRafterMembers

def getSampledPCDofBeamList(beams):
    pcd = o3d.geometry.PointCloud()
    for b in beams:
        pcd += template.sampleOBB(b.obb,max(b.obb.extent) * 1000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=12))
    return pcd

def checkRafterSymmetry(beams, refined, force_detection= False):
    rafter_pcd = refined['rafter_pcd']
    img,img_size,img_ext = refined['image_info_2d']
    rafter_plane = refined["rafter_plane"]
    #Is rafter symmetric
    rafter_center = rafter_pcd.get_center()
    #1 Detect the Vertical central beam as symmetry reference
    angles = np.array([geometry.getAngleBetweenVectors(b.unit_vector, [0,0,1]) for b in beams])   
    ang_candidates = np.argwhere(angles < 5.)

    if len(ang_candidates) > 1:
        dist_beams = [b for i,b in enumerate(beams) if i in ang_candidates]
        dists = np.array([geometry.getPoint2VectorDistance3D(b.axis[0], b.axis[1], rafter_center) for b in dist_beams])#beams])
        central_beam = dist_beams[np.argmin(dists)]
    elif len(ang_candidates) == 1:
        central_beam = beams[ang_candidates[0][0]]

    #2 Exclude horizonal beams ~90deg to vertical reference
    symmetry_candidates = np.argwhere(np.abs(angles - 90) > 5.).flatten()
    symmetry_beams = [b for i,b in enumerate(beams) if (b.id != central_beam.id and  i in symmetry_candidates)]

    #3 Check rest of the beams has symmetric memebers
    symmetry_created_obbs = [createSymmetricBeam(central_beam,b) for b in symmetry_beams]
    symmetry_created_beams = [Beam.obb2Beam(obb) for obb in symmetry_created_obbs]

    #4 Check the created symmetric beams intersect to existing ones
    symmetry_beams_idx = [b.id for b in symmetry_beams]
    symmetry_couples = []
    match_len_diff = []
    for i,b1 in enumerate(symmetry_beams):
        for j,b2 in enumerate(symmetry_created_beams):
            if i != j:
                #Check angle match first
                if geometry.getAngleBetweenVectors(b1.unit_vector, b2.unit_vector) <5.:
                    ch1 = b1.getConvexHull3D()
                    ch2 = b2.getConvexHull3D()
                    if ch1.is_intersecting(ch2):
                        #symmetry_couples.append([i, j])
                        diff = abs(max(b1.obb.extent) -max(b2.obb.extent))
                        match_len_diff.append(diff)
                        if True:#diff <0.5:
                            symmetry_couples.append([i, j])

    sym_created_idx = np.array(symmetry_couples).transpose()[1]
    cls_names, cls_counts = np.unique(sym_created_idx, return_counts=True)
    multi_created_idx = cls_names[np.argwhere(cls_counts > 1)]

   
    beams_has_symmetric_match_idx = np.unique(np.array(symmetry_couples).flatten())
    beams_has_no_symmetric_match_idx = [i for i,b in enumerate(symmetry_beams) if i not in beams_has_symmetric_match_idx]
   
    #5 Check if the created symmetric beams that do not match to 
    #existing beams fits on point cloud? generate overlap score
    for id in beams_has_no_symmetric_match_idx:
        candidate_beam = symmetry_created_beams[id]
        #rect = geometry.project3DPointsToPlane2D(candidate_beam.vertices ,rafter_plane)
        #rect_img_cs = []
        #for p in rect:
        #    rect_img_cs.append(imagePrc.cartesian2ImageCoordinates(p, img_ext[0], img_ext[3], img_size))
        #overlap_ratio = imagePrc.getImageRectListOverlapRatio(img,[rect_img_cs])
        overlap_ratio = computeOverlapRatio(candidate_beam,rafter_plane, img, img_size, img_ext)

        if overlap_ratio > 0.7:
            #This means that there is a beam visible in point cloud
            #Re-fitting is neeeded before confirm the beam
            registered_obb = template.getRegisteredOBB(candidate_beam.obb,rafter_pcd,threshold=0.2, max_iter=50)
            reg_beam = Beam.obb2Beam(registered_obb)
            #rect = geometry.project3DPointsToPlane2D(registered_obb.get_box_points() ,rafter_plane)
            #rect_img_cs = []
            #for p in rect:
            #    rect_img_cs.append(imagePrc.cartesian2ImageCoordinates(p, img_ext[0], img_ext[3], img_size))
            #overlap_ratio_after = imagePrc.getImageRectListOverlapRatio(img,[rect_img_cs])
            overlap_ratio_after = computeOverlapRatio(reg_beam, rafter_plane, img, img_size, img_ext)

            #o3d.visualization.draw([rafter_pcd, symmetry_created_obbs[id], registered_obb])
            
            if overlap_ratio_after > 0.7:
                create_beam = Beam.obb2Beam(registered_obb)
                create_beam.comment = "stage3_create_sym"
                #create_beam.cluster_id = m_beams[0].cluster_id
                create_beam.rafter_id = refined['rafter_id']
                create_dct = {'create_score': overlap_ratio_after, 'create_status': "symmetry", 'create_beam': create_beam}
                refined['create_beams'].append(create_dct)
        
            elif force_detection is True:
                create_beam = Beam.obb2Beam(candidate_beam.obb)
                create_beam.comment = "stage3_create_sym"
                #create_beam.cluster_id = m_beams[0].cluster_id
                create_beam.rafter_id = refined['rafter_id']
                create_dct = {'create_score': overlap_ratio_after, 'create_status': "symmetry", 'create_beam': create_beam}
                refined['create_beams'].append(create_dct)

        elif force_detection is True and overlap_ratio >0.5:
                 create_beam = Beam.obb2Beam(candidate_beam.obb)
                 create_beam.comment = "stage3_create_sym"
                 #create_beam.cluster_id = m_beams[0].cluster_id
                 create_beam.rafter_id = refined['rafter_id']
                 create_dct = {'create_score': overlap_ratio, 'create_status': "symmetry", 'create_beam': create_beam}
                 refined['create_beams'].append(create_dct)

    print("")

def createSymmetricBeam(vertical_ref_beam, search_beam):

    #Create vectors from top and bottom of serach beam 
    # to the closest point on vertical ref
    v1_proj = geometry.projectPoint2Vector3D(vertical_ref_beam.axis[0],vertical_ref_beam.axis[1], search_beam.axis[0])
    v2_proj = geometry.projectPoint2Vector3D(vertical_ref_beam.axis[0],vertical_ref_beam.axis[1], search_beam.axis[1])

    d1 = geometry.getDistance(v1_proj, search_beam.axis[0])
    d2 = geometry.getDistance(v2_proj, search_beam.axis[1])

    vec1 = geometry.getUnitVector(v1_proj - search_beam.axis[0])
    vec2 = geometry.getUnitVector(v2_proj - search_beam.axis[1])

    bot_vertices = [v + vec1 * 2 * d1  for v in search_beam.vertices[:4]] #bottom vertices corresponds to axis[0]
    top_vertices = [v + vec2 * 2 * d2  for v in search_beam.vertices[4:]] #top vertices corresponds to axis[1]

    all_vertices = np.vstack((bot_vertices, top_vertices))
    sym_pcd = o3d.geometry.PointCloud()
    sym_pcd.points = o3d.utility.Vector3dVector(all_vertices)
    
    #Translation check
    ref_plane,_,_ = geometry.getPlaneLS(np.vstack((vertical_ref_beam.axis, search_beam.axis)))
    sym_proj = geometry.project3DPointToPlane(sym_pcd.get_center(), ref_plane)
    trans_vec = sym_proj - sym_pcd.get_center()
    sym_pcd.translate(trans_vec)

    sym_obb = sym_pcd.get_minimal_oriented_bounding_box(robust=True)
    sym_obb.color=[0,1,0]
    ext = copy.deepcopy(np.asarray(search_beam.obb.extent))
    ext.sort()
    #sym_obb.extent = [sym_obb.extent[0], search_beam.obb.extent[2], search_beam.obb.extent[1]]
    sym_obb.extent = [ext[2], ext[0], ext[1]]

    sym_pcd2 = o3d.geometry.PointCloud()
    sym_pcd2.points = o3d.utility.Vector3dVector(sym_obb.get_box_points())    
    sym_obb2 = sym_pcd2.get_minimal_oriented_bounding_box(robust=True)
    sym_obb2.color=[0,1,0]
    #o3d.visualization.draw([vertical_ref_beam.obb, search_beam.obb, sym_obb2, sym_pcd])
    return sym_obb2

def computeOverlapRatio(beam, ref_plane, img, img_size, img_ext):
    rect = geometry.project3DPointsToPlane2D(beam.vertices ,ref_plane)
    rect_img_cs = []
    for p in rect:
        rect_img_cs.append(imagePrc.cartesian2ImageCoordinates(p, img_ext[0], img_ext[3], img_size))
    overlap_ratio = imagePrc.getImageRectListOverlapRatio(img,[rect_img_cs])
    return overlap_ratio

def cleanDBStage3(roof_db):
    if roof_db.conn.closed == 1:
        roof_db.connect(True)

    #Delete beam_new table
    sql_str = "delete from beam_new where comment like 'stage3%';"
    roof_db.cursor.execute(sql_str)

    #update beam table
    sql_str2 = "update beam set rafter_id = null, truss_id = null, comment = '' where comment like 'stage3%';"
    roof_db.cursor.execute(sql_str2)

def updateTrussBeams(beams, roof_db):
    for b in beams:
        if b.truss_id is not None:
            sql_str2 = "update beam set comment = 'stage3_truss' where id = " + str(b.id)
            roof_db.cursor.execute(sql_str2)

def setProcessResultsOnDB(refinedRafterDict, roof_db):
    # dict from processBeamGroup function is the input

    if roof_db.conn.closed == 1:
        roof_db.connect(True)

    #Update on beam table
    for rafter_dict in refinedRafterDict:
        rafter_id = rafter_dict['rafter_id']
        for beam in rafter_dict['all_beams']:
            if beam.id in rafter_dict['keep_beams']:
                sql_str = "update beam set rafter_id =" + str(rafter_id) + ", comment = 'stage3_keep' where id = " + str(beam.id)
                if roof_db.conn.closed == 1:
                    roof_db.connect(True)
                roof_db.cursor.execute(sql_str)
                beam.comment = "stage3_keep"            
            elif beam.id in rafter_dict['ignore_beams']:
                sql_str = "update beam set rafter_id =" + str(rafter_id) + ", comment = 'stage3_ignore' where id = " + str(beam.id)
                if roof_db.conn.closed == 1:
                    roof_db.connect(True)
                roof_db.cursor.execute(sql_str)
                beam.comment = "stage3_ignore"
            else:
                sql_str = "update beam set rafter_id =" + str(rafter_id) + ", comment = 'stage3_' where id = " + str(beam.id)
                if roof_db.conn.closed == 1:
                    roof_db.connect(True)
                roof_db.cursor.execute(sql_str)
                beam.comment = "stage3_"

        for merge_dict in rafter_dict['merge_beams']:
            for beam in rafter_dict['all_beams']:
                min_id = np.min(merge_dict['merge_idx'])
                if beam.id in merge_dict['merge_idx']:
                    if merge_dict['merge_status'] is True:
                        sql_str = "update beam set rafter_id =" + str(rafter_id) + ", merge_id = " + str(min_id) + ", comment = 'stage3_merge' where id = " + str(beam.id)
                        if roof_db.conn.closed == 1:
                            roof_db.connect(True)
                        roof_db.cursor.execute(sql_str)
                        beam.comment = "stage3_merge"
                        beam.old_id = min_id
                    else:
                        sql_str = "update beam set rafter_id =" + str(rafter_id) + ", merge_id = " + str(min_id) + ", comment = 'stage3_merge_ignore' where id = " + str(beam.id)
                        if roof_db.conn.closed == 1:
                            roof_db.connect(True)
                        roof_db.cursor.execute(sql_str)
                        beam.comment = "stage3_merge_ignore"
 
        #Insert to beam_new table
        raf_beams = [b for b in rafter_dict['all_beams'] if b.id in rafter_dict['keep_beams']]
        raf_beams.extend([m['beam'] for m in rafter_dict['merge_beams'] if m['merge_status'] is True ])
        raf_beams.extend([c['create_beam'] for c in rafter_dict['create_beams']])

        for beam in raf_beams:
            if beam.id > -1 and beam.old_id is None:
                beam.old_id = beam.id

        roof_db.fillBeamNewTable(raf_beams)

def rayBasedRafterBeamExtension(beams, cover_hull, min_dist = 0.05, max_dist= 5.):
    #Extension of each beam to the convex hull
    #Add the convex hull to the scene
    cover_hull = o3d.t.geometry.TriangleMesh.from_legacy(cover_hull)
    scene = o3d.t.geometry.RaycastingScene()
    hull_id = scene.add_triangles(cover_hull)

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

    #max_dist = 0.2
    for i,b in enumerate(beams):
        if dist_top[i] != float('inf') and dist_top[i] < max_dist and min_dist < dist_top[i]:
            p1 = b.axis[1] + dist_top[i] * b.unit_vector
        else:
            p1= None
        if dist_bot[i] != float('inf') and dist_bot[i] < max_dist and min_dist < dist_bot[i]:
            p2 = b.axis[0] + dist_bot[i] * -b.unit_vector
        else:
            p2 = None
        if not (p1 is None and p2 is None):
            b.extendAlongLongitudinalAxis(p1, p2)
            b.obb.color = [1,1,0]
        elif p1 is not None:
            b.extendAlongLongitudinalAxis(p1)
            b.obb.color = [1,1,0]
        elif p2 is not None:
            b.extendAlongLongitudinalAxis(p1)
            b.obb.color = [1,1,0]
    
    ext_obbs = [b.obb for b in beams] 
    #o3d.visualization.draw([cover_hull, *ext_obbs])

    return beams

def main(config_path):
    start = datetime.now()
    #Read config parameters
    config_data = exchange.readConfig(config_path)

    #Fetch data from beam table
    roof_db = database.modelDatabase(config_path)
    beam_records = roof_db.getBeams(["id", "cluster_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"], "roof_tile_id is null")

    beams_db = [Beam.Beam(id=r['id'], cluster_id = r['cluster_id'], 
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),comment = r['comment'],
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]]) for r in beam_records]

    beam_records2 = roof_db.getNewBeams(["id", "cluster_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"])

    beams_db2 = [Beam.Beam(id=r['id'], cluster_id = r['cluster_id'], 
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),comment = r['comment'],
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]]) for r in beam_records2]



    #Fetch rafter table
    rafter_records = roof_db.getRafters()
    rafters_db =  [Rafter.Rafter(b1_id = r['b1_id'], b2_id = r['b2_id'], joint_id = r['joint_id'],
                                   plane=[float(r['plane_a']), float(r['plane_b']), float(r['plane_c']), float(r['plane_d'])],
                                   id=r['id'], rafter_type = r['rafter_type'], convex_hull = wkb.loads(r['chull2d'], hex=True)) for r in rafter_records]

    #obbs_ignore = [b.setOBB() for b in beams_db if b.comment == "stage2_merge"]
    #o3d.visualization.draw(obbs_ignore)
    
    """
    #Test image procesing code block
    img = cv2.imread("test_image.png")
    img = cv2.bitwise_not(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = imagePrc.getLineSegments(gray)
    colors1 = imagePrc.getRandomColors(8)
    colors2 = imagePrc.getRandomColors(8,True)
    rect_mpts, boxes = imagePrc.linesToRects(gray, lines)
    test_image = cv2.merge((gray,gray,gray))
    for pts in rect_mpts:
        #hull = cv2.convexHull(np.array(pts,dtype='float32'))
        #hull = [np.array(hull).reshape((-1,1,2)).astype(np.int32)]
        #
        #cv2.drawContours(test_image, contours=hull, 
        #                 contourIdx = 0,                     
        #                 color=(255,0,255), thickness=-1)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(test_image, [box],0,(0,255,0),-1)
    cv2.imshow("TestRes", test_image)
    cv2.waitKey()
    #end of test block
    """
    #Set rafter_ids and long. truss candidates
    setRafterIds(beams_db, rafters_db)

    obbs = [b.setOBB() for b in beams_db if b.rafter_id is not None and b.comment != "stage1_outlier"]
    rafter_hulls = [r.convex_hull_3d for r in rafters_db]

    o3d.visualization.draw([*rafter_hulls, *obbs])
    #o3d.visualization.draw_geometries([*rafter_hulls, *obbs])

    obbs_truss = [b.setOBB() for b in beams_db if b.truss_id is not None and b.comment != "stage1_outlier"]

    for b in obbs_truss:
        b.color = [.5,.5,.5]

    o3d.visualization.draw([*rafter_hulls, *obbs_truss])

    # Truss type separation
    in_rafter_clusters = []
    beams_of_rafters = []
    for r in rafters_db:
        clusters = []
        beams = []
        for b in beams_db:
            if b.rafter_id == r.id:
                clusters.append(b.cluster_id)
                beams.append(b)
        in_rafter_clusters.append(np.unique(clusters))
        beams_of_rafters.append(beams)

    for i,r in enumerate(rafters_db):
        if len(in_rafter_clusters[i]) >3:
            #Primary rafter
            r.truss_type = "primary"
        else:
            r.truss_type = "secondary"

    rafters_primary_hull = [r.convex_hull_3d for r in rafters_db if r.truss_type == "primary"]

    o3d.visualization.draw(rafters_primary_hull)

    rafters_secondary_hull = [r.convex_hull_3d for r in rafters_db if r.truss_type == "secondary"]

    o3d.visualization.draw(rafters_secondary_hull)

    #Logic Extend for Central-vertical and bottom-horizontal beams

    for i,r in enumerate(rafters_db):
        if r.truss_type == "primary":
            #obbs = [b.obb for b in beams_of_rafters[i]]
            #o3d.visualization.draw([*obbs, r.convex_hull_3d])
            b1 = Beam.getBeamById(beams_db2, r.b1_id)
            b2 = Beam.getBeamById(beams_db2, r.b2_id)
            
            rafter_hull = Beam.getConvexHullofBeamList([b1,b2])
            rafter_center = rafter_hull.get_center()

            #1 extend central vertical beam
            angles = np.array([geometry.getAngleBetweenVectors(b.unit_vector, [0,0,1]) for b in  beams_of_rafters[i]])   
            ang_candidates = np.argwhere(angles < 5.)          
            if len(ang_candidates) >= 1:
                dist_beams = [b for i,b in enumerate(beams_of_rafters[i]) if i in ang_candidates]
                dists = np.array([geometry.getPoint2VectorDistance3D(b.axis[0], b.axis[1], rafter_center) for b in dist_beams])
                central_beam = dist_beams[np.argmin(dists)]
            if central_beam:
                extended_central_beam = rayBasedRafterBeamExtension([central_beam], rafter_hull)
                #o3d.visualization.draw([central_beam.obb, rafter_hull, r.convex_hull_3d])
            
            #2 extend horizontal bottom beam
            #rafter_hull.scale(1.2, center=rafter_hull.get_center())
            ang_candidates = np.argwhere(np.abs(np.abs(angles) - 90.) < 5.)
            if len(ang_candidates) >= 1:
                dist_beams = [b for i,b in enumerate(beams_of_rafters[i]) if i in ang_candidates]
                dists = np.array([geometry.getPoint2VectorDistance3D(b.axis[0], b.axis[1], central_beam.axis[0]) for b in dist_beams]) # distance to bottom of central beam
                if min(dists) < 0.5:
                    bottom_h_beam = dist_beams[np.argmin(dists)]
                    if bottom_h_beam:
                        extended_central_beam = rayBasedRafterBeamExtension([bottom_h_beam], rafter_hull, 0.05, 8.)
                        #o3d.visualization.draw([bottom_h_beam.obb, rafter_hull, r.convex_hull_3d])

    #Read point cloud
    print("Reading point cloud ...")
    points, normals, segments = exchange.readODM(config_data['db_odm1']) # Read segmented point cloud (before split)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    
    refinedRafterDicts= templateAccumulate(pcd, beams_of_rafters, rafters_db, rafter_type = "primary", check_symmetry=False)
  
    primary_rafter_obbs= []
    colors = imagePrc.getRandomColors(len(refinedRafterDicts), False)
    for i,r in enumerate(refinedRafterDicts):
        raf_beams = [b for b in r['all_beams'] if b.id in r['keep_beams']]
        raf_beams.extend([m['beam'] for m in r['merge_beams'] if m['merge_status'] is True ])
        raf_beams.extend([c['create_beam'] for c in r['create_beams']])
        raf_obbs =  [b.obb for b in raf_beams]
        for b in raf_obbs:
            if b is not None:
                b.color = colors[i]
        primary_rafter_obbs.extend(raf_obbs)
    o3d.visualization.draw([pcd, *primary_rafter_obbs])
        
    refinedRafterDictsSec = templateAccumulate(pcd, beams_of_rafters, rafters_db, rafter_type = "secondary", check_symmetry=False)
    secondary_rafter_obbs = []
    colors = imagePrc.getRandomColors(len(refinedRafterDictsSec), False)
    for i,r in enumerate(refinedRafterDictsSec):
        raf_beams = [b for b in r['all_beams'] if b.id in r['keep_beams']]
        raf_beams.extend([m['beam'] for m in r['merge_beams'] if m['merge_status'] is True ])
        raf_beams.extend([c['create_beam'] for c in r['create_beams']])
        raf_obbs =  [b.obb for b in raf_beams]
        for b in raf_obbs:
            b.color = colors[i]
        secondary_rafter_obbs.extend(raf_obbs)
    o3d.visualization.draw([pcd, *secondary_rafter_obbs])
    o3d.visualization.draw([pcd, *primary_rafter_obbs, *secondary_rafter_obbs])

    #Clean&Prepare DB
    cleanDBStage3(roof_db)
    updateTrussBeams(beams_db, roof_db)
    #Update & Insert Beam & Beam_New
    setProcessResultsOnDB(refinedRafterDicts, roof_db)
    setProcessResultsOnDB(refinedRafterDictsSec, roof_db)
    

    """
    for i,r in enumerate(rafters_db):
        if r.truss_type == "primary":
            obbs = [b.obb for b in beams_of_rafters[i]]
            o3d.visualization.draw([*obbs, r.convex_hull_3d])
            b1 = Beam.getBeamById(beams_db2, r.b1_id)
            b2 = Beam.getBeamById(beams_db2, r.b2_id)
            beams_of_rafters.append(b1)
            beams_of_rafters.append(b2)

            #refineRafterBeams(pcd, b1, b2, r) # TODO       
            r.setBeamObjects(beams_db2)
                   
            #if r.id == 13:
            prc_dict = processRafterBeams(pcd, beams_of_rafters[i], r)
            
            keep_obbs = [b.obb for b in prc_dict["keep_beams"]]
            extend_obbs = [b.obb for b in prc_dict["extend_beams"]]
            merge_obbs = [b.obb for b in prc_dict["merge_beams"]]
            create_obbs = [b.obb for b in prc_dict["created_beams"]]
            rafter_pcd = prc_dict["group_pcd"]
            
            b1_obb = b1.setOBB()
            b1_obb.color = [0.,1.,0]
            b2_obb = b2.setOBB()
            b2_obb.color = [0.,1.,0]
            
            o3d.visualization.draw([rafter_pcd, b1_obb,b2_obb,*keep_obbs, *extend_obbs, *merge_obbs, *create_obbs])
    """

    end = datetime.now()
    print("Rafters defined :\t", (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="In Rafter Beams Processing")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)