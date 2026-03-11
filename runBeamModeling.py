import os
import argparse
import yaml

import numpy as np
from datetime import datetime

import open3d as o3d
from toolBox import geometry, imagePrc, io
from components.components import Segment, Face, Primitive
import components.Beam as Beam

def getSegments(segmentIdx, points, min_seg_size):
    segments = []
    idx = np.unique(segmentIdx)
    for id in idx:
        if id> 0:
            pts = points[np.argwhere(segmentIdx==id).flatten()]
            if len(pts) > min_seg_size:
                segments.append(Segment(id,pts))
    return segments

def matchLinearSegments(linear_segments, colors, config):
    # Sort by height
    seg_heights = np.asarray([s.mbrHeight for s in linear_segments]) * -1.
    linear_segments[:] = [linear_segments[i] for i in np.argsort(seg_heights)] 

    #Create point cloud of CoGs and MBR vertices to spatially limit the comparison
    cog_pts = np.asarray([s.mbrCoG for s in linear_segments])
    mv1_pts =  np.asarray([s.mbrVertices[0] for s in linear_segments])
    mv2_pts =  np.asarray([s.mbrVertices[1] for s in linear_segments])
    mv3_pts =  np.asarray([s.mbrVertices[2] for s in linear_segments])
    mv4_pts =  np.asarray([s.mbrVertices[3] for s in linear_segments])

    # Search for adjacent segments
    processed_idx = []
    matches_pcd = []
    adjacent_segments = []
    for i, s in enumerate(linear_segments):

        ##Test cases!!
        #if s.id in [2381, 1889, 3042, 2563, 2382, 2592, 2743, 2084, 2366]:
        #    wait =1
        #if s.id in [1917,1737,2421,1859,1962,3222,1959,1860,2758,1861]:
        #    wait =2
        #if s.id in [2887,2577,3498,3034,3378]:
        #    wait=3

        if i not in processed_idx:
            seg_pcd = o3d.geometry.PointCloud()
            seg_pcd.points = o3d.utility.Vector3dVector(s.pts3D)
            
            ### Visualizazion step###
            #mbr_pcd = o3d.geometry.PointCloud()
            #mbr_pcd.points = o3d.utility.Vector3dVector(s.mbrVertices)
            #mbr_pcd.paint_uniform_color([1,0,0])
            #o3d.visualization.draw([seg_pcd, mbr_pcd])
            ### ends ###
            
            # Create a searching OBB to detect nearby segments
            offset = config["max_beam_width"]  / 2.
            p1 = s.mbrCoG + offset * s.ev1
            d1 = - np.dot(s.ev1, p1)
            pl1 = np.append(s.ev1, d1) # first side plane
            p2 = s.mbrCoG - offset * s.ev1
            d2 = - np.dot(s.ev1, p2)
            pl2 = np.append(s.ev1, d2) # second side plane
            vertices1 = np.asarray([geometry.project3DPointToPlane(v,pl1) for v in s.mbrVertices])
            vertices2 = np.asarray([geometry.project3DPointToPlane(v,pl2) for v in s.mbrVertices])
            search_pcd = o3d.geometry.PointCloud()
            search_pcd.points = o3d.utility.Vector3dVector(np.vstack((vertices1,vertices2)))
            search_obb = search_pcd.get_oriented_bounding_box()
            search_obb.scale(1.1, search_obb.get_center()) # %10 buffer applied

            #Extend search box - temporary solution for local coordinate system
            search_beam = Beam.obb2Beam(search_obb)
            search_beam.extendAlongLongitudinalAxis(ext_point1=np.asarray((-10000,-10000,-10000)),
                                        ext_point2=np.asarray((10000,10000,10000)), shorten = False)
            search_obb = search_beam.obb
            
            #o3d.visualization.draw([mbr_pcd, search_pcd, search_obb, seg_pcd])            
            # Query the inner segments via CoG + 4 Vertices of MBR coordinates
            #TODO: search_obb extension needs to be optimized
            inner_idx0 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(cog_pts))
            inner_idx1 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv1_pts))
            inner_idx2 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv2_pts))
            inner_idx3 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv3_pts))
            inner_idx4 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv4_pts))           
            inner_idx = np.unique(np.hstack((inner_idx0, inner_idx1, inner_idx2, inner_idx3, inner_idx4)))
            
            len_of_inner_idx = len(inner_idx)
            box_vertices = np.asarray(search_obb.get_box_points())
            if len_of_inner_idx:
                for id in inner_idx:
                   box_vertices = np.vstack((box_vertices, geometry.projectPoint2Vector3D(s.getMBRAxis()[0], s.getMBRAxis()[1], linear_segments[id].getMBRAxis()[0])))
                   box_vertices = np.vstack((box_vertices, geometry.projectPoint2Vector3D(s.getMBRAxis()[0], s.getMBRAxis()[1], linear_segments[id].getMBRAxis()[1])))
                search_pcd = o3d.geometry.PointCloud()
                search_pcd.points = o3d.utility.Vector3dVector(box_vertices)
                search_obb = search_pcd.get_oriented_bounding_box()
                inner_idx0 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(cog_pts))
                inner_idx1 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv1_pts))
                inner_idx2 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv2_pts))
                inner_idx3 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv3_pts))
                inner_idx4 = search_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(mv4_pts))           
                inner_idx = np.unique(np.hstack((inner_idx0, inner_idx1, inner_idx2, inner_idx3, inner_idx4)))
                     
            inner_pcd_list = []
            inner_matches = []
            for id in inner_idx:
                if id not in processed_idx and id != i:
                    inner_seg = linear_segments[id]                
                    #Check longitudinal parallellity
                    angle = geometry.getAngleBetweenVectors(s.ev3, inner_seg.ev3)
                    if abs(angle) < config["max_long_angle"] or abs(180 - angle) <config["max_long_angle"]: #3
                        #Check normal vectors are parallel
                        angle2 = geometry.getAngleBetweenVectors(s.ev1, inner_seg.ev1)
                        if abs(angle2) < config["max_norm_angle"] or abs(180 - angle2) < config["max_norm_angle"]: #15
                            #Assume parallel segments: Same or opposite faces
                            #Check if relevant/shifted
                            th= geometry.getDistance(geometry.projectPoint2Vector3D(s.mbrCoG, s.mbrCoG+s.ev2, inner_seg.mbrCoG), s.mbrCoG)
                            if th < s.mbrWidth / 2.:# 4 before
                                #not shifted case
                                if abs(geometry.getPoint2PlaneDistance(inner_seg.mbrCoG, s.plane)) > config["min_beam_width"]:
                                    inner_matches.append({"segment": inner_seg, "side": 4, "is_valid": 1, "id": id}) #Opposite side
                                    processed_idx.append(id)
                                else:
                                    inner_matches.append({"segment": inner_seg, "side": 1, "is_valid": 1, "id": id}) #Same side
                                    processed_idx.append(id)
                        else:
                            #Not Parallel case
                            if abs(geometry.getPoint2PlaneDistance(inner_seg.mbrCoG, s.plane)) < config["max_beam_width"] /2.:
                                dist_of_vertices = [abs(geometry.getPoint2PlaneDistance(mbr_vert, s.plane)) for mbr_vert in inner_seg.mbrVertices]
                                if max(dist_of_vertices) < config["max_beam_width"]: # min dist seems ideal but gives less flexibility!
                                #if min(dist_of_vertices) < config["min_beam_width"]/4:            
                                    #This is connectivity checking for side segments                                  
                                    #Can be orthogonal (refers to cuboids) or have a different angle (possibly trapezoid)
                                    inner_matches.append({"segment": inner_seg, "side": 0, "is_valid": 0, "id": id}) # At this point, unknown side, id is index of list!

            if len (inner_matches):
                detected_faces = np.asarray([s["side"] for s in inner_matches])
                uq, cnt = np.unique(detected_faces, return_counts=True)
                if 0 in uq and 4 not in uq:
                    # Have no oppposite candidate, and have 1+ side candidates!
                    # Check if side candidates are valid together
                    # All segments in inner_matches -> side = 0
                    signed_dists = []
                    seg_by_signs = []
                    for cnd_seg in inner_matches:
                        if cnd_seg["side"] == 0:
                            seg_by_signs.append(cnd_seg)
                            signed_dists.append(geometry.getSignedDistance(s.mbrCoG, s.mbrCoG + s.ev1, cnd_seg["segment"].mbrCoG))
                    signs = np.sign(np.asarray(signed_dists))              
                    pcd_list_ = []

                    if len(np.unique(signs)) > 1:
                        len_side1 = (signs==1).sum()
                        len_side2 = (signs==-1).sum()
                        if  len_side1 != len_side2:
                            if len_side1 > len_side2:
                                #sign -> positive
                                side = 1
                            else:
                                #sign -> negative
                                side = -1
                            for ii,cnd_seg in enumerate(seg_by_signs):
                                if cnd_seg["side"] == 0 and signs[ii] == side:
                                    cnd_seg["is_valid"] = 1
                                    processed_idx.append(cnd_seg["id"])

                        else:
                            #Equal possibility case
                            str_msg = "Base segment: " + str(s.id) + ", has conflict with side segments: "
                            colorid = 0
                            
                            face1 = Face.fromSegments([inner_matches[0]["segment"]])
                            face2 = Face(points = inner_matches[0]["segment"].pts3D)
                            
                            for cnd_seg in inner_matches:
                                colorid += 1
                                str_msg += str(cnd_seg["segment"].id) + " ,"
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(cnd_seg["segment"].pts3D)
                                pcd.paint_uniform_color(colors[colorid])
                                pcd_list_.append(pcd)
                            print(str_msg)
                            box = seg_pcd.get_oriented_bounding_box()
                            #o3d.visualization.draw([seg_pcd, box, *pcd_list_])
                      
                    else:
                        # Match&
                        for cnd_seg in inner_matches:
                            if cnd_seg["side"] == 0:
                                cnd_seg["is_valid"] = 1
                                processed_idx.append(cnd_seg["id"])

                elif 0 in uq and 4 in uq:
                    #Have opposite candidate, which helps to eleminate wrong side match
                    oppo_segments = [cnd_seg["segment"] for cnd_seg in inner_matches if cnd_seg["side"] == 4]
                    ref_signed_dist = geometry.getSignedDistance(s.mbrCoG, s.mbrCoG + s.ev1, oppo_segments[0].mbrCoG)

                    pcd_oppo = o3d.geometry.PointCloud()
                    pcd_oppo.points = o3d.utility.Vector3dVector(oppo_segments[0].pts3D)

                    for cnd_seg in inner_matches:
                        if cnd_seg["side"] == 0:
                            seg_signed_dist = geometry.getSignedDistance(s.mbrCoG, s.mbrCoG + s.ev1, cnd_seg["segment"].mbrCoG)
                            if ref_signed_dist * seg_signed_dist > 0:
                                # Valid side segment case
                                cnd_seg["is_valid"] = 1
                                processed_idx.append(cnd_seg["id"])
                            else:
                                #Invalid side segment case
                                print("Side segment: {0} is not valid for the Base segment: {1}".format(cnd_seg["segment"].id, s.id))

                matching_segments = [seg for seg in inner_matches if seg["is_valid"] == 1]
                matching_results = []
                if len(matching_segments):
                    processed_idx.append(i)
                    matching_results.append({"segment": s, "side": 1, "is_valid": 1, "id": i})
                    matching_results.extend(matching_segments)
                    adjacent_segments.append(matching_results)
                    for seg in  matching_segments:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(seg["segment"].pts3D)
                        seg_pcd+= pcd
                    seg_pcd.paint_uniform_color(colors[i])
                    matches_pcd.append(seg_pcd)


    not_processed_pcd = []
    for i,seg in enumerate(linear_segments):
        if i not in processed_idx:
            np_pcd = o3d.geometry.PointCloud()
            np_pcd.points = o3d.utility.Vector3dVector(seg.pts3D)
            not_processed_pcd.append(np_pcd)
    return matches_pcd, not_processed_pcd, adjacent_segments

def adjacentSegments2Beams(adjacent_segments):
    #Adjacent segments to oriented faces: Face1->Front, Face2->Left, Face4->Right
    #Define Left-Right for side=0 for the current order
    beam_faces_list = []
    beam_mesh_list = []
    valid_idx = []
    for valid_id, adj_list in enumerate(adjacent_segments):
        front_side = adj_list[0]["segment"]
        for s in adj_list:            
            if s["side"] == 0:
                signed_d = geometry.getSignedDistance(front_side.mbrCoG, front_side.mbrCoG+ front_side.ev2, s["segment"].mbrCoG)
                if signed_d < 0:
                    s["side"] = 2
                else:
                    s["side"] = 3
        sides = [s["side"] for s in adj_list]
        uq_sides = np.unique(sides)

        beam_faces = []
        if len(uq_sides) == 1:
            # Single - Side
            continue
        elif len(uq_sides) == 2:
            # Mostly cuboids, 
            # If the faces are not orthogonal, 2 possible Trapezoids exists         
            for side in [1,2,3,4]:
                if side in uq_sides:
                    segments_in_face = [se["segment"] for se in adj_list if se["side"] == side]
                    beam_faces.append({"face": Face.fromSegments(segments_in_face), "side": side})


            angle = geometry.getAngleBetweenVectors(beam_faces[0]["face"].ev1, beam_faces[1]["face"].ev1)
            if abs(angle) < 5. or abs(180 - angle) <5.:
                #Orthogonals -> Only cuboid
                beam_faces_list.append({"face_list":beam_faces, "primitives":["cuboid"]})
            else:
                #2 possible trapezoids exists
                beam_faces_list.append({"face_list":beam_faces, "primitives":["trapezoid1", "trapezoid2"]})

            front_face, left_face, right_face, back_face = [None, None, None, None]
            for f in beam_faces:
                if f["side"] == 1:
                    front_face = f["face"]
                elif f["side"] == 2:
                    left_face = f["face"]
                elif f["side"] == 3:
                    right_face = f["face"]
                elif f["side"] == 4:
                    back_face = f["face"]
            if back_face:
                beam_primitive = Primitive([front_face, left_face, right_face, back_face], True)
                #beam_primitive.fit_trapezoidal_prism()
                beam_primitive.fit_cuboid()
                beam_mesh = beam_primitive.get_trapezoidal_prsim_mesh()
                beam_mesh_list.append(beam_mesh)
                valid_idx.append(valid_id)
                front_face, left_face, right_face, back_face = None, None, None, None
                #pcd1 = o3d.geometry.PointCloud()
                #pcd1.points = o3d.utility.Vector3dVector(front_face.segments[0].pts3D)
                #pcd1.paint_uniform_color([1,0,0])
                #pcd4 = o3d.geometry.PointCloud()
                #pcd4.points = o3d.utility.Vector3dVector(back_face.segments[0].pts3D)
                #pcd4.paint_uniform_color([1,1,0])             
                #o3d.visualization.draw([pcd1, pcd4,trapezoidal_mesh])

            elif back_face is None and left_face is None: #TODO handle later
                ##beam_primitive = Primitive([front_face, left_face, right_face, back_face], True)
                beam_primitive = Primitive([right_face, front_face, left_face, back_face], True)
                #trapezoidal_prism.fit_trapezoidal_prism()
                beam_primitive.fit_cuboid()
                beam_mesh = beam_primitive.get_trapezoidal_prsim_mesh()
                beam_mesh_list.append(beam_mesh)
                valid_idx.append(valid_id)
                front_face, left_face, right_face, back_face = None, None, None, None
            elif back_face is None and right_face is None:
                beam_primitive = Primitive([front_face, left_face, right_face, back_face], True)
                beam_primitive.fit_cuboid()
                #trapezoidal_prism.fit_trapezoidal_prism()
                beam_mesh = beam_primitive.get_trapezoidal_prsim_mesh()
                beam_mesh_list.append(beam_mesh)
                valid_idx.append(valid_id)
                front_face, left_face, right_face, back_face = None, None, None, None

        elif len(uq_sides) == 4:
            # If faces are not orthogonal, trapezoid case needs to be considered
            # Face order may change in case of trap.
            for side in [1,2,3,4]:               
                segments_in_face = [se["segment"] for se in adj_list if se["side"] == side]
                beam_faces.append({"face": Face.fromSegments(segments_in_face), "side": side})

            # Front -> Back comparison
            angle1 = geometry.getAngleBetweenVectors(beam_faces[0]["face"].ev1, beam_faces[3]["face"].ev1)
            # Left -> Right comparison
            angle2 = geometry.getAngleBetweenVectors(beam_faces[1]["face"].ev1, beam_faces[2]["face"].ev1)
            # Front -> Left comparison
            angle3 = geometry.getAngleBetweenVectors(beam_faces[0]["face"].ev1, beam_faces[1]["face"].ev1)
            # Front -> Right comparison
            angle4 = geometry.getAngleBetweenVectors(beam_faces[0]["face"].ev1, beam_faces[2]["face"].ev1)
            # Back -> Left comparison
            angle5 = geometry.getAngleBetweenVectors(beam_faces[3]["face"].ev1, beam_faces[1]["face"].ev1)
            # Back -> Right comparison
            angle6 = geometry.getAngleBetweenVectors(beam_faces[3]["face"].ev1, beam_faces[2]["face"].ev1)


            if abs(angle1) < 5. or abs(180 - angle1) <5.:
                #Front - Back parallel
                if abs(angle2) < 5. or abs(180 - angle2) <5.:
                    # Left - Right parallel
                    beam_faces_list.append({"face_list":beam_faces, "primitives":["cuboid"]})
                else:
                    # Left - Right forms Trapezoid ?? TODO: Support by handled angles
                    beam_faces_list.append({"face_list":beam_faces, "primitives":["cuboid","trapezoid1"]}) ## Alterntively cuboid test
            
            elif abs(angle2) < 5. or abs(180 - angle2) <5.:
                ## Need to convert faces
                beam_faces[1]["side"] = 1
                beam_faces[2]["side"] = 4
                front_side = beam_faces[1]["face"].segments[0]#["segments"][0]
                old_front_side = beam_faces[0]["face"].segments[0]#["segments"][0]
                signed_d = geometry.getSignedDistance(front_side.mbrCoG, front_side.mbrCoG+ front_side.ev2, old_front_side.mbrCoG)
                if signed_d < 0:
                    s["side"] = 2
                    beam_faces[0]["side"] = 2
                    beam_faces[3]["side"] = 3
                else:
                    beam_faces[0]["side"] = 3
                    beam_faces[3]["side"] = 2
                beam_faces_list.append({"face_list":beam_faces, "primitives":["cuboid","trapezoid1"]}) ## Alterntively cuboid test

            for f in beam_faces:
                if f["side"] == 1:
                    front_face = f["face"]
                elif f["side"] == 2:
                    left_face = f["face"]
                elif f["side"] == 3:
                    right_face = f["face"]
                elif f["side"] == 4:
                    back_face = f["face"]
            beam_primitive = Primitive([front_face, left_face, right_face, back_face], True)
            #trapezoidal_prism.fit_trapezoidal_prism()
            beam_primitive.fit_cuboid()
            beam_mesh = beam_primitive.get_trapezoidal_prsim_mesh()
            beam_mesh_list.append(beam_mesh)
            valid_idx.append(valid_id)
            front_face, left_face, right_face, back_face = None, None, None, None

        elif len(uq_sides) == 3:
            # If faces are not orthogonal, trapezoid case needs to be considered
            # Face order may change in case of trap.
            for side in [1,2,3,4]:
                if side in uq_sides:
                    segments_in_face = [se["segment"] for se in adj_list if se["side"] == side]
                    beam_faces.append({"face": Face.fromSegments(segments_in_face), "side": side})

            if 4 in uq_sides:
                #Face order need to be changed
                if 2 in uq_sides:
                   for b in beam_faces:
                       if b["side"] == 1:
                           b["side"] = 3
                       elif b["side"] == 2:
                           b["side"] = 1
                       elif b["side"] == 4:
                          b["side"] = 2
                elif 3 in uq_sides:
                       for b in beam_faces:
                           if b["side"] == 1:
                               b["side"] = 2
                           elif b["side"] == 3:
                               b["side"] = 1
                           elif b["side"] == 4:
                              b["side"] = 3

            for f in beam_faces:
                if f["side"] == 1:
                    front_face = f["face"]
                elif f["side"] == 2:
                    left_face = f["face"]
                elif f["side"] == 3:
                    right_face = f["face"]

            # Front -> Left comparison
            angle1 = geometry.getAngleBetweenVectors(front_face.ev1, left_face.ev1)
            # Front -> Right comparison
            angle2 = geometry.getAngleBetweenVectors(front_face.ev1, right_face.ev1)
            # Left -> Right comparison
            angle3 = geometry.getAngleBetweenVectors(left_face.ev1, right_face.ev1)

            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(front_face.segments[0].pts3D)
            pcd1.paint_uniform_color([1,0,0])

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(left_face.segments[0].pts3D)
            pcd2.paint_uniform_color([0,1,0])

            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(right_face.segments[0].pts3D)
            pcd3.paint_uniform_color([0,0,1])

            beam_primitive = Primitive([front_face, left_face, right_face, None], True)
            #trapezoidal_prism.fit_trapezoidal_prism()
            beam_primitive.fit_cuboid()
            beam_mesh = beam_primitive.get_trapezoidal_prsim_mesh()
            beam_mesh_list.append(beam_mesh)
            valid_idx.append(valid_id)
            front_face, left_face, right_face, back_face = None, None, None, None
    return beam_mesh_list, valid_idx

def main():
    start_total = datetime.now()

    # --- Load Config params ---
    parser = argparse.ArgumentParser(description="Beam Modeling")
    parser.add_argument('cfg', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()
    config_data = yaml.safe_load(args.cfg)
    
    #Internal parameters:
    #input
    input_cloud_path = args.cfg.name.replace("config.yml", config_data['beamModeling']['point_cloud'])
    min_seg_size = config_data['beamModeling']['min_seg_size']
    min_beam_width = config_data['beamModeling']['min_beam_width']
    max_beam_width = config_data['beamModeling']['max_beam_width']
    max_long_angle = config_data['beamModeling']['max_long_angle']
    max_norm_angle = config_data['beamModeling']['max_norm_angle']
    
    #outputs
    beams_mesh_path = args.cfg.name.replace("config.yml", config_data['beamModeling']['beams_mesh'])
    beams_pcd_path = args.cfg.name.replace("config.yml", config_data['beamModeling']['beams_pcd'])

    #Config params
    config = {
    "min_seg_size": min_seg_size,
    "min_beam_width": min_beam_width,
    "max_beam_width": max_beam_width,
    "max_norm_angle": max_norm_angle,
     "max_long_angle": max_long_angle
    }

    # --- 1. Load Data ---
    print("Loading point cloud...")
    try:
        pcd_data = io.readPLY(input_cloud_path)
        print(f"Total points: {len(pcd_data['points'])}")
    except:
        print("Error: Could not find %s"%segmented_pcd_path)
        return
    
    points = pcd_data['points']
    normals = pcd_data['normals']
    segments = pcd_data['segmentid']

    # --- 2. Segment handling ---
    print("Handling segments...")
    segment_handling_starts = datetime.now()
    roof_segments = getSegments(segments,points,min_seg_size)
    print(f"{len(roof_segments)} segments handled: {str(datetime.now() - segment_handling_starts)[:-4]}")

    # --- 3. Adjacent segments searching ---
    print("Adjacent segments searching...")
    adj_seg_search_starts = datetime.now()
    max_seg_id = np.max(np.array([seg.id for seg in roof_segments]))
    #colors = imagePrc.getRandomColors(int(max_seg_id + 1))
    colors = np.random.uniform(0.1, 1.0, (max_seg_id + 1, 3))
    colors[-1] = [0, 0, 0]  # Set noise to Black 

    linear_segments = [s for s in roof_segments if s.type == "a" and s.mbrWidth < max_beam_width and s.mbrWidth > min_beam_width]
    matches_pcd, not_processed_pcd, matching_results = matchLinearSegments(linear_segments, colors, config)

    print(f"Adjacent segments estimation took: {str(datetime.now() - adj_seg_search_starts)[:-4]}")

    # --- 4. Best fitting beam modeling ---
    print("Fitting beam geometries...")
    beam_fitting_starts = datetime.now()
    beam_meshes, valid_idx = adjacentSegments2Beams(matching_results)
    print(f"Geometry fitting took: {str(datetime.now() - beam_fitting_starts)[:-4]}")

    #beams_pcd = o3d.geometry.PointCloud()
    #for po in matches_pcd:
    #    beams_pcd += po
    #o3d.io.write_point_cloud("beams_pcd.ply", beams_pcd)

    print("Exporting results...")
    export_starts = datetime.now()
    valid_pcds = [pcd for i, pcd in enumerate(matches_pcd) if i in valid_idx]

    #Prepare only valid beams & points to be exported
    result_beams = []
    result_pcds =[]
    for i,beam_mesh in enumerate(beam_meshes):
        vertices = np.asarray(beam_mesh.vertices)
        box_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        box = box_pts.get_minimal_oriented_bounding_box(robust=False) 
        sorted_ext = np.sort(box.extent)
        if sorted_ext[0] > min_beam_width and sorted_ext[1] < max_beam_width:
            result_beams.append(beam_mesh)
            result_pcds.append(valid_pcds[i])

    result_mesh = o3d.geometry.TriangleMesh()
    for t in result_beams:
        result_mesh += t
    o3d.io.write_triangle_mesh(beams_mesh_path, result_mesh)

    #points_tuple = (np.asarray(pcd.points) for pcd in result_pcds)
    #points_merged = np.vstack(points_tuple)
    #colors_tuple = (np.asarray(pcd.colors) for pcd in result_pcds)

    result_labels = []
    result_pcd = o3d.geometry.PointCloud()
    for i,p in enumerate(result_pcds):
        result_pcd += p
        result_labels.append((np.full(len(np.asarray(p.points)), i)))
    res_lbl = np.hstack(result_labels)

    result_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    res_pts = np.asarray(result_pcd.points)
    res_nrm = np.asarray(result_pcd.normals)
    res_col = np.asarray(result_pcd.colors)

    res_col = np.asarray(result_pcd.colors) * 255
    res_col = res_col.astype(np.int32)
    
    io.writePLY(beams_pcd_path, res_pts, res_nrm, res_col, res_lbl)

    print(f"Beam export took: {str(datetime.now() - export_starts)[:-4]}")

    nr_beams = len(result_beams)
    print("-" * 40)
    print(f"Beams modeled: {nr_beams}")
    print(f"Total Process Time: {str(datetime.now() - start_total)[:-4]}")
    print("-" * 40)

if __name__ == "__main__":
    main()