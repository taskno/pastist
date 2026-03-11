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

import roof.RoofTile as RoofTile
import roof.Beam as Beam
import roof.Joint as Joint
import roof.Rafter as Rafter
import roof.BeamGroup as BeamGroup

import open3d as o3d
import ezdxf
import alphashape
from shapely import wkb
from shapely.geometry import Point
from sklearn.cluster import KMeans

def getJoints(beams, tolerance):  
    joints = []
    processed_beams = []
    for i,b1 in enumerate(beams):
        for j,b2 in enumerate(beams):
            if b1.roof_tile_id == b2.roof_tile_id or b2.id in processed_beams:
                continue
            else:
                if b1.roof_tile_id < b2.roof_tile_id:
                    jo = Joint.detectJoint(b1,b2, tolerance)
                    if jo:
                        jo.b1_group = b1.roof_tile_id
                        jo.b2_group = b2.roof_tile_id
                        joints.append(jo)
                else:
                    jo = Joint.detectJoint(b2,b1, tolerance)
                    if jo:
                        jo.b1_group = b2.roof_tile_id
                        jo.b2_group = b1.roof_tile_id
                        joints.append(jo)
        processed_beams.append(b1.id)
    return joints

def extendRafterBeams(beams, missing_joints, roof_tiles, tolerance):
    # Extend operation considered upwards direction (only top position changes)
    extended_beams = []
    extra_joints = []
    for jo in missing_joints:
        b1,b2 = None, None 
        for b in beams:
            if b1 is None and b.id == jo.beam1_id:
                b1 = b
            elif b2 is None and b.id == jo.beam2_id:
                b2 = b
            elif b1 is not None and b2 is not None:
                break

        r1,r2 = None, None
        for r in roof_tiles:
            if r.id == jo.b1_group:
                r1 = r
            if r.id == jo.b2_group:
                r2 = r

        #Compare b1 -> roof tile of b2 and vice versa
        r2_planes = geometry.getParallelPlanes(r2.plane, b2.width / 8)
        r2_d = (abs(geometry.getPoint2PlaneDistance(b1.axis[1], r2_planes[0])), abs(geometry.getPoint2PlaneDistance(b1.axis[1], r2_planes[1])))
        r2_plane = r2_planes[np.argmin(r2_d)] # choose closest plane !!!! try farthest
        b1_ext_point = b1.axis[1] + b1.unit_vector * min(r2_d)
        b1.extendAlongLongitudinalAxis(b1.axis[0], b1_ext_point, True)
        

        r1_planes = geometry.getParallelPlanes(r1.plane, b1.width / 8)
        r1_d = (abs(geometry.getPoint2PlaneDistance(b2.axis[1], r1_planes[0])), abs(geometry.getPoint2PlaneDistance(b2.axis[1], r1_planes[1])))
        r1_plane = r1_planes[np.argmin(r1_d)] # choose closest plane
        b2_ext_point = b2.axis[1] + b2.unit_vector * min(r1_d)
        b2.extendAlongLongitudinalAxis(b2.axis[0], b2_ext_point, True)
       
        if geometry.getDistance(b1.axis[1], b2.axis[1]) < tolerance:
            extended_beams.append(b1)
            extended_beams.append(b2)
            extra_joints.append(getJoints((b1,b2), tolerance)[0])

        #obbs = [b.obb for b in beams]
        #o3d.visualization.draw(obbs)
        
    return {"extended_beams":extended_beams, "extra_joints":extra_joints}

def insertJointNews(roof_db, joints, rafter_matches):
    # This function is only for creating rafter connector joints
    roof_db.connect(True)
    #Clean the table first!
    clean_str = "delete from joint_new where joint_type like 'rafter_top_" + str(rafter_matches[0]) + "_" +str(rafter_matches[1]) + "%';"
    roof_db.cursor.execute(clean_str)

    for j in joints:
        p_b1 = "'POINT Z (" + str(j.p_b1[0]) + " " + str(j.p_b1[1]) + " " + str(j.p_b1[2]) + ")'::geometry"
        p_b2 = "'POINT Z (" + str(j.p_b2[0]) + " " + str(j.p_b2[1]) + " " + str(j.p_b2[2]) + ")'::geometry"

        if j.b1_group == rafter_matches[0] and j.b2_group == rafter_matches[1]:
            joint_type = "'rafter_top_" + str(j.b1_group) + "_" + str(j.b2_group) + "'"
        else:
            joint_type = "'not_defined'"
            #Unexpected case!

        values = (j.beam1_id, j.beam2_id, j.b1_position, j.b2_position, p_b1, p_b2, joint_type )
        insert_sql = "insert into joint_new (b1_id,b2_id,b1_position,b2_position,p_b1,p_b2, joint_type) values (%s, %s, %s, %s, %s, %s, %s)" % values
        roof_db.cursor.execute(insert_sql)   
    roof_db.closeSession()

def insertRafters(roof_db, rafters, rafter_matches):
    roof_db.connect(True)
    #Clean the table first!
    clean_str = "delete from rafter where rafter_type like 'rafter_top_" + str(rafter_matches[0]) + "_" +str(rafter_matches[1]) + "%';"
    roof_db.cursor.execute(clean_str)

    for r in rafters:
        rafter_hull = "ST_GeomFromText('" + r.convex_hull.wkt + "',0)"
        values = (r.b1_id, r.b2_id, r.plane[0], r.plane[1], r.plane[2], r.plane[3], "'"+ r.rafter_type + "'", rafter_hull)

        insert_sql = "insert into rafter (b1_id, b2_id, plane_a, plane_b, plane_c, plane_d, rafter_type, chull2d) values (%s, %s, %s, %s, %s, %s, %s, %s)" % values
        roof_db.cursor.execute(insert_sql)
    
    #update joint_id s
    update_str =  "update rafter set joint_id = joint_new.id from joint_new where joint_new.b1_id = rafter.b1_id and joint_new.b2_id = rafter.b2_id"
    roof_db.cursor.execute(update_str)

    beam_new_update1 = "update beam_new set rafter_id = rafter.id from rafter where rafter.b1_id = beam_new.id;"
    roof_db.cursor.execute(beam_new_update1)

    beam_new_update2 = "update beam_new set rafter_id = rafter.id from rafter where rafter.b2_id = beam_new.id;"
    roof_db.cursor.execute(beam_new_update2)
    roof_db.closeSession()

def updateExtendedBeams(roof_db, beams):
    roof_db.connect(True)

    for beam in beams:
        values = (             
                    beam.length,
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

        update_str = "update beam_new set length = %s, p0 = %s, p1 = %s, p2 = %s, p3 = %s, p4 = %s, p5 = %s, p6 = %s, p7 = %s, axis_start = %s, axis_end = %s " % values
        update_str += " where id = " + str(beam.id)
        roof_db.cursor.execute(update_str)
    roof_db.closeSession()

def rayBasedRafterBeamExtension(beams, cover_hull, roof_db):
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

    for i,b in enumerate(beams):
        if  dist_top[i] != float('inf'):
            p1 = b.axis[1] + dist_top[i] * b.unit_vector
        else:
            p1= None
        if dist_bot[i] != float('inf'):
            p2 = b.axis[0] + dist_bot[i] * -b.unit_vector
        else:
            p2 = None
        if not (p1 is None and p2 is None):
            b.extendAlongLongitudinalAxis(p1, p2)
    
    ext_obbs = [b.obb for b in beams] 
    o3d.visualization.draw([cover_hull, *ext_obbs])

    print("Extended beams will be uploaded on db ? (y/n):")
    answer = str(input())
    if answer == "y":
        updateExtendedBeams(roof_db, beams)
        print("Beams updated.\n")       
    else:
        print("No beam update based on Ray-Casting.\n")
    return beams


def distanceOptimization(beams, roof_tiles, cover_hull, tolerance = 0.2):
    hull_center = cover_hull.get_center()
    cover_hull = o3d.t.geometry.TriangleMesh.from_legacy(cover_hull)
    scene = o3d.t.geometry.RaycastingScene()
    hull_id = scene.add_triangles(cover_hull)

    #Beams to roof tile distance standardization function
    refined_beams = []
    for tile in roof_tiles:
        tile_beams= [b for b in beams if b.roof_tile_id == tile.id]

        #angles = [geometry.getAngleBetweenVectors(b.unit_vector, tile.plane[:3]) for b in tile_beams]
        dists = np.array([geometry.getPoint2PlaneDistance(b.axis[0], tile.plane) for b in tile_beams])

        dists = []
        for b in tile_beams:
            d2 = - np.dot(tile.plane[:3], np.mean((b.axis),axis=0))
            plane2 = np.append(tile.plane[:3] , d2)
            vertices2 = [geometry.project3DPointToPlane(p, plane2) for p in b.vertices[:4]]
            
            dists.append(np.min( np.array([geometry.getPoint2PlaneDistance(v, tile.plane) for v in vertices2])))
        dists = np.array(dists)
        out_idx = np.argwhere(np.abs(dists) > tolerance)

 
        if len(out_idx) > 0:
            # There are movable beams
            for id in out_idx:
                b = tile_beams[id[0]]

                #define a point closed to the convex_hull (instead of using roof tile plane)
                #use roof tile normal vector as moving direction

                #orient the normal vector from center to outside
                n = np.array(tile.plane[:3])
                n_o = geometry.orientNormalVector(hull_center, n, np.mean((b.axis), axis =0))
             
                ray_array = [np.hstack((v, n_o)) for v in vertices2]# b.vertices]
                rays = o3d.core.Tensor(ray_array, dtype=o3d.core.Dtype.Float32)
                ans = scene.cast_rays(rays)
                dist_arr = ans['t_hit'].numpy()

                new_vertices = np.array([v + n_o * min(dist_arr) for v in b.vertices])
                new_axis = np.array([v + n_o * min(dist_arr) for v in b.axis])

                new_beam = copy.deepcopy(b)
                new_beam.vertices = new_vertices
                new_beam.axis = new_axis
                new_beam.setOBB()
                new_beam.obb.color = [0,1,0]
                new_beam.comment += "_stage22_distbalance"

                refined_beams.append(new_beam)
                b.obb.color= [1,0,0]
               
        #obbs = [b.obb for b in tile_beams]
        #o3d.visualization.draw(obbs)

    if len(refined_beams):
        refined_obbs = [b.obb for b in refined_beams]
        refined_idx = [b.id for b in refined_beams]
        obbs = [b.obb for b in beams]
        o3d.visualization.draw([*obbs, *refined_obbs])

        print("Red beams are going to be replaced with green ones (y/n):")
        answer = str(input())
        if answer == "y":
            print("%s Beams replaced.\n" %len(refined_beams))
            
            beams_u = []
            for b in beams:
                if b.id in refined_idx:
                    #b = refined_beams[np.argwhere(np.array(refined_idx) == b.id)[0][0]]
                    beams_u.append(refined_beams[np.argwhere(np.array(refined_idx) == b.id)[0][0]])
                    #b.setOBB()
                else:
                    beams_u.append(b)
            beams = beams_u
        else:
            print("No distance balance refinement applied.\n")
            
    #obs = [b.obb for b in beams]
    #o3d.visualization.draw(obs)
    return beams




def setOBBs(beams):
    obbs = [b.setOBB() for b in beams]
    #o3d.visualization.draw(obbs)

def main(config_path):
    start = datetime.now()
    #Read config parameters
    config_data = exchange.readConfig(config_path)

    #Update cluster ids on beam_new if not yet assigned
    #roof_db.updataNewBeamClusters()

    #Fetch data from beam_new table and roof_tile table
    roof_db = database.modelDatabase(config_path)
    beam_records = roof_db.getNewBeams(["id", "roof_tile_id", "axis_start", "axis_end", 
                                     "nx", "ny", "nz", 
                                     "width", "height", "length", 
                                     "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "comment"], "roof_tile_id is not null")

    beams_new_db = [Beam.Beam(id=r['id'], roof_tile_id = r['roof_tile_id'], 
                       axis=[wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0]], 
                       unit_vector=[float(r['nx']), float(r['ny']), float(r['nz'])],
                       width=float(r['width']), height=float(r['height']), length=float(r['length']),
                       vertices = [wkb.loads(r['p0'], hex=True).coords[:][0], wkb.loads(r['p1'], hex=True).coords[:][0],
                                   wkb.loads(r['p2'], hex=True).coords[:][0], wkb.loads(r['p3'], hex=True).coords[:][0],
                                   wkb.loads(r['p4'], hex=True).coords[:][0], wkb.loads(r['p5'], hex=True).coords[:][0],
                                   wkb.loads(r['p6'], hex=True).coords[:][0], wkb.loads(r['p7'], hex=True).coords[:][0]], comment = str(r['comment'])) for r in beam_records]

  
    #Search for missing rafter connections
    roof_tile_records = roof_db.getRoofTiles()
    roof_tiles_db =  [RoofTile.RoofTile(id=r['id'], plane=[float(r['plane_a']), float(r['plane_b']), float(r['plane_c']), float(r['plane_d'])], alpha_shape2d = None) for r in roof_tile_records]

    setOBBs(beams_new_db)
  
    cover_hull= Beam.getConvexHullofBeamList(beams_new_db, False)
    #cover_hull.scale(1.01, center=cover_hull.get_center())

    # 1- Plane - Beam distance refinement
    beams_new_db = distanceOptimization(beams_new_db, roof_tiles_db, cover_hull, tolerance = 0.2)

    # 2- Ray based beam extend
    #cover_hull= Beam.getConvexHullofBeamList(beams_new_db, False)
    beams_new_db = rayBasedRafterBeamExtension(beams_new_db, cover_hull, roof_db)

    obbs = [b.obb for b in beams_new_db]
    o3d.visualization.draw(obbs)

    # 3--Searching for missing top-joint member--
    #config_data['max_joint_th'] # max connector distance
    joints = getJoints(beams_new_db, config_data['max_joint_th'])
    joint_repetition = np.array([np.sort((j.b1_group, j.b2_group)) for j in joints])
    most_repetative_joints = np.unique(joint_repetition, axis = 0, return_counts=True)
    rafter_matches = most_repetative_joints[0][np.argwhere(most_repetative_joints[1] > 4)]

 

    #Check if rafter beam with no joint to opposite group exist
    for match in rafter_matches:
        all_beams_r = np.array([b.id for b in beams_new_db if b.roof_tile_id == match[0][0] or b.roof_tile_id == match[0][1]])
        all_beams_j = np.array([(j.beam1_id, j.beam2_id) for j in joints if j.b1_group == match[0][0] and j.b2_group == match[0][1]]).flatten()
        beams_has_no_joint = np.setdiff1d(all_beams_r, all_beams_j)

        rafter_joints = [j for j in joints if j.beam1_id in all_beams_j or j.beam2_id in all_beams_j]

        missing_rafter_candidates = [b for b in beams_new_db if b.id in beams_has_no_joint]
        test_joints = getJoints(missing_rafter_candidates, 0.5)
        candidate_matches = [j for j in test_joints if j.beam1_id in beams_has_no_joint or j.beam2_id in beams_has_no_joint]
        
        for i,b in enumerate(missing_rafter_candidates):
            for j in candidate_matches:
                if b.id == j.beam1_id or b.id == j.beam2_id:
                    b.obb.color = [1.,0.,0.]
        
        obbs = [b.obb for b in beams_new_db] # if b.id in [206,211,256]]
        o3d.visualization.draw([cover_hull, *obbs])
        
        ext_dict = extendRafterBeams(beams_new_db, candidate_matches, roof_tiles_db, config_data['max_joint_th'])
        ext_beams = ext_dict['extended_beams']
        ext_joints = ext_dict['extra_joints']
        
        ext_idx = np.array([b.id for b in ext_beams])
        all_beams_j = np.hstack((all_beams_j, ext_idx))
        
        if len(ext_beams):
            ext_obbs = [b.obb for b in ext_beams]
            for b in ext_obbs:
                b.color = [0.,1.,0.]
            o3d.visualization.draw([*obbs, *ext_obbs])
            
            print("Red beams are going to be extended as green ones (y/n):")
            answer = str(input())
            if answer == "y":
                updateExtendedBeams(roof_db, ext_beams)
                print("Beams are updated on database.\n")                     
            else:
                print("No change on beam_new table.\n")
                print("Refine the beam extend parameters.")
                sys.exit(1)
                
        else:
            print("No beam detected to extend. Go on for rafter definiton?: (y/n):")
            answer = str(input())
            if answer != "y":
                sys.exit(1)
        
        rafter_obbs = [b.obb for b in beams_new_db if b.id in all_beams_j]       
        for b in rafter_obbs:
            b.color = [0.,0.,1.]
        o3d.visualization.draw(rafter_obbs)
        
        print("Define blue beams as rafters on database? (y/n):")
        answer = str(input())
        if answer != "y":
            sys.exit(1)
        else:
            # Insert rafter parameters to db, update joint_new & beam_new
           
            if len(ext_joints):
                insert_joints = rafter_joints.extend(ext_joints)
            else:
                insert_joints = rafter_joints
            insertJointNews(roof_db, insert_joints, match[0])
            print("New joints created on db")
                        
            rafter_beams= [b for b in beams_new_db if b.id in all_beams_j]     
            rafters =  Rafter.getRaftersByJoints(insert_joints, rafter_beams)
            insertRafters(roof_db, rafters, match[0])
            print("Rafters created on db")

    end = datetime.now()
    print("Rafters defined :\t", (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rafter Definition")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)