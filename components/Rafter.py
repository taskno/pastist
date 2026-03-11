import numpy as np
import ezdxf
import open3d as o3d

import roof.Beam as Beam
import toolBox.geometry as geometry
from shapely.geometry import Point, MultiPoint, Polygon

class Rafter():
    def __init__(self, b1_id, b2_id, joint_id, plane, id = -1, rafter_type = None, truss_type = None, convex_hull = None, alphashape = None ):
        self.id = id
        self.b1_id = b1_id
        self.b2_id = b2_id
        self.joint_id = joint_id
        self.plane = plane
        self.rafter_type = rafter_type
        self.truss_type = truss_type
        self.convex_hull = convex_hull
        self.alphashape = alphashape
        self.convex_hull_3d = None
        self.b1_obj = None
        self.b2_obj = None

    def setBeamObjects(self, beams):
        self.b1_obj = Beam.getBeamById(beams, self.b1_id)
        self.b2_obj = Beam.getBeamById(beams, self.b2_id)

    def getConvexHull3D(self):
        vertices_2d = np.dstack(self.convex_hull.boundary.xy).tolist()[0][:-1]
        vertices_3d = [geometry.reproject2DPointToPlane3D(v, self.plane) for v in vertices_2d]

        planes = geometry.getParallelPlanes(self.plane, 0.05)

        all_vertices = []
        for p in vertices_3d:
            p1 = geometry.project3DPointToPlane(p, planes[0])
            p2 = geometry.project3DPointToPlane(p, planes[1])
            all_vertices.append(p1)
            all_vertices.append(p2)

        #all_vertices = np.vstack((self.b1_obj.vertices, self.b2_obj.vertices))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_vertices) 
        hull, _ = pcd.compute_convex_hull()

        #o3d.visualization.draw([pcd, hull])

        self.convex_hull_3d = hull
        return hull

class RafterTemplate():
    def __init__(self, beams):
        self.beams = beams


def getRaftersByJoints(joints, beams):

    corresponding_beams = []
    for j in joints:
        match_beams = [None, None]
        for b in beams:
            if j.beam1_id == b.id and match_beams[0] is None:
                match_beams[0] = (b)
            elif j.beam2_id == b.id and match_beams[1] is None:
                match_beams[1] = (b)
            elif match_beams[0] is not None and match_beams[1] is not None:
                break
        corresponding_beams.append(match_beams)

    rafters = []
    for i,j in enumerate(joints):
        b1 = corresponding_beams[i][0]
        b2 = corresponding_beams[i][1]

        
        if not (b1 is None or b2 is None):
            rafter_type = "rafter_top_" + str(b1.roof_tile_id) + "_" + str(b2.roof_tile_id)
            
            axis_pts = np.vstack((b1.axis, b2.axis))
            rafter_vertices = np.vstack((b1.vertices, b2.vertices))
            plane,ev,ew = geometry.getPlaneLS(axis_pts)
            rafter_vertices_2d = geometry.project3DPointsToPlane2D(rafter_vertices, plane)
            rafter_hull = MultiPoint(rafter_vertices_2d).convex_hull
            
            rafters.append(Rafter(b1_id=b1.id,b2_id=b2.id,joint_id= -1,plane= plane,id=-1,rafter_type = rafter_type, convex_hull = rafter_hull))

    return rafters