import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import scipy.stats as st
import copy
import open3d as o3d
from toolBox import geometry
from toolBox import imagePrc

class Segment:
    def __init__(self,id = -1, pts3D = [], pts2D = [], plane = [0.,0.,0.,0.], rmse = -1.,
                 ev1 = [0.,0.,0.], ev2 = [0.,0.,0.], ev3 = [0.,0.,0.],  
                 ew1 = [0.,0.,0.], ew2 = [0.,0.,0.], ew3 = [0.,0.,0.],
                 img={}, mbr = {}, 
                 mbrArea = -1., mbrWidth = -1., mbrHeight = -1., mbrVertices = [], mbrCoG = -1., mbrAxis = [],
                 fArea = -1., fElong = -1., type = None):
        self.id = id
        self.pts3D = pts3D
        self.pts2D = pts2D
        self.plane = plane
        self.rmse = rmse
        self.ev1 = ev1 # normal vector
        self.ev2 = ev2 # side axis
        self.ev3 = ev3 # longitudinal axis
        self.ew1 = ew1
        self.ew2 = ew2
        self.ew3 = ew3
        self.img = img # {"image": img, "size": img_size, "extent": img_ext}
        self.mbr = mbr # {"mbr_pixels": 4_pixel_coords, "mbr_size": with,height, "fArea": overlap_ratio}
        self.mbrArea = mbrArea
        self.mbrWidth = mbrWidth
        self.mbrHeight = mbrHeight
        self.mbrVertices = mbrVertices
        self.mbrCoG = mbrCoG
        self.mbrAxis = mbrAxis
        self.fArea = fArea
        self.fElong = fElong
        self.type = type

        #Set segment parameters based on points
        self.setPlaneParams()
        self.setImage()
        self.setfElong()
        self.setfArea()
        self.setType()

    def setPlaneParams(self):
        plane_params, ev_sorted, ew_sorted = geometry.getPlaneLS(self.pts3D)
        self.plane = plane_params[:4]  
        self.rmse = plane_params[4]
        self.ev1 = np.array([ev_sorted[:, 0]])[0]
        self.ev2 = np.array([ev_sorted[:, 1]])[0]
        self.ev3 = np.array([ev_sorted[:, 2]])[0]
        self.ew1 = ew_sorted[0]
        self.ew2 = ew_sorted[1]
        self.ew3 = ew_sorted[2]

    def setPts2D(self):
        self.pts2D= geometry.project3DPointsToPlane2D(np.array(self.pts3D), self.plane)

    def setImage(self):
        if len(self.pts2D) == 0:
            self.setPts2D()
        img, img_size, img_ext = imagePrc.getImageFromPoints(self.pts2D,scale = 1)
        self.img = {"image": img, "size": img_size, "extent": img_ext}
        self.mbr = imagePrc.getMBRStats(self.img["image"])

        self.mbrHeight = np.max(self.mbr["mbr_size"]) / self.img["size"]
        self.mbrWidth = np.min(self.mbr["mbr_size"]) / self.img["size"]
        self.mbrArea = self.mbrHeight * self.mbrWidth

        mbr_vertices_2D = np.asarray([imagePrc.image2CartesianCoordinates(p ,img_ext[0], img_ext[3], img_size) for p in self.mbr["mbr_pixels"]])
        self.mbrVertices = [geometry.reproject2DPointToPlane3D(p, self.plane) for p in mbr_vertices_2D]
        self.mbrCoG = np.mean(self.mbrVertices, axis=0)

    def setfArea(self):
        self.fArea= self.mbr["fArea"]

    def setfElong(self):
        self.fElong= np.sqrt(self.ew3 / self.ew2)

    def getMBRAxis(self):
        if len(self.mbrAxis) == 0:
            d = np.array([abs(geometry.getDistance(v, self.mbrVertices[0])) for i,v in enumerate(self.mbrVertices) if i >0]) ###Test2 abs not needed??
            idx = np.array([0,1,2,3])
            v1 = np.mean((self.mbrVertices[0], self.mbrVertices[np.argmin(d)+1]), axis = 0)
            opposites = [i for i in idx if i not in (0, np.argmin(d)+1)]
            v2 = np.mean((self.mbrVertices[opposites[0]], self.mbrVertices[opposites[1]]), axis = 0)
            self.mbrAxis = (v1,v2)
            return self.mbrAxis
        else:
            return self.mbrAxis

    def setType(self):
    #Shape classification decision tree
        if self.rmse > 0.04:
            self.type = "d" # non-Planar segment
        else:
            if self.fElong > 5 and self.fArea > 0.5:
                self.type = "a" #Linear segment
            elif self.fElong < 4.5 and self.fArea > 0.8:
                self.type = "c" #Compact segment
            else:
                self.type = "b" # Splittable segment

class Face:
    def __init__(self,points = [], segments = [], side = -1,  ev1 = [0.,0.,0.], ev2 = [0.,0.,0.], ev3 = [0.,0.,0.], plane  =[0.,0.,0.,0.], rmse = -1):
        self.points = points
        self.segments = segments
        self.ev1 = ev1 ### n
        self.ev2 = ev2 ### v
        self.ev3 = ev3 ### u
        self.plane = plane
        self.rmse = rmse
        self.global_cog = -1. ##TODO
        if len(segments) != 1:
            self.setParams()
        else:
            self.setParamsFromSegment()
    
    @classmethod
    def fromSegments(cls, segments):
        all_pts = np.vstack([s.pts3D for s in segments])
        return cls(all_pts, segments)

    def setParams(self):
        plane_params, ev_sorted, ew_sorted = geometry.getPlaneLS(self.points)
        self.plane = plane_params[:4]  
        self.rmse = plane_params[4]
        self.ev1 = np.array([ev_sorted[:, 0]])[0]
        self.ev2 = np.array([ev_sorted[:, 1]])[0]
        self.ev3 = np.array([ev_sorted[:, 2]])[0]
    
    def setParamsFromSegment(self):
        self.plane = self.segments[0].plane
        self.rmse = self.segments[0].rmse
        self.ev1 = self.segments[0].ev1
        self.ev2 = self.segments[0].ev2
        self.ev3 = self.segments[0].ev3

class Primitive:
    def __init__(self, faces, transform_faces = True):
        # Face list order : [front, left, right, back]
        self.face1 = faces[0]
        self.face2 = faces[1]
        self.face3 = faces[2]
        self.face4 = faces[3]
        self.transform_faces = transform_faces
        self.trans_mat = []
        self.trans_mat_inv = []
        self.trapezoid_params = []
        self.cuboid_params = []
        self.cog = None
        if transform_faces:
            self.transformFaces()

    def getCoG(self):
        #if self.cog is None:
        pts = self.face1.points
        if self.face2:
            pts = np.vstack((pts,self.face2.points))
        if self.face3:
            pts = np.vstack((pts,self.face3.points))
        if self.face4:
            pts = np.vstack((pts,self.face4.points))
        self.cog = np.mean(pts, axis=0)
        return self.cog
        #else:
        #    return self.cog

    def transformFaces(self):
        #Orientation approximation for better fitting results        
        #Front face
        pcd_front = o3d.geometry.PointCloud()
        pcd_front.points = o3d.utility.Vector3dVector(self.face1.points)
        #Left face
        if self.face2:
            pcd_left = o3d.geometry.PointCloud()
            pcd_left.points = o3d.utility.Vector3dVector(self.face2.points)
        #Right face
        if self.face3:
            pcd_right = o3d.geometry.PointCloud()
            pcd_right.points = o3d.utility.Vector3dVector(self.face3.points)
        #Back face
        if self.face4:
            pcd_back = o3d.geometry.PointCloud()
            pcd_back.points = o3d.utility.Vector3dVector(self.face4.points)

        global_cog = self.getCoG()

        cog_face1 = pcd_front.get_center()
        r1 = self.face1.ev2 # X direction
        r2 = self.face1.ev1 # Y direction // normal vector of face1
        r3 = self.face1.ev3 # Z direction
        r2 = geometry.orientNormalVector(cog_face1,r2,global_cog, True)# TODO if global cog was not computed      
        if self.face2:
            cog_face2 = pcd_left.get_center()
            r1 = geometry.orientNormalVector(cog_face1,r1,cog_face2, False)
        elif self.face2:
            cog_face3 = pcd_right.get_center()
            r1 = geometry.orientNormalVector(cog_face1,r1,cog_face3, True)

        s_trans1 = cog_face1 + r1
        s_trans2 = cog_face1 + r2
        s_trans3 = cog_face1 + r3
        t_trans1 = cog_face1 + [1., 0., 0.]
        t_trans2 = cog_face1 + [0., 1., 0.]
        t_trans3 = cog_face1 + [0., 0., -1.]
        
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector([cog_face1, s_trans1, s_trans2, s_trans3])
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector([cog_face1, t_trans1, t_trans2, t_trans3]) 
        corr_list = o3d.utility.Vector2iVector([[0,0],[1,1],[2,2],[3,3]])
        self.trans_mat = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(corr_list))
        
        global_cog = np.append(global_cog, 1.)
        global_cog_tr = np.matmul(self.trans_mat, global_cog)

        if global_cog_tr[1] < cog_face1[1]:
            t_trans3 = cog_face1 + [0., 0., 1.]
            target_pcd.points = o3d.utility.Vector3dVector([cog_face1, t_trans1, t_trans2, t_trans3]) 
            self.trans_mat = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(corr_list))
        
        self.trans_mat_inv = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(target_pcd, source_pcd, o3d.utility.Vector2iVector(corr_list))
        face1_pts = pcd_front.transform(self.trans_mat)
        self.face1 = Face(points = np.asarray(face1_pts.points))
        if self.face2:
            face2_pts = pcd_left.transform(self.trans_mat)
            self.face2 = Face(points = np.asarray(face2_pts.points))
        if self.face3:
            face3_pts = pcd_right.transform(self.trans_mat)
            self.face3 = Face(points = np.asarray(face3_pts.points))
        if self.face4:
            face4_pts = pcd_back.transform(self.trans_mat)
            self.face4 = Face(points = np.asarray(face4_pts.points))
    
        """
        #Old code
        cog_face1 = pcd_front.get_center()
        s_trans1 = cog_face1 + self.face1.ev3 * 1.
        s_trans2 = cog_face1 + self.face1.ev2 * 1.
        t_trans1 = cog_face1 + [0., 0., 1.]
        t_trans2 = cog_face1 + [1., 0., 0.] 
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector([cog_face1, s_trans1, s_trans2])
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector([cog_face1, t_trans1, t_trans2]) 
        corr_list = o3d.utility.Vector2iVector([[0,0],[1,1],[2,2]])   
        self.trans_mat = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(corr_list))
        self.trans_mat_inv = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(target_pcd, source_pcd, o3d.utility.Vector2iVector(corr_list))


        face1_pts = pcd_front.transform(self.trans_mat)
        self.face1 = Face(points = np.asarray(face1_pts.points))
        if self.face2:
            face2_pts = pcd_left.transform(self.trans_mat)
            self.face2 = Face(points = np.asarray(face2_pts.points))
        if self.face3:
            face3_pts = pcd_right.transform(self.trans_mat)
            self.face3 = Face(points = np.asarray(face3_pts.points))
        if self.face4:
            face2_pts = pcd_back.transform(self.trans_mat)
            self.face4 = Face(points = np.asarray(face4_pts.points))
        ## Check if transformation is valid 
        ## cog_face1.Y < global_cog.Y
        if self.face2:
            ref_cog = np.mean(self.face2.points, axis = 0)
        elif self.face3:
            ref_cog = np.mean(self.face3.points, axis = 0)
        elif self.face4:
            ref_cog = np.mean(self.face4.points, axis = 0)
        ref_cog = np.append(ref_cog, 1.)
        trans_test = np.vstack((self.trans_mat, [0.,0.,0.,1.]))
        ref_cog_tr = np.matmul(trans_test, ref_cog)

        if cog_face1[1] > ref_cog_tr[1]:
            t_trans1 = cog_face1 + [0., 0., -1.]
            target_pcd.points = o3d.utility.Vector3dVector([cog_face1, t_trans1, t_trans2]) 
            self.trans_mat = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(corr_list))
        self.trans_mat_inv = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(target_pcd, source_pcd, o3d.utility.Vector2iVector(corr_list))

 
        face1_pts = pcd_front.transform(self.trans_mat)
        self.face1 = Face(points = np.asarray(face1_pts.points))
        if self.face2:
            face2_pts = pcd_left.transform(self.trans_mat)
            self.face2 = Face(points = np.asarray(face2_pts.points))
        if self.face3:
            face3_pts = pcd_right.transform(self.trans_mat)
            self.face3 = Face(points = np.asarray(face3_pts.points))
        if self.face4:
            face2_pts = pcd_back.transform(self.trans_mat)
            self.face4 = Face(points = np.asarray(face4_pts.points))

        # Face order may change after transformation!!
        # re-check face order
        front_cog = pcd_front.get_center()
        face2_copy = None
        face3_copy = None
        swap_lr = False
        if self.face2:
            left_cog = pcd_left.get_center()

            if left_cog[0] > front_cog[0]:
                swap_lr = True
                face2_copy = copy.deepcopy(self.face2)
                face3_copy = copy.deepcopy(self.face3)
        elif self.face3:
            right_cog = pcd_right.get_center()
            if right_cog[0] < front_cog[0]:
                swap_lr = True
                face3_copy = copy.deepcopy(self.face3)
                face2_copy = copy.deepcopy(self.face2)
        if swap_lr:
                self.face3 = face2_copy
                self.face2 = face3_copy
        """

    def fit_trapezoidal_prism(self):
        #Estimate starting parameters
        try:
            pts = np.vstack((self.face1.points, self.face2.points, self.face3.points))
        except:
            return self.fit_cuboid()
        CoG = pts.mean(axis=0)
        self.face1.global_cog = CoG
        
        d_cog = - np.dot(self.face1.ev3, CoG)
        plane_cog = np.append(self.face1.ev3, d_cog)
        p0 = geometry.intersection_point(self.face1.plane,self.face2.plane, plane_cog)
        p1 = geometry.intersection_point(self.face1.plane,self.face3.plane, plane_cog)
        
        t = geometry.getPoint2PlaneDistance(CoG,self.face1.plane)
        
        d_s = - np.dot(self.face1.ev2, p0)
        plane_s = np.append(self.face1.ev2, d_s)
        s = geometry.getPoint2PlaneDistance(CoG, plane_s)
        
        a = abs(geometry.getDistance(p0, p1)) #Test1: Abs not needed!
        
        #proof of cog-p0
        p0_ = CoG - self.face1.ev2 * s - self.face1.ev1 * t     
        theta = (geometry.getAngleBetweenVectors(self.face1.ev1, self.face2.ev1) + geometry.getAngleBetweenVectors(self.face1.ev1, self.face3.ev1)) / 2.

        initial_params = [0., 0., 0., s, t, a, 0.1, theta] # [alpha,beta,gamma,s,t,a,b,theta]
        
        start = datetime.now()
        optimization_result = minimize(trapezoid_function, initial_params, args=(self.face1,self.face2,self.face3,self.face4), method='SLSQP')#SLSQP, options={'ftol':0.0001, 'maxiter':1000})
        end = datetime.now()
        print("7 Params optimization: ", end - start)
        
        fitted_primitive = optimization_result.x
        self.trapezoid_params = optimization_result.x
        return fitted_primitive

    def get_trapezoidal_prsim_mesh(self):
        fitting_parameters = self.trapezoid_params
        # [alpha,beta,gamma,s,t,a,b,theta]
        r2 = np.asarray(self.face1.ev1)
        r3 = np.asarray(self.face1.ev3)
        r1 = np.asarray(self.face1.ev2)
        
        #Be sure the axes directions correct1
        if r3[2] > 0:
            r3 *= -1
        if r2[1] < 0:
            r2 *= -1
        if r1[0] < 0:
            r1 *= -1
        
        #Rotate around r1 (alpha), r2(beta) and r3(gamma)
        alpha = fitting_parameters[0]  * np.pi / 180.
        r2_1= geometry.rotateVector(r2, r1, alpha)
        r3_1= geometry.rotateVector(r3, r1, alpha)
        #if r3_1[2] > 0:
        #    r3_1 *= -1
        #if r2_1[1] < 0:
        #    r2_1 *= -1
        
        beta = fitting_parameters[1]  * np.pi / 180.
        r1_1= geometry.rotateVector(r1, r2_1, beta)
        r3_2= geometry.rotateVector(r3_1, r2_1, beta)
        #if r3_2[2] > 0:
        #    r3_2 *= -1
        #if r1_1[0] < 0:
        #    r1_1 *= -1
        
        gamma = fitting_parameters[2]  * np.pi / 180.
        r1_2= geometry.rotateVector(r1_1, r3_2, gamma)
        r2_2= geometry.rotateVector(r2_1, r3_2, gamma)
        if r2_2[1] < 0:
            r2_2 *= -1
        if r1_2[0] < 0:
            r1_2 *= -1
        
        if r3_2[2] > 0:
            r3_2 *= -1
        
        r1 = geometry.getUnitVector(r1_2)
        r2 = geometry.getUnitVector(r2_2)
        r3 = geometry.getUnitVector(r3_2)
        
        CoG = self.getCoG()
        p0 = CoG - r1 * abs(fitting_parameters[3]) - r2 * abs(fitting_parameters[4])
      
        d_front = - np.dot(p0, r2)
        plane_1 = np.append(r2, d_front) # face1
        
        if len(fitting_parameters) == 8:
            #Trapezoid
            theta = fitting_parameters[7] * np.pi / 180.
        else:
            #Cuboid
            theta = np.pi / 2.
        n2 = geometry.rotateVector(r2, r3, theta)
        n2 = geometry.getUnitVector(n2)
        d_2 = - np.dot(p0, n2)
        plane_2 = np.append(n2, d_2)
        
        n3 = geometry.rotateVector(r2, r3, - theta)
        n3 = geometry.getUnitVector(n3)
        d_3 = - np.dot(p0 + abs(fitting_parameters[5]) * r1, n3) #####Check change: ABS
        plane_3 = np.append(n3, d_3)

        #all_pts = np.vstack((self.face1.points, self.face2.points,self.face3.points)) 
        all_pts= self.face1.points
        if self.face2:
            all_pts = np.vstack((all_pts, self.face2.points))
        if self.face3:
            all_pts = np.vstack((all_pts, self.face3.points))
        if self.face4:
            all_pts = np.vstack((all_pts, self.face4.points))
        
        zmin = np.min(all_pts[:,2:])
        zmax = np.max(all_pts[:,2:])
        
        pmin = all_pts[np.argwhere(all_pts[:,2:] == zmin)][0][0]
        pmax = all_pts[np.argwhere(all_pts[:,2:] == zmax)][0][0]
        dbot = - np.dot(r3, pmin)
        dtop = - np.dot(r3, pmax)
        
        plane_bot = np.append(r3, dbot)
        plane_top = np.append(r3, dtop)
        
        p1 = geometry.intersection_point(plane_1,plane_2, plane_bot)
        p2 = geometry.intersection_point(plane_1,plane_2, plane_top)
        p3 = geometry.intersection_point(plane_1,plane_3, plane_bot)
        p4 = geometry.intersection_point(plane_1,plane_3, plane_top)
           
        if self.face3 and self.face2 == None:
            ### Redefine plane2 differently

            ref_p0 = self.face1.points[np.argmin(self.face1.points[:,0])]
            p0_ = p3 - r1 * abs(fitting_parameters[5])
            d_2_ = - np.dot(p0_, n2)
            plane_2_ = np.append(n2, d_2_)
            p1 = geometry.intersection_point(plane_1,plane_2_, plane_bot)
            p2 = geometry.intersection_point(plane_1,plane_2_, plane_top)
            
        #dist_cog_front = abs(geometry.getDistance(CoG, p0))
        #p_back = p0 + r2 * abs(fitting_parameters[4]) * 2

        if self.face4 is None:
            if self.face2:
                c2 = np.median(self.face2.points, axis = 0)
            if self.face3:
                c3 = np.median(self.face3.points, axis = 0)
            if self.face2 and self.face3:
                local_center = np.mean((c2,c3), axis =0)
            elif self.face2:
                local_center = c2
            elif self.face3:
                local_center = c3
            dist_back = abs(geometry.getPoint2PlaneDistance(local_center, plane_1))
            dist_back *= 2.

        else:
            # We have already Face4/plane4 to use!!!           
            dist_back = abs(fitting_parameters[6])

        #dist_back = abs(geometry.getPoint2PlaneDistance(local_center, plane_1))
        p_back = p0 + r2 * dist_back



        ##  Temporary solution: Use farthest point from front face!
        #distances2 = np.array([geometry.getPoint2PlaneDistance(p,plane_1) for p in self.face2.points])
        #distances3 = np.array([geometry.getPoint2PlaneDistance(p,plane_1) for p in self.face3.points])
        #
        #if np.max(distances2) >= np.max(distances3):
        #    p_back = self.face2.points[np.argmax(distances2)]
        #else:
        #    p_back = self.face3.points[np.argmax(distances3)]
        
        d_back = - np.dot(plane_1[:3], p_back)
        plane_back = np.append(plane_1[:3], d_back)
     
        p5 = geometry.intersection_point(plane_back,plane_2, plane_bot)
        p6 = geometry.intersection_point(plane_back,plane_2, plane_top)
        p7 = geometry.intersection_point(plane_back,plane_3, plane_bot)
        p8 = geometry.intersection_point(plane_back,plane_3, plane_top)
        
        trapezoid_pts = np.vstack((p1,p2,p3,p4,p5,p6,p7,p8))
        trapezoid_pcd = o3d.geometry.PointCloud()
        trapezoid_pcd.points = o3d.utility.Vector3dVector(trapezoid_pts)  
        
        if self.transform_faces:
            trapezoid_pcd = trapezoid_pcd.transform(self.trans_mat_inv)

        trapezoid_hull, _ = trapezoid_pcd.compute_convex_hull()       
        return trapezoid_hull
        
    def fit_cuboid(self):
        #Estimate starting parameters
        CoG = self.getCoG()
        self.face1.global_cog = CoG
        d_cog = - np.dot(self.face1.ev3, CoG)
        plane_cog = np.append(self.face1.ev3, d_cog)

        b = 0.1 ## TODO: Check all cases 

        if self.face2 and self.face3 and self.face4:
            p0 = geometry.intersection_point(self.face1.plane,self.face2.plane, plane_cog)
            p1 = geometry.intersection_point(self.face1.plane,self.face3.plane, plane_cog)
            
            t = geometry.getPoint2PlaneDistance(CoG,self.face1.plane)       
            d_s = - np.dot(self.face1.ev2, p0)
            plane_s = np.append(self.face1.ev2, d_s)
            s = geometry.getPoint2PlaneDistance(CoG, plane_s)    
            a = abs(geometry.getDistance(p0, p1))

            face4_cog = np.mean(self.face4.points, axis = 0)
            b = geometry.getPoint2PlaneDistance(face4_cog, self.face1.plane)

        elif self.face2 and self.face3:
            p0 = geometry.intersection_point(self.face1.plane,self.face2.plane, plane_cog)
            p1 = geometry.intersection_point(self.face1.plane,self.face3.plane, plane_cog)

            t = geometry.getPoint2PlaneDistance(CoG,self.face1.plane)       
            d_s = - np.dot(self.face1.ev2, p0)
            plane_s = np.append(self.face1.ev2, d_s)
            s = geometry.getPoint2PlaneDistance(CoG, plane_s)    
            a = abs(geometry.getDistance(p0, p1))

            local_center = np.median(np.vstack((self.face2.points, self.face3.points)), axis = 0) # To estimate b
            b = geometry.getPoint2PlaneDistance(local_center, self.face1.plane) * 2.


        elif self.face2:
            p0 = geometry.intersection_point(self.face1.plane,self.face2.plane, plane_cog)
            
            t = geometry.getPoint2PlaneDistance(CoG,self.face1.plane)       
            d_s = - np.dot(self.face1.ev2, p0)
            plane_s = np.append(self.face1.ev2, d_s)
            s = geometry.getPoint2PlaneDistance(CoG, plane_s)    

            c_1 = np.median(self.face1.points, axis = 0)            
            a = geometry.getPoint2PlaneDistance(c_1, plane_s) * 2

            local_center = np.median(self.face2.points, axis = 0) # To estimate b
            b = geometry.getPoint2PlaneDistance(local_center, self.face1.plane) * 2.
            

        elif self.face3 and self.face2 is None:
            pass # This case switched to face1-face2 match
            
            """
            p1 = geometry.intersection_point(self.face1.plane,self.face3.plane, plane_cog)

            t = geometry.getPoint2PlaneDistance(CoG,self.face1.plane)
            
            d_s3 = - np.dot(self.face1.ev2, p1)
            plane_s3 = np.append(self.face1.ev2, d_s3)
            #s = geometry.getPoint2PlaneDistance(CoG, plane_s)    

            c_1 = np.median(self.face1.points, axis = 0)     
            a = geometry.getPoint2PlaneDistance(c_1, plane_s3) * 2

            p0 = p1 - self.face1.ev2 * abs(a)
            d_s = - np.dot(self.face1.ev2, p0)
            plane_s = np.append(self.face1.ev2, d_s)
            s = geometry.getPoint2PlaneDistance(CoG, plane_s) #abs(a) - abs(geometry.getPoint2PlaneDistance(CoG, plane_s3))#geometry.getPoint2PlaneDistance(CoG, plane_s)

            local_center = np.mean(self.face3.points, axis =0) # To estimate b
            b = geometry.getPoint2PlaneDistance(local_center, self.face1.plane) * 2.
            """

        elif self.face4 and self.face2 is None and self.face3 is None:
            t = geometry.getPoint2PlaneDistance(CoG,self.face1.plane)
            cog = self.getCoG()
            d_c = - np.dot(self.face1.ev2, cog)
            plane_c = np.append(self.face1.ev2, d_c)

            pts = np.vstack((self.face1.points, self.face4.points))
            #x_coor = pts[:,0]
            #std_x = np.std(x_coor)
            #sp0 = pts[np.argmin(x_coor)]
            #sp1 = pts[np.argmax(x_coor)]

            signed_dists = np.asarray([geometry.getSignedDistance(cog, cog + self.face1.ev2, p) for p in pts])
            #signed_dists_0 = signed_dists[np.argwhere(signed_dists < 0)]
            #signed_dists_1 = signed_dists[np.argwhere(signed_dists > 0)]
            #s0 = np.median(signed_dists_0, axis=0)
            #s1 = np.median(signed_dists_1, axis=0)
            #
            #sp0 = cog + self.face1.ev2 * s0 * 2.
            #sp1 = cog + self.face1.ev2 * s1 * 2.

            #sp0 = pts[np.argmin(signed_dists)]
            #sp1 = pts[np.argmax(signed_dists)]

            bin_width = 0.05
            z_coor = pts[:,2]     
            bin_start = z_coor.min()
            bin_end = z_coor.min() + bin_width
            min_sd = []
            max_sd = []
            while True:
                if bin_end < z_coor.max():
                    sd = signed_dists[np.argwhere((z_coor >= bin_start) & (z_coor < bin_end))]
                    if len(sd):
                        min_sd.append(min(sd))
                        max_sd.append(max(sd))
                    bin_start = bin_end
                    bin_end = bin_start + bin_width
                else:
                    sd = signed_dists[np.argwhere((z_coor >= bin_start) & (z_coor < bin_end))]
                    if len(sd):
                        min_sd.append(min(sd))
                        max_sd.append(max(sd))
                    bin_start = bin_end
                    bin_end = bin_start + bin_width
                    break

            min_sd = np.asarray(min_sd)
            max_sd = np.asarray(max_sd)

            dist0 = abs(np.median(min_sd))
            dist1 = abs(np.median(max_sd))
            sp0 = cog - self.face1.ev2 * dist0
            sp1 = cog + self.face1.ev2 * dist1

            sd0 = - np.dot(self.face1.ev2, sp0)
            plane0 = np.append(self.face1.ev2, sd0)
            sd1 = - np.dot(self.face1.ev2, sp1)
            plane1 = np.append(self.face1.ev2, sd1)

            p0 = geometry.intersection_point(self.face1.plane, plane0, plane_cog)
            p1 = geometry.intersection_point(self.face1.plane, plane1, plane_cog)
                            
            d_s = - np.dot(self.face1.ev2, p0)
            plane_s = np.append(self.face1.ev2, d_s)# Check if necessary, can be replaced by plane0

            s = geometry.getPoint2PlaneDistance(CoG, plane_s)    
            a = abs(geometry.getDistance(p0, p1))

            face4_cog = np.mean(self.face4.points, axis = 0)
            b = geometry.getPoint2PlaneDistance(face4_cog, self.face1.plane)
            
      
        #proof of cog-p0
        p0_ = CoG - self.face1.ev2 * abs(s) - self.face1.ev1 *abs(t)     
        initial_params = [0., 0., 0., s, t, a, b] # [alpha,beta,gamma,s,t,a,b]
        
        optimization_result = minimize(cuboid_function, initial_params, args=(self.face1,self.face2,self.face3,self.face4), method='SLSQP')
        
        fitted_primitive = optimization_result.x
        self.trapezoid_params = optimization_result.x
        return fitted_primitive

def trapezoid_function(inital_params, face1, face2, face3, face4):  
        #r1 : local x axis 1,0,0
        #r2 : local y axis (normal vector of front face) 0,1,0
        #r3 : local z axis (longitudinal axis), 0,0,-1
        #inital_parameters[alpha, beta, gamma, s, t, a, b, theta]
        
        r2 = np.asarray(face1.ev1)
        r3 = np.asarray(face1.ev3)
        r1 = np.asarray(face1.ev2)
        
        #Be sure the axes directions correct!!
        if r3[2] > 0:
            r3 *= -1
        if r2[1] < 0:
            r2 *= -1
        if r1[0] < 0:
            r1 *= -1
        
        #Rotate around r1 (alpha), r2(beta) and r3(gamma)
        alpha = inital_params[0]  * np.pi / 180.
        r2_1= geometry.rotateVector(r2, r1, alpha)
        r3_1= geometry.rotateVector(r3, r1, alpha)
        #if r3_1[2] > 0:
        #    r3_1 *= -1
        #if r2_1[1] < 0:
        #    r2_1 *= -1
        
        beta = inital_params[1]  * np.pi / 180.
        r1_1= geometry.rotateVector(r1, r2_1, beta)
        r3_2= geometry.rotateVector(r3_1, r2_1, beta)
        #if r3_2[2] > 0:
        #    r3_2 *= -1
        #if r1_1[0] < 0:
        #    r1_1 *= -1
        
        gamma = inital_params[2]  * np.pi / 180.
        r1_2= geometry.rotateVector(r1_1, r3_2, gamma)
        r2_2= geometry.rotateVector(r2_1, r3_2, gamma)
        if r2_2[1] < 0:
            r2_2 *= -1
        if r1_2[0] < 0:
            r1_2 *= -1
        
        if r3_2[2] > 0:
            r3_2 *= -1
        
        r1 = geometry.getUnitVector(r1_2)
        r2 = geometry.getUnitVector(r2_2)
        r3 = geometry.getUnitVector(r3_2)
        
        #Estimate p0 via CoG
        #pts = np.vstack((face1.points, face2.points, face3.points))
        #CoG = pts.mean(axis=0)
        CoG = face1.global_cog
        p0 = CoG - r1 * abs(inital_params[3]) - r2 * abs(inital_params[4]) #TODO CHeck!!
        
        d_front = - np.dot(p0, r2)
        plane_1 = np.append(r2, d_front) # face1
        
        n2 = geometry.rotateVector(r2, r3, inital_params[7] * np.pi / 180.)
        n2 = geometry.getUnitVector(n2)
        d_2 = - np.dot(p0, n2)
        plane_2 = np.append(n2, d_2)
        
        n3 = geometry.rotateVector(r2, r3, - inital_params[7] * np.pi / 180.)
        n3 = geometry.getUnitVector(n3)
        d_3 = - np.dot(p0 + inital_params[5] * r1, n3)
        plane_3 = np.append(n3, d_3)

        #angle1 = geometry.getAngleBetweenVectors(r1, r2)
        #angle2 = geometry.getAngleBetweenVectors(r1, r3)
        #angle3 = geometry.getAngleBetweenVectors(r2, r3)
                       
        dist1 = geometry.getPoints2PlaneDistances(face1.points, plane_1)
        dist2 = geometry.getPoints2PlaneDistances(face2.points, plane_2)
        dist3 = geometry.getPoints2PlaneDistances(face3.points, plane_3)
        
        sum_sq_dist = np.dot(dist1, dist1) + np.dot(dist2, dist2) + np.dot(dist3, dist3)
        #rmse = np.sqrt(total_distance / (len(dist1) + len(dist2) + len(dist3)))
        return sum_sq_dist

def cuboid_function(inital_params, face1, face2, face3, face4):  
        #r1 : local x axis 1,0,0
        #r2 : local y axis (normal vector of front face) 0,1,0
        #r3 : local z axis (longitudinal axis), 0,0,-1
        #inital_parameters[alpha, beta, gamma, s, t, a, b, theta]
        
        r2 = np.asarray(face1.ev1)
        r3 = np.asarray(face1.ev3)
        r1 = np.asarray(face1.ev2)
        
        #Be sure the axes directions correct!!
        if r3[2] > 0:
            r3 *= -1
        if r2[1] < 0:
            r2 *= -1
        if r1[0] < 0:
            r1 *= -1
        
        #Rotate around r1 (alpha), r2(beta) and r3(gamma)
        alpha = inital_params[0]  * np.pi / 180.
        r2_1= geometry.rotateVector(r2, r1, alpha)
        r3_1= geometry.rotateVector(r3, r1, alpha)
        #if r3_1[2] > 0:
        #    r3_1 *= -1
        #if r2_1[1] < 0:
        #    r2_1 *= -1
        
        beta = inital_params[1]  * np.pi / 180.
        r1_1= geometry.rotateVector(r1, r2_1, beta)
        r3_2= geometry.rotateVector(r3_1, r2_1, beta)
        #if r3_2[2] > 0:
        #    r3_2 *= -1
        #if r1_1[0] < 0:
        #    r1_1 *= -1
        
        gamma = inital_params[2]  * np.pi / 180.
        r1_2= geometry.rotateVector(r1_1, r3_2, gamma)
        r2_2= geometry.rotateVector(r2_1, r3_2, gamma)
        if r2_2[1] < 0:
            r2_2 *= -1
        if r1_2[0] < 0:
            r1_2 *= -1
        
        if r3_2[2] > 0:
            r3_2 *= -1
        
        r1 = geometry.getUnitVector(r1_2)
        r2 = geometry.getUnitVector(r2_2)
        r3 = geometry.getUnitVector(r3_2)
        
        #Estimate p0 via CoG
        #pts = np.vstack((face1.points, face2.points, face3.points))
        #CoG = pts.mean(axis=0)
        CoG = face1.global_cog
        p0 = CoG - r1 * abs(inital_params[3]) - r2 * abs(inital_params[4]) #TODO CHeck!!
        
        d_front = - np.dot(p0, r2)
        plane_1 = np.append(r2, d_front) # face1
        
        n2 = r1 #geometry.rotateVector(r2, r3,  np.pi / 2.)
        #n2 = geometry.getUnitVector(n2)
        d_2 = - np.dot(p0, n2)
        plane_2 = np.append(n2, d_2)
        
        n3 = r1 #geometry.rotateVector(r2, r3, -np.pi / 2.)
        #n3 = geometry.getUnitVector(n3)

        d_3 = - np.dot(p0 + inital_params[5] * r1, n3)
        plane_3 = np.append(n3, d_3)

        d_4 = - np.dot(p0 + inital_params[6] * r2 , r2)
        plane_4 = np.append(r2, d_4)
        
        #angle1 = geometry.getAngleBetweenVectors(r1, r2)
        #angle2 = geometry.getAngleBetweenVectors(r1, r3)
        #angle3 = geometry.getAngleBetweenVectors(r2, r3)
                       
        dist1 = geometry.getPoints2PlaneDistances(face1.points, plane_1)
        if face2:
            dist2 = geometry.getPoints2PlaneDistances(face2.points, plane_2)
        else:
            dist2 = 0.
        if face3:
            dist3 = geometry.getPoints2PlaneDistances(face3.points, plane_3)
        else:
            dist3 = 0.
        if face4:
            dist4 = geometry.getPoints2PlaneDistances(face4.points, plane_4)
        else:
            dist4 = 0.
        sum_sq_dist = np.dot(dist1, dist1) + np.dot(dist2, dist2) + np.dot(dist3, dist3) + np.dot(dist4, dist4)
        #rmse = np.sqrt(total_distance / (len(dist1) + len(dist2) + len(dist3)))
        return sum_sq_dist