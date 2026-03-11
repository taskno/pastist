import numpy as np
import copy
from sklearn.decomposition import PCA
from scipy  import optimize
from scipy.optimize import least_squares

import open3d as o3d

import toolBox.geometry as geometry
import components.Beam as Beam

def getPointsInBox(pcd, obb):
    idx = obb.get_point_indices_within_bounding_box(pcd.points)   
    pts_in_box = pcd.select_by_index(idx, invert=False)
    return pts_in_box

def getTemplateOBB(obb, target_length = 0.5, target_dims = None):
    #obb_beam = Beam.obb2Beam(obb)
    #end_point = obb_beam.axis[0] + target_length * obb_beam.unit_vector

    #test_obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, (target_length, target_dims[0], target_dims[1]))
    #o3d.visualization.draw([obb, test_obb])

    if target_dims is not None:
        new_obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, (target_length, target_dims[0], target_dims[1]))
        new_obb.color = [1,0,0]
    else:
        new_obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, (target_length, obb.extent[1], obb.extent[2]))
        new_obb.color = [1,1,0]
    
    """
    #Evaluation of the methot
    d_plane = - np.dot(obb_beam.unit_vector, end_point)
    ref_plane = np.append(obb_beam.unit_vector, d_plane)    
    top_vertices = [geometry.project3DPointToPlane(p,ref_plane) for p in obb_beam.vertices[4:]]
    vertices_all = [*obb_beam.vertices[:4],*top_vertices]    
    new_obb = o3d.geometry.OrientedBoundingBox()
    new_obb =  new_obb.create_from_points(points=o3d.utility.Vector3dVector(vertices_all))
    """
    return new_obb

def sampleOBB(template_obb, nr_samples = 500):

    obb_vertices = np.asarray(template_obb.get_box_points())
    obb_pcd = o3d.geometry.PointCloud()
    obb_pcd.points = o3d.utility.Vector3dVector(obb_vertices) 
    hull, _ = obb_pcd.compute_convex_hull()
    #o3d.io.write_triangle_mesh(file_name, all_mesh)

    sampled_pcd = hull.sample_points_uniformly(number_of_points=int(nr_samples))
    #o3d.visualization.draw([template_obb, obb_pcd, hull, sampled_pcd])

    return sampled_pcd


def getBeamInSearchBox(pcd, search_obb, template_obb, template_len = 15., target_dims = None, check_density = False, vis = False):

    # Get the template and resize along longitudinal axis
    #template_obb_ext = getTemplateOBB(template_obb, target_length = template_len, target_dims =(0.2,0.2))

    if target_dims is None:
        template_obb_ext = getTemplateOBB(template_obb, target_length = template_len)
    else:
        template_obb_ext = getTemplateOBB(template_obb, target_length = template_len, target_dims =target_dims)
 
    #Get points in searc box
    target_pcd = getPointsInBox(pcd, search_obb)

    if len(target_pcd.points) < 300:
        return None
    else:
        # Coarse registration
        trans_vec = search_obb.center - template_obb_ext.center
        template_obb_ext.translate(trans_vec, True)
        
        # Sampling of the template
        temp_pcd = sampleOBB(template_obb_ext, int(template_len* 1000))
        if vis:
            o3d.visualization.draw([template_obb_ext, temp_pcd, search_obb])
        
        #ICP registration template -> points in searchbox
        init = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        icp_transform = o3d.pipelines.registration.registration_icp(
            temp_pcd, target_pcd, 0.2,init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())#TransformationEstimationPointToPlane())
        
        temp_pcd_icp = temp_pcd.transform(icp_transform.transformation)
        if vis:
            o3d.visualization.draw([pcd, temp_pcd, search_obb, target_pcd, temp_pcd_icp])
        
        #temp_obb_unit_reg = copy.deepcopy(temp_obb_unit)
        #temp_obb_unit_reg = temp_obb_unit.transform(icp_transform.transformation) # convert to pcd -> translate -> get obb
        
        box_pts = o3d.geometry.PointCloud()
        box_pts.points = o3d.utility.Vector3dVector(template_obb_ext.get_box_points())

        box_pts = box_pts.transform(icp_transform.transformation)
        template_obb_ext_icp = box_pts.get_oriented_bounding_box()
        #template_obb_ext_icp = o3d.geometry.OrientedBoundingBox()
        #template_obb_ext_icp =  template_obb_ext_icp.create_from_points(points=temp_pcd_icp.points, robust=True)

        #o3d.visualization.draw([template_obb_ext, box_pts, template_obb_ext_icp, temp_pcd_icp])        
        # To define longitudinal axis start&end get points in icp registered box
        
        pts_in_reg_obb = getPointsInBox(target_pcd, template_obb_ext_icp)
        
        if len(pts_in_reg_obb.points) < 300:
            return None
        else:
            if check_density:
                labels = np.array(pts_in_reg_obb.cluster_dbscan(0.2, 300))                
                idx = np.argwhere(labels >= 0)               
                labeled_pts = pts_in_reg_obb.select_by_index(idx, invert=False)                
                #o3d.visualization.draw([pts_in_reg_obb, labeled_pts])                
                obb_2 = labeled_pts.get_oriented_bounding_box()
            else:
                obb_2 = pts_in_reg_obb.get_oriented_bounding_box()
            if vis:
                o3d.visualization.draw([pcd, target_pcd, pts_in_reg_obb, template_obb_ext_icp, obb_2])          
            #Refine the limits of oriented cuboid
            beam_ref = Beam.obb2Beam(template_obb_ext_icp)   
            beam_limits = Beam.obb2Beam(obb_2)
            beam_ref.extendAlongLongitudinalAxis(beam_limits.axis[0], beam_limits.axis[1], True)
            return beam_ref.obb

def getRegisteredOBB(obb, pcd, threshold=0.2, max_iter = 30):
    temp_pcd = sampleOBB(obb, int(max(obb.extent)* 1000))
    #ICP registration template -> points in searchbox
    init = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    icp_transform = o3d.pipelines.registration.registration_icp(
        temp_pcd, pcd, threshold,init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

    #icp_transform = o3d.pipelines.registration.registration_generalized_icp(
    #    temp_pcd, pcd, threshold,init,
    #    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    
    if icp_transform.fitness != 0:
        temp_pcd_icp = temp_pcd.transform(icp_transform.transformation)
        result_obb = temp_pcd_icp.get_minimal_oriented_bounding_box(robust=True)
        #o3d.visualization.draw([temp_pcd,pcd, obb, result_obb])
        return result_obb
    else:
        return obb

def translateOBB(obb, trans_mat):
    obb_pcd = o3d.geometry.PointCloud()
    obb_pcd.points = o3d.utility.Vector3dVector(obb.get_box_points())
    trans_pcd = obb_pcd.transform(trans_mat)
    trans_obb = trans_pcd.get_minimal_oriented_bounding_box(robust=True)
    return trans_obb

def getICPTransform(pcd_source, pcd_target, threshold=0.2, max_iter = 30):
      init = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
      icp_transform = o3d.pipelines.registration.registration_icp(
      pcd_source, pcd_target, threshold,init,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
      return icp_transform

def getReverseTransformationMat(transformation_mat):
    R = transformation_mat[:,:3][:3]
    T = transformation_mat[:,3:][:3]

    RT = R.transpose()
    T2 = np.matmul(RT,-T)

    reverse_mat = np.hstack((RT,T2))
    reverse_mat = np.vstack((reverse_mat, transformation_mat[3]))
    return reverse_mat