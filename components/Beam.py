import numpy as np
import open3d as o3d
import toolBox.geometry as geometry
import toolBox.template as template
from sklearn.cluster import KMeans, MeanShift

class Beam():

    def __init__(self, id ,axis, vertices = None, 
                 group_id = None, cluster_id = None, rafter_id= None, roof_tile_id = None,
                unit_vector = None, width = None, height = None, length = None, comment = None, old_id = None):
        self.id = id
        self.group_id = group_id
        self.cluster_id = cluster_id
        self.rafter_id = rafter_id
        self.roof_tile_id = roof_tile_id

        self.unit_vector = np.array(unit_vector)
        self.vertices = np.array(vertices)
        self.axis = np.array(axis)

        self.width = width
        self.height = height
        self.length = length
        self.comment = comment
        self.old_id = old_id

        self.truss_id = None
        self.obb = None
        self.convex_hull_3d = None

        self.img_overlap = None
        
    def setOBB(self):
        if self.vertices is not None:
            self.obb = o3d.geometry.OrientedBoundingBox()
            self.obb = self.obb.create_from_points(points=o3d.utility.Vector3dVector(self.vertices))
            return self.obb

    def extendAlongLongitudinalAxis(self, ext_point1, ext_point2, shorten = False):
        #d1 = geometry.getDistance(axis[0], ext_point1) # distance to start point
        #d2 = geometry.getDistance(axis[1], ext_point2) # distance to start point
        if self.obb is None:
            setOBB()

        vertices1 = []
        if ext_point1 is not None:
            d1 = - np.dot(self.unit_vector, ext_point1)
            plane1 = np.append(self.unit_vector , d1)
            vertices1 = [geometry.project3DPointToPlane(p, plane1) for p in self.vertices[:4]]

            ext_p1_on_axis = geometry.project3DPointToPlane(self.obb.center, plane1)

        vertices2 = []
        if ext_point2 is not None:
            d2 = - np.dot(self.unit_vector, ext_point2)
            plane2 = np.append(self.unit_vector , d2)
            vertices2 = [geometry.project3DPointToPlane(p, plane2) for p in self.vertices[:4]]

            ext_p2_on_axis = geometry.project3DPointToPlane(self.obb.center, plane2)

        if not shorten:
            all_vertices = [*self.vertices, *vertices1, *vertices2]

        else:
            all_vertices = [*vertices1, *vertices2]          
            new_center = np.mean((ext_p1_on_axis, ext_p2_on_axis), axis = 0)
            new_length = geometry.getDistance(ext_p1_on_axis,ext_p2_on_axis)
            new_obb =  o3d.geometry.OrientedBoundingBox(self.obb.center, self.obb.R, (new_length, self.obb.extent[1], self.obb.extent[2]))
            #o3d.visualization.draw([self.obb, new_obb])
        #TODO: more flexible decision: define if extension points shorten / not
        new_obb = o3d.geometry.OrientedBoundingBox()
        new_obb = self.obb.create_from_points(points=o3d.utility.Vector3dVector(all_vertices))
        # o3d.visualization.draw([self.obb, new_obb])
        tmp_beam = obb2Beam(new_obb)

        self.axis = tmp_beam.axis
        self.vertices = tmp_beam.vertices
        self.width = tmp_beam.width
        self.height = tmp_beam.height
        self.length = tmp_beam.length
        self.obb = new_obb

    def getConvexHull3D(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vertices) 
        hull, _ = pcd.compute_convex_hull()
        self.convex_hull_3d = hull
        return hull

def obb2Beam(obb, orientation = [1000, 1000, 1000], comment = None):
    beam = Beam(id= -1, axis = None)

    all_pts = np.array(obb.get_box_points())
    d = [geometry.getDistance(all_pts[0], p) for p in all_pts]
    idx = np.argsort(d)
    v1 = all_pts[idx[0:4]] # assumed as start
    v2 = all_pts[idx[4:]] # assumed as end

    v1_center = np.mean(v1, axis=0)
    v2_center = np.mean(v2, axis=0)

    uvec = geometry.getUnitVector(v2_center - v1_center)
    ouvec = geometry.orientNormalVector(v1_center, uvec, orientation, True)

    if np.array_equal(uvec, ouvec):
        beam.unit_vector = uvec
        beam.axis = np.array((v1_center, v2_center))
        beam.vertices = np.array([*v1, *v2])
        beam.width = obb.extent[2]
        beam.height = obb.extent[1]
        beam.length = obb.extent[0]
        beam.comment = comment
        beam.obb = obb
    else:
        beam.unit_vector = ouvec
        beam.axis = np.array((v2_center, v1_center))
        beam.vertices = np.array([*v2, *v1])
        beam.width = obb.extent[2]
        beam.height = obb.extent[1]
        beam.length = obb.extent[0]
        beam.comment = comment
        beam.obb = obb
    return beam

def getBeamById(beams, id):
    for b in beams:
        if id == b.id:
            return b
            break

def getOBBofBeamList(beams, minimal_box = False):
    all_ver = [ b.vertices for b in beams]
    all_vertices = np.vstack(all_ver)

    group_pts = o3d.geometry.PointCloud()
    group_pts.points = o3d.utility.Vector3dVector(all_vertices)    
    if minimal_box:
        group_obb = group_pts.get_minimal_oriented_bounding_box(robust=True)
    else:
        group_obb = group_pts.get_oriented_bounding_box()
    return all_vertices, group_obb

def getConvexHullofBeamList(beams, only_axes=False):

    if only_axes:
        all_ver = [ b.axis for b in beams]
    else:
        all_ver = [ b.vertices for b in beams]
    all_vertices = np.vstack(all_ver)

    group_pts = o3d.geometry.PointCloud()
    group_pts.points = o3d.utility.Vector3dVector(all_vertices)
    hull, _ = group_pts.compute_convex_hull()
    return hull

def getMergeMatches(beams):
    scene = o3d.t.geometry.RaycastingScene()
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
    dist_top = ans_top['geometry_ids'].numpy()
    matches1 =  np.array([np.sort((i,val)) for i,val in enumerate(dist_top) if val < len(beams)])

    ans_bot = scene.cast_rays(bot_rays)
    dist_bot = ans_bot['geometry_ids'].numpy()
    matches2 =  np.array([np.sort((i,val)) for i,val in enumerate(dist_bot) if val < len(beams)])

    matches = np.unique(np.vstack((matches1,matches2)), axis = 0)

    final_matches = []
    for m in matches:
        if not len(final_matches):
            final_matches.append(m)
        else:
            for i,mtc in enumerate(final_matches):
                inter = np.intersect1d(mtc, m)
                if len(inter):
                    final_matches[i] = np.unique(np.vstack((mtc,m)))
                else:final_matches.append(m)

    #Phase2 : Intersected mergable beams
    beam_centers = [np.mean(b.axis, axis=0) for b in beams]
    beam_centers_pcd = o3d.geometry.PointCloud()
    beam_centers_pcd.points = o3d.utility.Vector3dVector(beam_centers)

    intersecting_matches = []
    for i,b in enumerate(beams):
        idx = b.obb.get_point_indices_within_bounding_box(beam_centers_pcd.points)
        if len(idx) > 1:
            intersecting_matches.append(idx)
    intersecting_matches = np.unique(np.array(intersecting_matches))

    if len(intersecting_matches):
        if intersecting_matches[0].dtype == np.dtype('int32'):
            intersecting_matches = [intersecting_matches]

    for m in intersecting_matches:
        m_new = True
        for i,f in enumerate(final_matches):
            inter = np.intersect1d(f, m)
            if len(inter):
                    final_matches[i] = np.unique(np.vstack((f,m)))
                    m_new = False
        if m_new:
            final_matches.append(m)

    if not len(final_matches) and len(intersecting_matches):
        final_matches = intersecting_matches
    return final_matches

def mergeBeams(beams, pcd):
    all_vert, merged_obb = getOBBofBeamList(beams, minimal_box = True)
    result_obb = template.getRegisteredOBB(merged_obb, pcd, 0.2, 50)
    #o3d.visualization.draw([pcd, merged_obb, result_obb])   
    c1 = merged_obb.center
    c2 = result_obb.center
    if geometry.getDistance(c1, c2) < 0.05:
        return result_obb
    else:
        return merged_obb

def kmeansClusterBeams(beams, n_clusters, vis = False):
        beam_vecs = np.array([b.unit_vector for b in beams])

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans = kmeans.fit(beam_vecs)
        labels = kmeans.predict(beam_vecs)
        
        cluster_centers_vecs = [geometry.getUnitVector(c) for c in kmeans.cluster_centers_]
        
        angles = [geometry.getAngleBetweenVectors(vec, cluster_centers_vecs[labels[i]]) for i, vec in enumerate(beam_vecs)]
        
        beam_ids_inlier = np.where(np.array(angles) <5)[0]
        beam_ids_outlier = np.where(np.array(angles) >=5)[0]

        affected_clusters = np.unique(labels[beam_ids_outlier])
        #recompute affected cluster centers

        for i in affected_clusters:
            ids_in_cluster = np.where(labels == i)[0]
            ids_in_cluster_in = [idx for idx in ids_in_cluster if idx not in beam_ids_outlier]
            new_cluster = beam_vecs[ids_in_cluster_in]

            new_cluster_center = np.mean(new_cluster, axis =0)
            new_cluster_center = geometry.getUnitVector(new_cluster_center)
            cluster_centers_vecs[i] = new_cluster_center

        #Loop over outliers to involve an existing cluster / create new one
        for idx in beam_ids_outlier:
            beam_vec = beam_vecs[idx]

            angles_to_clusters = np.array([geometry.getAngleBetweenVectors(c,beam_vec) for c in cluster_centers_vecs])

            closest_cluster_id = np.argmin(angles_to_clusters)
            
            if angles_to_clusters[closest_cluster_id] < 5.:
                
                #recompute cluster center
                sum_new = cluster_centers_vecs[closest_cluster_id] * len(np.where(labels == closest_cluster_id)[0]) + beam_vec
                mean_new = sum_new / (len(np.where(labels == closest_cluster_id)[0]) + 1)
                center_new = geometry.getUnitVector(mean_new)

                labels[idx] = closest_cluster_id
                cluster_centers_vecs[closest_cluster_id] = center_new

            else:
                labels[idx] = len(cluster_centers_vecs)
                cluster_centers_vecs.append(beam_vec)
                
        #TODO: group merge strategy should come here:
        '''
        angle_dist_map = []
        for i,vec in enumerate(cluster_centers_vecs):
            angles = []
            for j, vec2 in enumerate(cluster_centers_vecs):
                angles.append(geometry.getAngleBetweenVectors(vec,vec2))

            angle_dist_map.append(angles)
        '''

        colors = [[1.0, 1.0, 0], # Yellow
            [0.0, 1.0, 1.0], # Turquoise
            [0.0, 1.0, 0.0], # Green
            [1.0, 0.0, 0.0], # Red
            [0.0, 0.0, 1.0], # Blue
            [0.5, 0.0, 1.0], # color6
            [0.1, 0.5, 0.0] # color7
            ]
        import random
        for i in range(500):
            random.uniform(0, 1)
            colors.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

        clustered_obbs = [b.setOBB() for b in beams]
        for i, b in enumerate(clustered_obbs):
                b.color = colors[labels[i]]

        if vis:
            o3d.visualization.draw(clustered_obbs)

        for i,b in enumerate(beams):
            b.cluster_id = labels[i] + 1 # cluster id on database

        return beams, cluster_centers_vecs

def meanShiftClusterBeams(beams, vis = False):
        beam_vecs = np.array([b.unit_vector for b in beams])
        #kmeans = KMeans(n_clusters=n_clusters)
        #kmeans = kmeans.fit(beam_vecs)
        #labels = kmeans.predict(beam_vecs)

        ms = MeanShift(bandwidth=0.04, bin_seeding=True)
        ms.fit(beam_vecs)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_  
        cluster_centers_vecs = [geometry.getUnitVector(c) for c in cluster_centers]
        
              
        colors = [[1.0, 1.0, 0], # Yellow
            [0.0, 1.0, 1.0], # Turquoise
            [0.0, 1.0, 0.0], # Green
            [1.0, 0.0, 0.0], # Red
            [0.0, 0.0, 1.0], # Blue
            [0.5, 0.0, 1.0], # color6
            [0.1, 0.5, 0.0] # color7
            ]
        import random
        for i in range(500):
            random.uniform(0, 1)
            colors.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

        clustered_obbs = [b.setOBB() for b in beams]
        for i, b in enumerate(clustered_obbs):
                b.color = colors[labels[i]]

        if vis:
            o3d.visualization.draw(clustered_obbs)

        for i,b in enumerate(beams):
            b.cluster_id = labels[i] + 1 # cluster id on database

        return beams, cluster_centers_vecs