import numpy as np
import math
from operator import itemgetter

import ezdxf
from ezdxf import bbox
from ezdxf.addons import r12writer

from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon

import toolBox.geometry as geometry
import toolBox.exchange as exchange

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KDTree
from enum import Enum

import open3d as o3d


class BEAM_SIDE(Enum):
    BASE = 0
    RIGHT = 1
    LEFT = 2
    OPPOSITE = 3

class BeamGroup():

    def __init__(self, beam_axes, beams = None, roof_plane = None):
        self.id = None
        self.name = ""
        self.beam_axes = [] #Beam axes are assumed as oriented to (+, +, +)
        self.beam_vertices = []
        self.beam_cuboids = []

        self.beam_axes_2D = []
        self.beam_equations_2D = []
        self.optimal_reference_lines = None # top, center, bottom
        self.optimal_plane = None
        self.optimal_slope = None
        self.ref_iterval = None
        self.roof_plane = roof_plane
        self.optimal_uvec = None
        self.coordinates_xy = True
        self.template_obb = None
        self.avg_width = None
        self.avg_height = None
        self.avg_length = None

        #Analsyis result of 3 sub-clusters
        self.reliable_beams = [] # list of indices
        self.merge_beams = [] # list of list indices
        self.create_beams_eq = [] # list of line equations
        self.other_beams = [] # list of indices
        self.ignore_beams = [] # interior offset beams


        self.beam_axes = beam_axes
        self.beams = beams
        self.getOptimalPlane()
        self.projectBeams2D()
        self.getBeamEquations2D()
        self.getOptimalAxes()
        self.getBeamIntervals()
        self.getOptimalUvec()

    def getOptimalPlane(self):
        for b in self.beam_axes:
            self.beam_vertices.append(np.array(b.dxf.start.xyz))
            self.beam_vertices.append(np.array(b.dxf.end.xyz))
        
        plane,_,_ = geometry.getPlaneLS(np.array(self.beam_vertices))
        self.optimal_plane = plane[:4]

    def projectBeams2D(self):
        for b in self.beam_axes:
            s = geometry.project3DPointToPlane2D(np.array(b.dxf.start.xyz),self.optimal_plane)
            e = geometry.project3DPointToPlane2D(np.array(b.dxf.end.xyz), self.optimal_plane)
            self.beam_axes_2D.append((s, e))
        self.beam_axes_2D = np.array(self.beam_axes_2D)

        x_start = self.beam_axes_2D[:,0][:,0]
        y_start = self.beam_axes_2D[:,0][:,1]
        x_end = self.beam_axes_2D[:,1][:,0]
        y_end = self.beam_axes_2D[:,1][:,1]

        x_dev = np.mean(np.abs(x_end - x_start))
        y_dev = np.mean(np.abs(y_end - y_start))

        if x_dev < 1. and y_dev > 1:
            #Swith xy coords to yx coords temporarily
            self.coordinates_xy = False
            new_starts = np.vstack((y_start,x_start))
            new_ends= np.vstack((y_end,x_end))
            new_coordinates= np.vstack((new_starts,new_ends)).transpose()
            self.beam_axes_2D = np.array([[[c[0],c[1]],[c[2], c[3]]] for c in new_coordinates])
  
    def getBeamEquations2D(self):
        self.beam_equations_2D = [geometry.getLineEquation2D(b[0], b[1]) for b in self.beam_axes_2D]
        self.beam_equations_2D = np.array(self.beam_equations_2D)
       
    def getOptimalAxes(self):
        #This function identifies dominant line equation of all axes 2d
        #Orthogonal line is also defined for sequential search
        slopes = self.beam_equations_2D[:,0]
        kmeans = KMeans(n_clusters=3)
        kmeans = kmeans.fit(slopes.reshape(-1,1))
        self.optimal_slope = kmeans.cluster_centers_[0]

        if self.optimal_slope != 0:
            m2 = -1./self.optimal_slope # perpendicular slope
            
            v_bottom = self.beam_axes_2D[:,0] #bottom vertices
            v_top = self.beam_axes_2D[:,1] #top vertices
            
            #Define a central point as reference
            x_mean = np.mean((np.mean(v_top[:,0]), np.mean(v_bottom[:,0])))
            y_mean = np.mean((np.mean(v_top[:,1]), np.mean(v_bottom[:,1])))

            #Project vertices to Vertical line (before distance comparison)
            b_prj = y_mean - self.optimal_slope * x_mean
            line_prj = (self.optimal_slope, b_prj)

            v_bottom_prj = [geometry.projectPointToLine2D(line_prj,v) for v in v_bottom]
            v_top_prj = [geometry.projectPointToLine2D(line_prj,v) for v in v_top]

            # Any (bottom, top) match is giving the upwards vector direction for the signed distance
            ref_pt_bot = v_bottom_prj[0]
            ref_pt_top = v_top_prj[0]
            
            #Central line
            b_center = y_mean - m2 * x_mean
            line_center = (m2[0], b_center[0])

            #Top line
            #d_top_center = [ geometry.getPoint2LineDistance(line_center,v) for v in v_top_prj]
            d_top_center = [ geometry.getSignedDistance(ref_pt_bot,ref_pt_top, v) for v in v_top_prj]
            d_top_center = np.array(d_top_center)
            idx_top = np.argmax(d_top_center) # Top line reference vertex has max distance to the center          
            b_top = v_top[idx_top][1] - m2 * v_top[idx_top][0]
            line_top = (m2[0], b_top[0])

            #Bottom line
            #d_bottom_center = [ geometry.getPoint2LineDistance(line_center,v) for v in v_bottom_prj]
            d_bottom_center = [ geometry.getSignedDistance(ref_pt_bot,ref_pt_top, v) for v in v_bottom_prj]
            d_bottom_center = np.array(d_bottom_center)
            idx_bottom = np.argmin(d_bottom_center) # Bottom line reference vertex has max distance to the center          
            b_bottom = v_bottom[idx_bottom][1] - m2 * v_bottom[idx_bottom][0]
            line_bottom = (m2[0], b_bottom[0])
        #else: TODO

        #Test cases
        """
        x_min = np.min((np.min(v_top[:,0]), np.min(v_bottom[:,0])))
        x_max = np.max((np.max(v_top[:,0]), np.max(v_bottom[:,0])))

        y_on_top_min = line_top[0] * x_min + line_top[1]
        y_on_top_max = line_top[0] * x_max + line_top[1]

        y_on_center_min = line_center[0] * x_min + line_center[1]
        y_on_center_max = line_center[0] * x_max + line_center[1]

        y_on_bottom_min = line_bottom[0] * x_min + line_bottom[1]
        y_on_bottom_max = line_bottom[0] * x_max + line_bottom[1]


        l1 = LineString([(x_min, y_on_top_min), (x_max, y_on_top_max)])
        l2 = LineString([(x_min, y_on_center_min), (x_max, y_on_center_max)])
        l3 = LineString([(x_min, y_on_bottom_min), (x_max, y_on_bottom_max)])
        """
        self.optimal_reference_lines = (line_top, line_center, line_bottom)

    def getBeamIntervals(self):
        # Get Beam Lines to Central reference line intersetions
        pts = np.array([ geometry.getLineIntersection(l1, self.optimal_reference_lines[1]) for l1 in self.beam_equations_2D])

        #Define the reference point
        ref_point = pts[np.argmin(pts[:,0])] # min x refers to the beginning of search

        tree = KDTree(pts, leaf_size=2,  metric='euclidean')              
        distances, id = tree.query(ref_point.reshape(-1,2), k=len(pts))    
        
        intervals = np.array([ d - distances[0][i-1]  for i,d in enumerate(distances[0]) if i>0]) 
        valid_intervals = intervals[intervals>0.4]#Intervals are expected to be around 70-90cm
        kmeans = KMeans(n_clusters=4)# 5 before
        kmeans = kmeans.fit(valid_intervals.reshape(-1,1))
        km_unique, km_idx, km_counts = np.unique(kmeans.labels_,return_index=True,return_counts=True)

        ref_interval = kmeans.cluster_centers_[np.argmax(km_counts)]
        ref_std = np.std(valid_intervals[kmeans.labels_ == np.argmax(km_counts)])
        self.ref_iterval = (ref_interval, ref_std)

        merge_list = [] # list of id list
        create_list = [] # suggested new line equations
        reliable_list = [] # ids of reliable beams (no change)
        other_list = []
        ignore_list = [] # internal offset beams

        for i, d in enumerate(intervals):
            

            if i ==0:
                p1 = pts[id[0][0]]
                p2 = pts[id[0][1]]
            
                u = (p2 - p1)
                u = geometry.getUnitVector(u)
            
                p_back = p1 - ref_interval * u
                b = p_back[1] - self.optimal_slope * p_back[0]
                create_list.append((self.optimal_slope[0], b[0]))
                            
            if d < ref_interval + 2 * ref_std and d> ref_interval - 2 * ref_std:
                # these 2 beams has reliable distance to each other
                reliable_list.append(id[0][i])
                reliable_list.append(id[0][i+1])        
            
            elif d < 0.12: #TODO this threshold should come from average beam dimensions!!!                
                # additional check for interior offset beams
                if self.roof_plane is not None:
                    c_0 = np.mean((self.beam_axes[id[0][i]].dxf.start,self.beam_axes[id[0][i]].dxf.end), axis = 0) #center of first beam
                    c_1 = np.mean((self.beam_axes[id[0][i+1]].dxf.start,self.beam_axes[id[0][i+1]].dxf.end), axis = 0) #center of second beam

                    '''
                    n_0 = geometry.getUnitVector((self.beam_axes[id[0][i]].dxf.end - self.beam_axes[id[0][i]].dxf.start))
                    p_0 = self.beam_axes[id[0][i]].dxf.start
                    p_1 = self.beam_axes[id[0][i+1]].dxf.start
                    pl_d = - np.dot(n_0, p_0)
                    plane_gr0 = np.append(n_0, pl_d)
                    p_1_prj = geometry.project3DPointToPlane(p_1, plane_gr0)
                    dist = geometry.getDistance(p_0, p_1_prj)
                     '''
                    d_0 = np.abs(geometry.getPoint2PlaneDistance(c_0,self.roof_plane))
                    d_1 = np.abs(geometry.getPoint2PlaneDistance(c_1,self.roof_plane))
                    d_0_1 = np.abs(d_0 - d_1)
                   
                    if d_0_1 < 0.1:
                        merge_list.append((id[0][i], id[0][i+1]))
                    else:
                        out_id = np.argmax((d_0, d_1))
                        ignore_list.append(id[0][i+out_id])
                else:
                    #these 2 beams are closed to be merged
                    merge_list.append((id[0][i], id[0][i+1])) #TODO merge_list can have more than 2 ids
                                  
            elif d > ref_interval + ref_std:
                #Between these 2 beams there could be some beams
                nr_parts= round(d/ ref_interval[0])
                p1 = pts[id[0][i]]
                p2 = pts[id[0][i+1]]
                new_pts =[]
                for x in range(1, nr_parts):
                    new_pts.append(p1 + (x/nr_parts) * (p2 - p1))
            
                for p in new_pts:
                    b = p[1] - self.optimal_slope * p[0]
                    create_list.append((self.optimal_slope[0], b[0]))
            else:
                other_list.append(id[0][i])
                other_list.append(id[0][i+1])
                            
            if i == len(intervals) -1:
                p1 = pts[id[0][i]]
                p2 = pts[id[0][i+1]]
                
                u = (p2 - p1)
                u = geometry.getUnitVector(u)
                
                p_fw = p2 + ref_interval * u
                b = p_fw[1] - self.optimal_slope * p_fw[0]
                create_list.append((self.optimal_slope[0], b[0]))
            
        # Correct the merge list
        if len(merge_list) > 1:

            merge_beams_aggr = []
            merge_checked = np.zeros(len(merge_list))
            for i, m in enumerate(merge_list):
                if merge_checked[i] == 0:
                    test_couple = np.array(m)
                    merge_checked[i] = 1
                    for j, c in enumerate(merge_checked):
                        if c == 0:
                            candidate = np.array(merge_list[j])
                            intersection = np.intersect1d(test_couple, candidate)
                            if len(intersection) > 0:
                                test_couple = np.unique(np.hstack((test_couple, candidate)))
                                merge_checked[j] = 1
                    merge_beams_aggr.append(test_couple)

        else:
            merge_beams_aggr = merge_list

        # Correct the reliable list
        reliable_list_aggr = np.array([i for i in range (len(self.beam_axes)) ])

        if len(ignore_list) >0:
            reliable_list_aggr = np.setdiff1d(reliable_list_aggr, np.array(ignore_list))
        if len(merge_beams_aggr) >0:
            merge_beams_all = np.unique(np.hstack(merge_beams_aggr))
            reliable_list_aggr = np.setdiff1d(reliable_list_aggr, merge_beams_all)

        self.reliable_beams = np.unique(reliable_list_aggr)
        self.create_beams_eq = create_list
        self.merge_beams = merge_beams_aggr
        #self.other_beams = other_list
        self.ignore_beams = ignore_list

    def getMergableBeams(self, full_extend = False):
        merged_lines = []
        merged_lines_eq = []
        for ids in self.merge_beams:
            #eq1 = self.beam_equations_2D[ids[0]]
            #eq2 = self.beam_equations_2D[ids[1]]

            eqs = [ self.beam_equations_2D[id] for id in ids]

            all_vertices = []
            for id in ids:
                all_vertices.append(self.beam_axes_2D[id][0])
                all_vertices.append(self.beam_axes_2D[id][1])
            all_vertices = np.array(all_vertices)

            #a = self.beam_axes_2D[ids[0]][0]
            #b = self.beam_axes_2D[ids[0]][1]
            #c = self.beam_axes_2D[ids[1]][0]
            #d = self.beam_axes_2D[ids[1]][1]
            #all_vertices = np.array([a, b, c, d])

            d_to_top = np.array([ geometry.getPoint2LineDistance(self.optimal_reference_lines[0],v) for v in all_vertices])
            idx_top = np.argmin(d_to_top)     
            v_top = all_vertices[idx_top]
            idx_bottom = np.argmax(d_to_top)     
            v_bottom = all_vertices[idx_bottom]

            eq = np.mean(eqs, axis=0)
            merged_lines_eq.append(eq)

            if full_extend:
                v_top_new = geometry.getLineIntersection(eq, self.optimal_reference_lines[0])
                v_bottom_new = geometry.getLineIntersection(eq, self.optimal_reference_lines[2])             
            else:
                v_top_new = np.array(geometry.projectPointToLine2D(eq, v_top)) #projectPointToLine2D
                v_bottom_new = np.array(geometry.projectPointToLine2D(eq, v_bottom))
            
            #if self.coordinates_xy is True:
            best_line = self.getBestSearcLine2D((v_bottom_new, v_top_new))
            merged_lines.append(best_line)
            #merged_lines.append((v_bottom_new, v_top_new))
            #else:
            #    merged_lines.append((np.array((v_bottom_new[1], v_bottom_new[0])), np.array((v_top_new[1], v_top_new[0]))))

        mls = MultiLineString(merged_lines)
        return merged_lines

    def getAdditionalBeams(self, align_to_neighbours = False):
        create_lines = []
        ref_planes = []
        if align_to_neighbours:
            #This case is to update reference plane. 
            #Local plane defined with 2 reliable neighbours instead of global one.
                        
            for x in self.create_beams_eq:
                diff = self.beam_equations_2D[self.reliable_beams] - x
                dist = diff[:,0] * diff[:,0] + diff[:,1] * diff[:,1]
                neighbours = np.argsort(dist)[:4]
                
                ref_vertices = []
                for n in neighbours:
                    b = self.beam_axes[n]
                    ref_vertices.append(np.array(b.dxf.start.xyz))
                    ref_vertices.append(np.array(b.dxf.end.xyz))

                
                new_plane,_,_ = geometry.getPlaneLS(np.array(ref_vertices))
                ref_plane = new_plane[:4]
                
                #New plane alignment: Aling to global ref plane
                #ref_nrm = geometry.getBisectorVector(ref_plane[:3], self.optimal_plane[:3])
                #ref_plane = np.append(ref_nrm, np.mean((ref_plane[3],self.optimal_plane[3])))

                p_bottom = geometry.getLineIntersection(x, self.optimal_reference_lines[2])
                p_top = geometry.getLineIntersection(x, self.optimal_reference_lines[0])

                #Project back to first ref
                p_bottom_3d = geometry.reproject2DPointToPlane3D(p_bottom, self.optimal_plane)
                p_top_3d = geometry.reproject2DPointToPlane3D(p_top, self.optimal_plane)
                #Re-project to new reference plane
                p_bottom = geometry.project3DPointToPlane2D(p_bottom_3d,ref_plane)
                p_top = geometry.project3DPointToPlane2D(p_top_3d, ref_plane)
                #if self.coordinates_xy is True:

                #best_line = self.getBestSearcLine2D((p_bottom, p_top))
                #create_lines.append(best_line)
                create_lines.append((p_bottom, p_top))
                #              
                #else:
                #    create_lines.append((np.array((p_bottom[1],p_bottom[0])), np.array((p_top[1], p_top[0]))))
                ref_planes.append(ref_plane)
        else:
            ref_plane = self.optimal_plane
            for x in self.create_beams_eq:
                p_bottom = geometry.getLineIntersection(x, self.optimal_reference_lines[2])
                p_top = geometry.getLineIntersection(x, self.optimal_reference_lines[0])
                #if self.coordinates_xy is True:
                create_lines.append((p_bottom, p_top))
                #else:
                #    create_lines.append((np.array((p_bottom[1],p_bottom[0])), np.array((p_top[1], p_top[0]))))
                ref_planes.append(ref_plane)
        return create_lines, ref_planes

    def getExtendableBeams(self, threshold = 0.1):
        #candidate_idx = np.hstack((self.reliable_beams, self.other_beams))
        #candidate_idx = [int(id) for id in candidate_idx if id not in np.array(self.merge_beams).flatten()]
        candidate_idx = self.reliable_beams

        extend_lines = []
        extend_lines_eq = []

        for i in candidate_idx:
            eq = self.beam_equations_2D[i]
            axis_2D = self.beam_axes_2D[i]

            d_bottom = geometry.getPoint2LineDistance(self.optimal_reference_lines[2],axis_2D[0])
            d_top = geometry.getPoint2LineDistance(self.optimal_reference_lines[0],axis_2D[1])

            if d_top > threshold or d_bottom > threshold:
                if d_top > threshold:
                    v_top_new = geometry.getLineIntersection(eq, self.optimal_reference_lines[0])
                else:
                    v_top_new = axis_2D[1]
                if d_bottom > threshold:
                    v_bottom_new = geometry.getLineIntersection(eq, self.optimal_reference_lines[2])
                else:
                    v_bottom_new = axis_2D[0]

                #if self.coordinates_xy is True:
                best_line = self.getBestSearcLine2D((v_bottom_new, v_top_new))
                extend_lines.append(best_line)
                #extend_lines.append((v_bottom_new, v_top_new))
                #else:
                #    extend_lines.append((np.array((v_bottom_new[1], v_bottom_new[0])), np.array((v_top_new[1], v_top_new[0]))))
            else:
                extend_lines.append(None)
        return extend_lines

    def getAllAExtendableBeams(self, threshold = 0.1):
        # Extend all beams to the guide lines (top&bottom) : no point cloud support check!

        extend_lines = []
        extend_lines_eq = []
        no_change = []

        for i, axis_2D in enumerate(self.beam_axes_2D):
            eq = self.beam_equations_2D[i]
            #axis_2D = self.beam_axes_2D[i]

            d_bottom = geometry.getPoint2LineDistance(self.optimal_reference_lines[2],axis_2D[0])
            d_top = geometry.getPoint2LineDistance(self.optimal_reference_lines[0],axis_2D[1])

            if d_top > threshold or d_bottom > threshold:
                if d_top > threshold:
                    v_top_new = geometry.getLineIntersection(eq, self.optimal_reference_lines[0])
                else:
                    v_top_new = axis_2D[1]
                if d_bottom > threshold:
                    v_bottom_new = geometry.getLineIntersection(eq, self.optimal_reference_lines[2])
                else:
                    v_bottom_new = axis_2D[0]
                extend_lines.append((v_bottom_new, v_top_new))
            else:
                extend_lines.append(self.beam_axes_2D[i])
                no_change.append(i)

            
        return extend_lines, no_change

    def getFullExtendBeams(self,extend_lines):

        #TODO coordinate_xy is not considered yet!

        beams_obb = []
        # 2D lines from getAllAExtendableBeams
        for i, line_2D in enumerate(extend_lines):
            ### line_2D : tuple of (bottom_point, end_point)
        
            #1 Reproject line to 3D space
            p_bottom_3d = geometry.reproject2DPointToPlane3D(line_2D[0], self.optimal_plane)
            p_top_3d = geometry.reproject2DPointToPlane3D(line_2D[1], self.optimal_plane)
            
            #2 Define a plane that original beam-axis pass through (move the optimal plane)
            point_on_line = np.array(self.beam_axes[i].dxf.start.xyz)
            nrm = self.optimal_plane[:3]
            d = - np.dot(nrm, point_on_line)
            plane_new = np.append(nrm,d)



            plane,_,_ = geometry.getPlaneLS(np.array(self.beam_vertices))
            self.optimal_plane = plane[:4]
        
            #3 Re-project line points onto defined plane
            p_bottom = geometry.project3DPointToPlane(p_bottom_3d, plane_new)
            p_top = geometry.project3DPointToPlane(p_top_3d, plane_new)

            #4 Define planes at top&bottom of the cuboid
            nrm_beam = np.array(geometry.getUnitVector(p_top - p_bottom))
            d1 = - np.dot(nrm_beam, p_bottom)
            plane_bottom = np.append(nrm_beam,d1)
            d2 = - np.dot(nrm_beam, p_top)
            plane_top = np.append(nrm_beam,d2)

            #5 Project cuboid vertices to the extend (new plane refs)
            faces = exchange.getCuboidFaces(self.beams[i])
            begin_pts = faces[0] # Top points are used as reference // bottom points are faces[5] and both have to be equal on the projection

            # Projected box points to the extend references:
            bottom_prj = [geometry.project3DPointToPlane(p, plane_bottom) for p in begin_pts]
            top_prj = [geometry.project3DPointToPlane(p, plane_top) for p in begin_pts]

            all_pts = [*bottom_prj, *top_prj]

            obb = o3d.geometry.OrientedBoundingBox()
            obb =  obb.create_from_points(points=o3d.utility.Vector3dVector(all_pts))
            beams_obb.append(obb)
        return beams_obb
    
    def write2DReferences(self, path):
        
        merged_lines = self.getMergableBeams()
        create_lines,rp = self.getAdditionalBeams()
        with r12writer(path) as dxf:

            for i, line in enumerate(self.beam_axes_2D[ self.reliable_beams]):         
                dxf.add_line(line[0],line[1], color=1, layer = "Reliable")

            for i, line in enumerate(merged_lines):
                dxf.add_line(line[0],line[1], color=2, layer = "Merged")
           
            for i, line in enumerate(create_lines):
                dxf.add_line(line[0],line[1], color=5, layer = "Create")

    def getOptimalUvec(self):
        beam_uvecs = np.array([geometry.getUnitVector(ax.dxf.end - ax.dxf.start)  for i, ax in enumerate(self.beam_axes) if i not in self.ignore_beams])
        beam_uvecs_avg = np.average(beam_uvecs, axis = 0)
        beam_avg = geometry.getUnitVector(beam_uvecs_avg)
        self.optimal_uvec = beam_avg

    def getBestSearcLine2D(self, line):
        mp_chull = MultiPoint(np.vstack((self.beam_axes_2D[:,0], self.beam_axes_2D[:,1]))).convex_hull #TODO optimize later
        mp_chull_buff = mp_chull.buffer(1.)
        mp_rect = mp_chull_buff.minimum_rotated_rectangle

        #line is in (start,end) order

        uvec = geometry.getUnitVector(line[1] - line[0])

        new_start = line[0] - 1000 * uvec
        new_end = line[1] + 1000 * uvec

        #old_len = geometry.getDistance(line[0], linee[1])
        #new_len = geometry.getDistance(new_start, new_end)

        old_line = LineString(line)
        new_line = LineString((new_start,new_end))

        old_inter = old_line.intersection(mp_rect)
        new_inter = new_line.intersection(mp_rect)

        if new_inter.intersects(mp_rect):
            l_candidate = np.array(new_inter.xy).transpose()
            d1 = geometry.getDistance(line[0], l_candidate[0]) #compare to start
            d2 = geometry.getDistance(line[0], l_candidate[1])
            if d1 > d2:
                l_candidate = np.array((l_candidate[1] , l_candidate[0]))

        else:
            l_candidate = np.array(line)
        return l_candidate

def kmeansBeams(beamAxes, beams):
    orientationPos = np.array([1000.0, 1000.0, 1000.0])

    #1 Beams to oriented normal vectors
    uvecs, ouvecs = exchange.lines2UVecs(beamAxes, orientationPos)
  


    db = DBSCAN(eps=0.03, min_samples=10).fit(ouvecs* 10)
    idx = np.where(db.labels_ == 0)
    cluster = ouvecs[idx]
    beams_cluster = list(itemgetter(*idx[0])(beams))
    beam_cluster_list = []
 
    for b in beams_cluster:
        beam_cluster_list.append(exchange.getBeamOBB(b))

    
    o3d.visualization.draw(beam_cluster_list)


    #2 Cluster Beams-Axes
    kmeans = KMeans(n_clusters=4)
    kmeans = kmeans.fit(ouvecs)
    labels = kmeans.predict(ouvecs)

    for i in range(kmeans.n_clusters):
        idx = np.where(labels == i)
        cluster = ouvecs[idx]
        beams_cluster = list(itemgetter(*idx[0])(beams))

        angles = geometry.getAngleBetweenVectors(cluster, kmeans.cluster_centers_[i])
        angles2 = geometry.getAngleBetweenVectors(cluster, cluster[np.argmin(angles)])



        index_list1 = np.where(angles < 2)
        index_list2 = np.where(angles2 < 1)

        res_beams1 = list(itemgetter(*index_list1[0])(beams_cluster))
        res_beams2 = list(itemgetter(*index_list2[0])(beams_cluster))

        kmeans2 = KMeans(n_clusters=2)
        kmeans2 = kmeans2.fit(cluster)
        labels2 = kmeans2.predict(cluster)
        idx2 = np.where(labels2 == 0)
        cluster2 = ouvecs[idx2]
        beams_cluster2 = list(itemgetter(*idx2[0])(beams_cluster))


        beam_cluster_list = []
        beam_cluster_list2 = []
        res_beams1_list = []
        res_beams2_list =[]

        for b in beams_cluster:
            beam_cluster_list.append(exchange.getBeamOBB(b))

        for b in res_beams1:
            res_beams1_list.append(exchange.getBeamOBB(b))

        for b in res_beams2:
            res_beams2_list.append(exchange.getBeamOBB(b))

        for b in beams_cluster2:
            beam_cluster_list2.append(exchange.getBeamOBB(b))


        
        o3d.visualization.draw(beam_cluster_list)
        o3d.visualization.draw(beam_cluster_list2)
        o3d.visualization.draw(res_beams1_list)
        o3d.visualization.draw(res_beams2_list)

        foo = 12
