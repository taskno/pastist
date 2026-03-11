import itertools
import numpy as np
from dxfwrite import DXFEngine as dxf
import external.pbs_enums

class Beam:
    newid = itertools.count() #newid = itertools.count().next Clarify new code!!
    def __init__(self, beam_points, max_iter=10, debug=False):
        self.id = next(self.newid)
        self.beam_points = beam_points        
        self.elongation = [ 0,   # elongation on side of the base point
                            0 ]  # elongation of the end point
        self.cross_section_class = None

    def get_beam_axis(self, elongation_mode=external.pbs_enums.ELONGATION_MODE.ELONGATE):
        # base point is in the origin of the beam-coordinate system (x_b, y_b, z_b) = (0, 0, 0)
        if self.basepoint is None or self.R is None or self.dimensions is None:
            return [(None, None), (None, None), (None, None)]

        if elongation_mode == external.pbs_enums.ELONGATION_MODE.CREATE_NEW:
            if self.elongation[0] > 0:
                p_start1 = self.basepoint + \
                    np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0] / 2 + \
                    np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1] / 2
                p_end1 = self.basepoint + \
                    np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0] / 2 + \
                    np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1] / 2 + \
                    np.dot(self.R, np.array([[0], [0], [1]])) * self.elongation[0]

                elon1 = (p_start1.flatten(), p_end1.flatten())
            else:
                elon1 = (None, None)

            p_start2 = self.basepoint + \
                    np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0] / 2 + \
                    np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1] / 2 + \
                    np.dot(self.R, np.array([[0], [0], [1]])) * self.elongation[0]
            p_end2 = self.basepoint + \
                    np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0] / 2 + \
                    np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1] / 2 + \
                    np.dot(self.R, np.array([[0], [0], [1]])) * (self.dimensions[2] - self.elongation[1])

            main_beam = (p_start2.flatten(), p_end2.flatten())

            if self.elongation[1] > 0:
                p_start3 = self.basepoint + \
                        np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0] / 2 + \
                        np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1] / 2 + \
                        np.dot(self.R, np.array([[0], [0], [1]])) * (self.dimensions[2] - self.elongation[1])
                p_end3 = self.basepoint + \
                        np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0] / 2 + \
                        np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1] / 2 + \
                        np.dot(self.R, np.array([[0], [0], [1]])) * self.dimensions[2]

                elon2 = (p_start3.flatten(), p_end3.flatten())
            else:
                elon2 = (None, None)

            return [elon1, main_beam, elon2]

        p_start = self.basepoint + \
                np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0]/2 + \
                np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1]/2
        p_end = self.basepoint + \
                np.dot(self.R, np.array([[1], [0], [0]])) * self.dimensions[0]/2 + \
                np.dot(self.R, np.array([[0], [1], [0]])) * self.dimensions[1]/2 + \
                np.dot(self.R, np.array([[0], [0], [1]])) * self.dimensions[2]

        return [(None, None), (p_start.flatten(), p_end.flatten()), (None, None)]

    def get_dxfwrite_cuboid(self, color=None, layer=None):
        def scale(point):
            tmp_d = self.R * self.dimensions * np.array(point)
            # scaled point = bp + a * r1 * point(0) + b * r2 * point(1) + e * r3 * point(2)
            return self.basepoint + np.sum(tmp_d, 1)[np.newaxis].T

        if self.basepoint is None or self.R is None or self.dimensions is None:
            return None

        pface = dxf.polyface(layer=layer)
        # cube corner points
        p1 = scale([0, 0, 0])
        p2 = scale([0, 0, 1])
        p3 = scale([0, 1, 0])
        p4 = scale([0, 1, 1])
        p5 = scale([1, 0, 0])
        p6 = scale([1, 0, 1])
        p7 = scale([1, 1, 0])
        p8 = scale([1, 1, 1])

        # print("p1 =", p1)
        # print("p2 =", p2)
        # print("p3 =", p3)
        # print("p4 =", p4)
        # print("p5 =", p5)
        # print("p6 =", p6)
        # print("p7 =", p7)
        # print("p8 =", p8)

        if color is None and layer is None:
            base_color = 1
            left_color = 2
            front_color = 3
            right_color = 4
            back_color = 5
            top_color = 6
        if color is not None:
            if isinstance(color, list) and len(color) == 6:
                base_color, left_color, front_color, right_color, back_color, top_color = color
            else:
                base_color = left_color = front_color = right_color = back_color = top_color = color
        elif layer is not None:
            base_color = left_color = front_color = right_color = back_color = top_color = None

        # define the 6 cube faces
        # look into -x direction
        # Every add_face adds 4 vertices 6x4 = 24 vertices
        # On dxf output double vertices will be removed.
        pface.add_face([p1.flatten(), p5.flatten(), p7.flatten(), p3.flatten()], color = base_color)  # base
        pface.add_face([p1.flatten(), p5.flatten(), p6.flatten(), p2.flatten()], color = left_color)  # left
        pface.add_face([p5.flatten(), p7.flatten(), p8.flatten(), p6.flatten()], color = front_color)  # front
        pface.add_face([p7.flatten(), p8.flatten(), p4.flatten(), p3.flatten()], color = right_color)  # right
        pface.add_face([p1.flatten(), p3.flatten(), p4.flatten(), p2.flatten()], color = back_color)  # back
        pface.add_face([p2.flatten(), p6.flatten(), p8.flatten(), p4.flatten()], color = top_color)  # top
        return pface

    def get_corner_points(self, elongated_side=None, buffer=0.0):
        """           
        :return: tuple (xmin, ymin, zmin, xmax, ymax, zmax)         
        :param elongated_side: None -> whole beam; 0 -> center part; 1 -> elongated1; 2 -> elongated2
        """

        def scale(point):
            dim = self.get_dimensions(elongated_side=elongated_side)[:]
            bp = self.get_basepoint(elongated_side=elongated_side)[:]

            dim[0] += buffer
            dim[1] += buffer

            tmp_d = self.R * dim * np.array(point)
            # scaled point = bp + a * r1 * point(0) + b * r2 * point(1) + e * r3 * point(2)
            return bp + np.sum(tmp_d, 1)[np.newaxis].T

        if self.basepoint is None or self.R is None or self.dimensions is None:
            return None

        # cube corner points
        p1 = scale([0, 0, 0])
        p2 = scale([0, 0, 1])
        p3 = scale([0, 1, 0])
        p4 = scale([0, 1, 1])
        p5 = scale([1, 0, 0])
        p6 = scale([1, 0, 1])
        p7 = scale([1, 1, 0])
        p8 = scale([1, 1, 1])
        return (p1, p2, p3, p4, p5, p6, p7, p8)

    def get_bbox(self, elongated_side=None, buffer=0.0):
        allpts = np.array(self.get_corner_points(elongated_side=elongated_side, buffer=buffer))
        min_pt = np.min(allpts, axis=0)
        max_pt = np.max(allpts, axis=0)
        return tuple(min_pt.flatten())+tuple(max_pt.flatten())

    def get_dimensions(self, elongated_side=None):
        if elongated_side is None:
            dim = self.dimensions[:]  # copy list
        elif elongated_side == 0:
            dim = self.dimensions[:]
            dim[2] = dim[2] - np.sum(self.elongation)
        elif elongated_side == 1:
            dim = self.dimensions[:]
            dim[2] = self.elongation[0]
        elif elongated_side == 2:
            dim = self.dimensions[:]
            dim[2] = self.elongation[1]
        return dim

    def get_basepoint(self, elongated_side=None):
        tmpR = self.R[:]
        tmp_e = tmpR[:, 2, np.newaxis]
        if elongated_side is None:
            bp = self.basepoint[:]
        elif elongated_side == 0:
            bp = self.basepoint + self.elongation[0] * tmp_e
        elif elongated_side == 1:
            bp = self.basepoint[:]
        elif elongated_side == 2:
            bp = self.basepoint + (self.dimensions[2] - self.elongation[1]) * tmp_e
        return bp

    def contains_points(self, pts_array, elongated_side=None, buffer=0):
        beam_center = np.array(self.get_corner_points(elongated_side)).mean(axis=0).T
        dims = self.get_dimensions(elongated_side=elongated_side)

        tmpR = self.R[:] # copy array
        pt_distances_dim0 = np.dot(pts_array - beam_center, tmpR[:, 0, np.newaxis])
        pt_distances_dim1 = np.dot(pts_array - beam_center, tmpR[:, 1, np.newaxis])
        pt_distances_dim2 = np.dot(pts_array - beam_center, tmpR[:, 2, np.newaxis])

        return np.where(abs(pt_distances_dim0) <= (dims[0]/2)-buffer) \
               and np.where(abs(pt_distances_dim1) <= (dims[1]/2)-buffer) \
               and np.where(abs(pt_distances_dim2) <= (dims[2]/2)-buffer)
