import numpy as np
import ezdxf

import toolBox.geometry as geometry
import external.pbs_geometry
import external.pbs_joint

class Joint():
    def __init__(self, beam1_id, beam2_id, b1_position, b2_position, p_b1, p_b2, joint_type=None):
        self.beam1_id = beam1_id
        self.beam2_id = beam2_id
        self.b1_position = b1_position
        self.b2_position = b2_position
        self.p_b1 = p_b1
        self.p_b2 = p_b2
        self.b1_group = None
        self.b2_group = None

def detectJoint(b1, b2, tolerance = 0.1):
    #p_on_beam = np.array([ezdxf.math.intersection_ray_ray_3d((b1[1], b1[2]), (b2[1],b2[2])) for b2 in tmp_beam2])
    p_b1, p_b2 = geometry.get_segment_to_segment_connector(b1.axis[0], b1.axis[1], 
                                                                    b2.axis[0], b2.axis[1])
    if geometry.getDistance(p_b1, p_b2) < tolerance:

        b1_pos = geometry.getDistance(p_b1, b1.axis[0]) / geometry.getDistance(b1.axis[0], b1.axis[1])# distance to start pos. / full length
        b2_pos = geometry.getDistance(p_b2, b2.axis[0]) / geometry.getDistance(b2.axis[0], b2.axis[1])

        return Joint(b1.id, b2.id, b1_pos,b2_pos, p_b1, p_b2)
    else:
        return None