from time import strftime, gmtime
import numpy as np
import external.pbs_enums 

class StpWriter:

    def __init__(self, filename, file_description="", materials = [], crosssections = None, unit_factor = 1000):
        self.filename = filename
        self.file_description = file_description
        self.cross_sections = []
        self.uf = unit_factor # m -> mm
        if crosssections is not None:
            for cs in crosssections:
                self.add_cross_section(cs[0]*self.uf, cs[1]*self.uf) # in mm unit
        else:
            self.add_cross_section(0.2*self.uf, 0.1*self.uf)  # add default in mm unit

        """ Material and cross-section for testing """
        if len(materials) == 0:
            self.add_material("Nadelholz C24")  # based on DSTV98a
        else:
            self.materials = materials
        self.curr_id = 1
        self.substructure_id = None
        self.time = None

    def add_material(self, material_name):
        self.materials.append(material_name)

    def add_cross_section(self, a, b):
        self.cross_sections.append({"a":a, "b":b})

    def add_beams(self, beams, elongation_mode=external.pbs_enums.ELONGATION_MODE.ELONGATE):
        self.beams = beams
        self.elongation_mode = elongation_mode

    def add_joints(self, joints):
        self.joints = joints

    def write_file(self):

        self.time = strftime("%Y-%m-%dT%H:%M:%S", gmtime())

        with open(self.filename, "w") as outfile:
            self.write_header(outfile)
            outfile.write("DATA;\n\n")

            self.write_materials(outfile)

            self.write_cross_sections(outfile)

            self.write_structural_data(outfile)

            """ Write Beams and Joints (Rigid coupling) """
            node_i = 1
            len_beams = len(self.beams)*3 # max. nr of beams (beam object can contain 3 beams-sectors)
            len_joints = len(self.joints)
            members_id = self.curr_id + 4 * len_beams + 4 * len_joints
            members_queue = []
            max_elem_id = -1
            outfile.write("/* NODES */\n")
            if len(self.materials) > 1:
                m_classes = [2, 1, 2]
            else:
                m_classes = [1, 1, 1]
            map_beamid_stpid = {}
            for beam in self.beams:
                b_axis = beam['beam_obj'].get_beam_axis(elongation_mode=self.elongation_mode)
                m_idx = -1
                for b_start, b_end in b_axis:
                    m_idx += 1
                    if b_start is None or b_end is None:
                        continue
                    # start point position vector
                    outfile.write("#%d=VERTEX($,%f,%f,%f,$);\n" % (self.curr_id, b_start[0]*self.uf, b_start[1]*self.uf, b_start[2]*self.uf))
                    beam_node_start = self.curr_id + 1
                    # start point position node
                    outfile.write("#%d=NODE('PBS_NODE_%d',%d,'%s',#%d,#%d);\n" %
                                  (beam_node_start, node_i, node_i, self.time, self.substructure_id, self.curr_id))
                    node_i += 1
                    self.curr_id += 2
                    # end point position vector
                    outfile.write("#%d=VERTEX($,%f,%f,%f,$);\n" % (self.curr_id, b_end[0]*self.uf, b_end[1]*self.uf, b_end[2]*self.uf))
                    beam_node_end = self.curr_id + 1
                    # end point position node
                    outfile.write("#%d=NODE('PBS_NODE_%d',%d,'%s',#%d,#%d);\n" %
                                  (beam_node_end, node_i, node_i, self.time, self.substructure_id, self.curr_id))
                    node_i += 1
                    self.curr_id += 2

                    # Beam coordinate system given in:
                    #   x = axis a (height or width)
                    #   y = axis b (width or height)
                    #   z = longitudinal axis
                    # Beam coordinate system required in:
                    #   x' = longitudinal axis
                    #   y' = "strong" axis (height)
                    #   z' = "weak" axis (width)

                    x_dim = np.array([[0], [0], [1]])
                    y_dim = np.array([[1], [0], [0]])
                    z_dim = np.array([[0], [1], [0]])
                    # get stp axes y' and z'
                    if beam['beam_obj'].dimensions[0] < beam['beam_obj'].dimensions[1]:
                        z_dim = np.array([[1], [0], [0]])
                        y_dim = np.array([[0], [1], [0]])

                    #print("Debug: beam_height - class_height = %.6f" % (beam['beam_obj'].dimensions[0] - beam['beam_obj'].cross_section_class['a']))
                    #print("Debug: beam_width - class_width = %.6f" % (beam['beam_obj'].dimensions[1] - beam['beam_obj'].cross_section_class['b']))
                    #print("y-dim: %s" % (y_dim))

                    vx = np.dot(beam['beam_obj'].R, x_dim)
                    vy = np.dot(beam['beam_obj'].R, y_dim)
                    vz = np.dot(beam['beam_obj'].R, z_dim)
                    if vy[2] > 0:
                        vx *= -1
                        vy *= -1 # flip vector direction - y-axis points downwards
                        vz *= -1

                    # Vertex specifies the rotation of the beam around the beam axis - vertex is in XZ-Plane of the beam
                    members_queue.append("#%d=VERTEX($,%f,%f,%f,$);\n" % (members_id, (b_start[0] + vz[0]) * self.uf,
                                                                                      (b_start[1] + vz[1]) * self.uf,
                                                                                      (b_start[2] + vz[2]) * self.uf))

                    material_id = m_classes[m_idx]
                    cs_id = beam['beam_obj'].cross_section_class['id']
                    cs_ref_id = -1
                    cs_ref_valid = False
                    if cs_id is not None and cs_id >= 0:
                        cs = self.cross_sections[cs_id]
                        if cs is not None:
                            cs_ref_id = cs['ref_id']
                        cs_ref_valid = cs_ref_id is not None and cs_ref_id > 0

                    if not cs_ref_valid:
                        print("No valid cross-section detected for beam %s - use default cross-section" % beam['beam_id'])

                    max_elem_id = max(beam['beam_id'], max_elem_id+1)
                    # Beam element from Node_start to Node_end with a given material and cross-section
                    # ID, NR, DATE, TYP, (NODES), Reference-Point, ALPHA Rotation around X, MemberID, CS Start, CS end, material
                    members_queue.append("#%d=ELEMENT('PBS_ELEM_%d_%d',%d,'%s',.BEAM.,(#%d,#%d),#%d,$,$,#%d,#%d,#%d);\n" %
                        (members_id+1, max_elem_id, m_idx, max_elem_id, self.time, beam_node_start, beam_node_end, members_id, cs_ref_id, cs_ref_id, material_id))
                    # members_id += 1
                    members_id += 2


            for joint in self.joints:
                j_start, j_end = joint.joint_points
                # start point position vector
                outfile.write("#%d=VERTEX($,%f,%f,%f,$);\n" % (self.curr_id, j_start[0]*self.uf, j_start[1]*self.uf, j_start[2]*self.uf))
                j_node_start = self.curr_id + 1
                # start point position node
                outfile.write("#%d=NODE('PBS_NODE_%d',%d,'%s',#%d,#%d);\n" %
                              (j_node_start, node_i, node_i, self.time, self.substructure_id, self.curr_id))
                node_i += 1
                self.curr_id += 2
                # end point position vector
                outfile.write("#%d=VERTEX($,%f,%f,%f,$);\n" % (self.curr_id, j_end[0]*self.uf, j_end[1]*self.uf, j_end[2]*self.uf))
                j_node_end = self.curr_id + 1
                # end point position node
                outfile.write("#%d=NODE('PBS_NODE_%d',%d,'%s',#%d,#%d);\n" %
                              (j_node_end, node_i, node_i, self.time, self.substructure_id, self.curr_id))
                node_i += 1
                self.curr_id += 2

                # members_queue.append("#%d=VERTEX($,%f,%f,%f,$);\n" % (members_id, 0, 0, 0)) # TODO: ??
                # members_queue.append(
                #     "#%d=ELEMENT('PBS_ELEM%s',%d,'%s',.RIGID.,(#%d,#%d),#%d,0.000000,$,$,$,$);\n" %
                #     (members_id + 1, 'J'+str(joint.id), joint.id, self.time, j_node_start, j_node_end,
                #      members_id))
                # members_id += 2
                members_queue.append(
                    "#%d=ELEMENT('PBS_ELEM_%s',%d,'%s',.RIGID.,(#%d,#%d),$,0.000000,$,$,$,$);\n" %
                    (members_id, 'J' + str(joint.id), max_elem_id + 1 + joint.id, self.time, j_node_start, j_node_end))
                members_id += 1


            outfile.write("\n/* MEMBERS */\n")
            outfile.writelines(members_queue)
            outfile.write("\n")
            outfile.write("/* SETS OF MEMBERS */\n")
            outfile.write("\n")
            outfile.write("/* COUPLING RIGID */\n")
            outfile.write("\n")
            outfile.write("/* MEMBER RELEASES */\n")
            outfile.write("\n")
            outfile.write("/* MEMBER ELASTIC FOUNDATIONS */\n")
            outfile.write("\n")

            outfile.write("ENDSEC;\n") # end data section
            outfile.write("END-ISO-10303-21;\n") # end file

    def write_header(self, outfile):
        outfile.write("ISO-10303-21;\n")
        outfile.write("HEADER;\n")

        outfile.write("FILE_DESCRIPTION(('" + self.file_description + "'),'2(IN)/2(OUT)');\n")
        outfile.write("FILE_NAME('" + self.filename + "', '" + self.time + "', ('PASTiSt Tool'), "
                        "('TU Wien, Research Unit Photogrammetry, "
                        "Karlsplatz 13 1040 WIEN'),'PASTiSt Tool', 'PASTiSt Tool', 'PASTiSt Tool');\n")
        outfile.write("FILE_SCHEMA(('PSS_2000_04'));\n")
        outfile.write("ENDSEC;\n\n")

    def write_materials(self, outfile):
        outfile.write("/* MATERIALS */\n")
        material_i = 1
        for material_name in self.materials:
            #  nr, name (DSTV98a), Elastic modulus E, Shear modulus G, spec. density RHO, alpha_temp, F_Y_K, F_U_K, EPS_U_K, GAMMA
            outfile.write("#%d=MATERIAL(%d,'%s',1.10E+04,690.000000,0.000005,0.000005,$,$,$,1.300000);\n" % (self.curr_id, material_i, material_name))
            material_i += 1
            self.curr_id += 1
        outfile.write("\n")

    def write_cross_sections(self, outfile):
        outfile.write("/* CROSS-SECTIONS */\n")
        cs_i = 0
        for cs in self.cross_sections:
            cs['ref_id'] = self.curr_id
            # nr, profiltyp, def, name (DSTV98a), profilreihe, H, B, ..20.., Querschnittsflaeche,
            outfile.write("#%d=CROSS_SECTION(%d,.B.,$,'%.6f*%.6f','HRechteck %d / %d',%.6f,%.6f,"
                          "$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,"
                          "$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$);\n" %
                          (cs['ref_id'], cs_i+1, cs['a'], cs['b'], round(cs['a'],0), round(cs['b'],0), cs['a'], cs['b']))
            cs_i += 1
            self.curr_id += 1
        outfile.write("\n")

    def write_structural_data(self, outfile):
        outfile.write("/* STRUCTURAL DATA */\n")
        # Origin of local coordinate system
        outfile.write("#%d=VERTEX($,0.000000,0.000000,0.000000,$);\n" % self.curr_id)
        # Point on new x-axis
        outfile.write("#%d=VERTEX($,1.000000,0.000000,0.000000,$);\n" % (self.curr_id+1))
        # Point on new xz-plane
        outfile.write("#%d=VERTEX($,0.000000,0.000000,1.000000,$);\n" % (self.curr_id+2))
        self.substructure_id = self.curr_id+3
        outfile.write("#%d=SUBSTRUCTURE(5,.THREE_DIM.,'',#%d,#%d,#%d);\n" % (self.substructure_id, self.curr_id, self.curr_id+1, self.curr_id+2))
        self.curr_id += 4
        outfile.write("\n")