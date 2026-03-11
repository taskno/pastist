import numpy as np
from dxfwrite import DXFEngine as dxf
import toolBox.geometry as geometry
import external.pbs_joint
from external.pbs_stp_writer import StpWriter

class Processor:
    def __init__(self, gui):

        self.PBS_GUI = gui
        self._PROCESSING_THREADS = []
        self._MP_QUEUE = None
        self._PROCESSES = []
        self._PROCESS_POOL = None
        self._UI_MP_QUEUE = None
        self._UI_PROCESSES = []
        self._UI_PROCESS_POOL = None

        self._PARAMS = {}
        self._PARAMS['filename'] = gui._FILENAME

        self._PARAMS['min_seg_area'] = gui._MIN_SEGMENT_AREA  # m^2
        self._PARAMS['max_seg_area'] = gui._MAX_SEGMENT_AREA  # m^2
        self._PARAMS['min_beam_width'] = gui._MIN_BEAM_WIDTH  # m
        self._PARAMS['max_beam_width'] = gui._MAX_BEAM_WIDTH  # m

        self._PARAMS['alpha_shape_mode'] = gui._ALPHA_SHAPE_MODE
        self._PARAMS['simplify_alpha_shape'] = gui._SIMPLIFY_DP
        self._PARAMS['save'] = gui._SAVE
        self._PARAMS['load'] = gui._LOAD
        self._PARAMS['debug'] = gui._DEBUG
        self._PARAMS['multiprocessing'] = gui._MULTIPROCESSING
        self._PARAMS['normalsout'] = gui._NORMALSOUT
        self._PARAMS['min_seg_pts'] = gui._MIN_SEGMENT_PTS
        self.thread_last_started = None

        self.AX = None
        self.AX_LINEAR = None

        self.beam_segments = []
        self.other_segments = []
        self._CLASSIFIED_SEGMENTS = []

        self.stopwatch = {
            "segment_classification":{"start":None, "end":None},
            "beam_modelling":{"start":None, "end":None}
        }
        self.beams = []
        self.joints = []      
        self.no_points = 0
        self.no_processed_points = 0
        self.processing_segments = []
        self.automatic_processing_finished = False
        self.user_processing_finished = True
        self.main_cad_drawing = None      
        return

    def export_beams_dxf(self, filename):
        beams_drawing = dxf.drawing(name=filename)
        beams_drawing.add_layer(name="Beams", color=1)
        beams_drawing.add_layer(name="Beam-Axes", color=7)
        beams_drawing.add_layer(name="Joints", color=2)
        for beam in self.beams:
            tmp_beam = beam['beam_obj']

            """ Color functions """
            def colorize_by_no_covered_sides(beam):
                tmp_sidescfg = beam.cuboid.sides_cfg
                #                   base, right, oppo, left
                if tmp_sidescfg == [True, True, False, False]:
                    color = 1  # red
                elif tmp_sidescfg == [True, False, False, True]:
                    color = 2  # yellow
                elif tmp_sidescfg == [True, True, False, True]:
                    color = 3  # green
                elif tmp_sidescfg == [True, False, True, True]:
                    color = 4  # turquoise
                elif tmp_sidescfg == [True, True, True, True]:
                    color = 5  # blue
                else:
                    color = 7  # white
                return color
            def colorize_by_sigma0(beam):
                sigma0 = beam.sigma0
                if sigma0 < 0.01:
                    color = 3 # green
                elif sigma0 < 0.025:
                    color = 2 # yellow
                else:
                    color = 1 # red
                return color

            tmp_dxf_cuboid = tmp_beam.get_dxfwrite_cuboid(color=colorize_by_sigma0(tmp_beam),layer="Beams")
            if tmp_dxf_cuboid is not None:
                beams_drawing.add(tmp_dxf_cuboid)

            b_axis = tmp_beam.get_beam_axis(elongation_mode=self.PBS_GUI._ELONGATION_MODE)
            for tmp_start, tmp_end in b_axis:
                if tmp_start is not None and tmp_end is not None:
                    tmp_beam_axis = dxf.line(start=(tmp_start[0], tmp_start[1], tmp_start[2]),
                                             end=(tmp_end[0], tmp_end[1], tmp_end[2]), layer="Beam-Axes")
                    tmp_beam_axis['color'] = 7
                    beams_drawing.add(tmp_beam_axis)

        for joint in self.joints:
            joint_start, joint_end = joint.joint_points
            if joint_start is not None and joint_end is not None:
                tmp_beam_axis = dxf.line(start=(joint_start[0], joint_start[1], joint_start[2]),
                                         end=(joint_end[0], joint_end[1], joint_end[2]), layer="Joints")
                tmp_beam_axis['color'] = 2
                beams_drawing.add(tmp_beam_axis)
        beams_drawing.save()

    def export_beams_stp(self, filename, model=None):
        if model is not None:
            k_means_result = model
        else:
            k_means_result = self.classify_cross_sections(model)
        #print("Cross section classes:")
        #print(k_means_result)

        tmp_materials = self.PBS_GUI._MATERIALS
        if self.PBS_GUI._USE_ELONGATE_MATERIAL == True:
            tmp_materials = tmp_materials + self.PBS_GUI._ELONGATE_MATERIAL

        stp_writer = StpWriter(filename, "PASTiSt STEP file Export", tmp_materials, k_means_result)
        stp_writer.add_beams(self.beams, elongation_mode=self.PBS_GUI._ELONGATION_MODE)
        stp_writer.add_joints(self.joints)

        stp_writer.write_file()

    def classify_cross_sections(self, model= None):
        if model == None:
            dim_list = []
            
            for beam in self.beams:
                dim = beam['beam_obj'].dimensions
                if dim[0] > dim[1]:
                    dimA = dim[0]
                    dimB = dim[1]
                else:
                    dimA = dim[1]
                    dimB = dim[0]
                dim_list.append((dimA, dimB))
            
            dim_list = np.array(dim_list, dtype=object)
            
            silhouette_avg = {}
            from sklearn.cluster import k_means, KMeans
            from sklearn.metrics import silhouette_score
            
            max_k = 20
            if dim_list.shape[0] < max_k:
                max_k = dim_list.shape[0]
            range_k = range(6, max_k)
            
            maxdist = -1
            best_k = None
            # http://scikit-learn.sourceforge.net/dev/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
            for k in range_k:
                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
            
                clusterer = KMeans(n_clusters=k, random_state=10)
                cluster_labels = clusterer.fit_predict(dim_list)
            
                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg[k] = silhouette_score(dim_list, cluster_labels)
                # print("silhouette score for k=%d: %.2f" % (k, silhouette_avg[k]))
            
            for k in range_k:
                pt_k = np.array([k, silhouette_avg[k]])
                pt_0 = np.array([range_k[0], silhouette_avg[range_k[0]]])
                pt_end = np.array([range_k[-1], silhouette_avg[range_k[-1]]])
            
                # http://geomalgorithms.com/a02-_lines.html
                signed_dist = ((pt_0[1] - pt_end[1]) * pt_k[0] + (pt_end[0] - pt_0[0]) * pt_k[1] + (
                    pt_0[0] * pt_end[1] - pt_end[0] * pt_0[1])) / np.linalg.norm(pt_end - pt_0)
                # print("signed_dist for k=%d: %.2f" % (k, signed_dist))
                if (signed_dist > maxdist):
                    best_k = k
                    maxdist = signed_dist
            
            print("Calculation of Cross-section-classes")
            print("--- Number of classes: k =", best_k)
            try:
                k_fitting_models, assigned_k, inertia_ = k_means(dim_list, best_k)
                k_means = {'fitting-model': k_fitting_models, 'assigned-k': assigned_k, 'k': best_k}
                # k_means = pbs_fit.k_means(orientation, k=k, iter=50, threshold=20*np.pi/180)
                # assigned_k = k_means['assigned-k']
                # k_fitting_models = k_means['fitting-model']
                point_list = []
            
                for i in range(best_k):
                    for beam in np.extract(assigned_k == i, self.beams):
                        dim_a = beam['beam_obj'].dimensions[0]
                        dim_b = beam['beam_obj'].dimensions[1]
                        if dim_a < dim_b:
                            dim_a = beam['beam_obj'].dimensions[1]
                            dim_b = beam['beam_obj'].dimensions[0]
                        beam['beam_obj'].cross_section_class = {'id': i,
                                                                'a': k_fitting_models[i,0],
                                                                'b': k_fitting_models[i,1]}
                        print("Beam %s: cross section class %.4f/%.4f - difference beam2class: %.4f/%.4f" %
                              (beam['beam_id'], beam['beam_obj'].cross_section_class['a'],
                               beam['beam_obj'].cross_section_class['b'],
                               dim_a - beam['beam_obj'].cross_section_class['a'],
                               dim_b - beam['beam_obj'].cross_section_class['b']))
                        # shapes[np.where(assigned_k == i)[0]].orientation_class = k_fitting_models[:, i]
                        point_list.append([dim_a, dim_b, i])
            except:
                point_list = []
                k_fitting_models = []
                for beam in self.beams:
                    dim_a = beam['beam_obj'].dimensions[0]
                    dim_b = beam['beam_obj'].dimensions[1]
                    if dim_a < dim_b:
                        dim_a = beam['beam_obj'].dimensions[1]
                        dim_b = beam['beam_obj'].dimensions[0]
                    k_fitting_models.append([dim_a, dim_b])
                    point_list.append([dim_a, dim_b, len(k_fitting_models)-1])
                    beam['beam_obj'].cross_section_class = {'id': len(k_fitting_models)-1, 'a': dim_a, 'b': dim_b}
            finally:
                #from external.beam_class_window import BeamClassWindow
                #self.aw = BeamClassWindow(point_list, k_fitting_models)
                #self.aw.show()
                return k_fitting_models

        else:
            for beam in self.beams:
                    dim_a = model[0][0]
                    dim_b = model[0][1]
                    beam['beam_obj'].cross_section_class = {'id': len(model)-1, 'a': dim_a, 'b': dim_b}
            return model

    def automatic_joint_detection(self):
        processed_beams = []
        self.joints = []
        for beam1 in self.beams:
            b_axis1 = beam1['beam_obj'].get_beam_axis()
            start1, end1 = b_axis1[1]
            for beam2 in self.beams:
                if beam1['beam_id'] is beam2['beam_id'] or beam2['beam_id'] in processed_beams:
                    continue
                b_axis2 = beam2['beam_obj'].get_beam_axis()
                start2, end2 = b_axis2[1]
                pa, pb = geometry.get_segment_to_segment_connector(start1, end1, start2, end2)
                if np.linalg.norm(pa-pb) <= self.PBS_GUI._MAX_JOINT_LEN:
                    joint = external.pbs_joint.Joint([beam1['beam_obj'],beam2['beam_obj']],joint_points=(pa,pb))
                    self.joints.append(joint)
                    #self.PBS_GUI.beamaxis_figure.ax.plot([pa[0], pb[0]],  # x
                    #                                     [pa[1], pb[1]],  # y
                    #                                  zs=[pa[2], pb[2]],  # z
                    #             label=str(joint.id), linewidth=2, c='orange', zorder=1)

            processed_beams.append(beam1['beam_id'])
        print("%d joints detected" % len(self.joints))