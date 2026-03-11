import numpy as np
import open3d as o3d
from pathlib import Path
import sys
import yaml

#from opals import pyDM
import laspy

import components.Beam as Beam
import external.pbs_beam
import external.pbs_enums
import external.pbs_processor

import ezdxf
from ezdxf import bbox
from ezdxf.addons import r12writer

from sklearn.cluster import KMeans
import toolBox.geometry as geometry

def readConfig(config_path):
    #Read parameters
    with open(config_path, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

"""
def readODM(odm):
    dm = pyDM.Datamanager.load(odm, False, False)
    if not dm:
        print("Unable to open ODM '" + odm + "'")
        sys.exit(1)
    
    stat = dm.getAddInfoStatistics()
    dmLayout = stat.layout()
    #output odm attribute layout
   
    lf = pyDM.AddInfoLayoutFactory()
    type, inDM = lf.addColumn(dm, "NormalX",   True); assert inDM == True
    type, inDM = lf.addColumn(dm, "NormalY", True); assert inDM == True
    type, inDM = lf.addColumn(dm, "NormalZ", True); assert inDM == True
    type, inDM = lf.addColumn(dm, "SegmentID", True); assert inDM == True
    layout = lf.getLayout()
      
    numpyDict = pyDM.NumpyConverter.createNumpyDict(dm.sizePoint(),layout,True)
    pointindex = dm.getPointIndex()
    
    rowIdx = 0
    count = float(pointindex.sizeLeaf())
    for idx,leaf in enumerate(pointindex.leafs()):
        rowIdx += pyDM.NumpyConverter.fillNumpyDict(numpyDict,rowIdx,leaf)
    
    points = np.vstack((numpyDict['x'], numpyDict['y'], numpyDict['z'])).transpose()
    normals = np.vstack((numpyDict['NormalX'], numpyDict['NormalY'], numpyDict['NormalZ'])).transpose()
    segments = numpyDict['SegmentID']
    return points, normals, segments
"""
def getPCD(path):

    extension = Path(path).suffix

    if extension in ('.las', '.laz'):
        try:
            #inFile = laspy.file.File(path, mode = 'r')
            las = laspy.read(path)
            #point_records = inFile.points       
            #coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(las.xyz)                       
            return pcd

        except Exception as e:
            print(e)
            #sys.exit(1)           
    elif extension == '.ply':
        try:
            pcd = o3d.io.read_point_cloud(path)
            return pcd
        except Exception as e:
            print(e)
            sys.exit(1)

def getBeamAxesAsDXF(beams):
    dxfBeamAxes = []
    for b in beams:
        line = ezdxf.entities.Line()
        line.dxf.start = b.axis[0]
        line.dxf.end = b.axis[1]
        line.dxf.layer = "Beam-Axes"
        dxfBeamAxes.append(line)
    return dxfBeamAxes

def lines2UVecs(lines, orientPos):
    if orientPos is None:
        orientPos = np.array([1000.0, 1000.0, 1000.0])

    uvecs = [geometry.getUnitVector(np.array(l.dxf.end.xyz) - np.array(l.dxf.start.xyz)) for l in lines]

    Pstart = [np.array(l.dxf.start.xyz) for l in lines]
    Pend = [np.array(l.dxf.end.xyz) for l in lines]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pstart)
    pcd.normals = o3d.utility.Vector3dVector(uvecs)
    
    #o3d.visualization.draw_geometries([pcd])

    pcd.orient_normals_towards_camera_location(camera_location = orientPos)
    #o3d.visualization.draw_geometries([pcd])


    ouvecs = np.asarray(pcd.normals)
    return uvecs, ouvecs

def readDXF(dxfPath):
    try:
        doc = ezdxf.readfile(dxfPath)
    except IOError:
        print(f'Not a DXF file or a generic I/O error.')
        sys.exit(1)
    except ezdxf.DXFStructureError:
        print(f'Invalid or corrupted DXF file.')
        sys.exit(2)
    return doc.modelspace()

def getDXFLayers(modelspace):
    from ezdxf.groupby import groupby
    layers = groupby(entities=modelspace, dxfattrib='layer')
    return layers

def getDXFLines(modelspace):
    lines = modelspace.query('LINE')
    return lines

def getDXFCuboids(modelspace):
    lines = modelspace.query('POLYLINE')
    return lines

def getBeamAxesofDXF(layers):
    return layers["Beam-Axes"]

def getBeamSurfacesofDXF(layers):
    return layers["Beams"]

def getBeamJointsofDXF(layers):
    if "Joints" in layers.keys():
        return layers["Joints"]
    else:
        return []

def readBeamsDXF(path):
        dxfModelSpace = readDXF(path)       
        #Read Beams from DXF
        dxfLayers = getDXFLayers(dxfModelSpace)
        dxfBeamAxes = getBeamAxesofDXF(dxfLayers)
        dxfBeams = getBeamSurfacesofDXF(dxfLayers)
        dxfJoints = getBeamJointsofDXF(dxfLayers)
        return {"Beam-Axes":dxfBeamAxes, "Beams":dxfBeams, "Joints":dxfJoints}

def readBeamsDXFOriented(path, pos=None):
        dxf_dict = readBeamsDXF(path)
        #Orient Beam-Axes
        dxfBeamAxes = dxf_dict["Beam-Axes"]
        orientationPos = pos
        uvecs, ouvecs = lines2UVecs(dxfBeamAxes, orientationPos)
        for i, vec in enumerate(uvecs):
            if not np.array_equal(vec, ouvecs[i]):
                line = ezdxf.entities.Line()
                line.dxf.start = dxfBeamAxes[i].dxf.end
                line.dxf.end = dxfBeamAxes[i].dxf.start
                line.dxf.layer = "Beam-Axes"
                dxfBeamAxes[i] = line
                #TODO: return ouvecs too -> write to db nx ny nz
        return {"Beam-Axes":dxf_dict["Beam-Axes"], "Beams":dxf_dict["Beams"], "Joints":dxf_dict["Joints"], "Beam-Orientation":ouvecs}

def mergeDXFs(paths):
    axes = []
    beams = []
    joints = []
    for p in paths:
        dxf_dict = readBeamsDXF(p)
        axes.extend(dxf_dict["Beam-Axes"])
        beams.extend(dxf_dict["Beams"])
        joints.extend(dxf_dict["Joints"])
    return {"Beam-Axes":axes, "Beams":beams, "Joints":joints}

def getCuboidFaces(cuboid):
    faces = []
    for f in cuboid.faces():
        fList = [ fe.dxf.location.xyz for fe in f]
        faces.append(fList[:4])
    return np.array(faces)

def getBeamOBB(cuboid):
    faces = getCuboidFaces(cuboid)
    if len(faces) == 6:
       pts = [*faces[0], *faces[5]] # top and bottom points of cuboid
       #pcd = o3d.geometry.PointCloud()
   #    pcd.points = o3d.utility.Vector3dVector(pts)
       #obb = pcd.get_oriented_bounding_box()
       obb = o3d.geometry.OrientedBoundingBox()
       obb =  obb.create_from_points(points=o3d.utility.Vector3dVector(pts))
       return obb

def obb2PbsBeam(obb):
    #Quality check - 1 : OBB reliability check
    if obb.extent[0] < obb.extent[1]:
        new_ext = [ obb.extent[1],  obb.extent[0],  obb.extent[2]]
        obb.extent = new_ext         
        new_rot = np.array([[obb.R[0][1], obb.R[0][0], obb.R[0][2]],[obb.R[1][1], obb.R[1][0], obb.R[1][2]],[obb.R[2][1], obb.R[2][0], obb.R[2][2]]])
        obb.R = new_rot
    if obb.extent[2] > obb.extent[0]:
        new_ext = [ obb.extent[2],  obb.extent[0],  obb.extent[1]]
        obb.extent = new_ext         
        new_rot = np.array([[obb.R[0][2], obb.R[0][0], obb.R[0][1]],[obb.R[1][2], obb.R[1][0], obb.R[1][1]],[obb.R[2][2], obb.R[2][0], obb.R[2][1]]])
        obb.R = new_rot

    if obb.extent[1] < obb.extent[0] and obb.extent[1] < obb.extent[2]:
        new_ext = [ obb.extent[0],  obb.extent[2],  obb.extent[1]]
        obb.extent = new_ext         
        new_rot = np.array([[obb.R[0][0], obb.R[0][2], obb.R[0][1]],[obb.R[1][0], obb.R[1][2], obb.R[1][1]],[obb.R[2][0], obb.R[2][2], obb.R[2][1]]])
        obb.R = new_rot

    #OBB->Beam Conversion test
    beam_ = external.pbs_beam.Beam(None)

    beam_.R = np.array([obb.R[:,2], obb.R[:,1], -obb.R[:,0]]).transpose()
    beam_.dimensions = np.array([obb.extent[2], obb.extent[1], obb.extent[0]])

    #Define the basepoint
    beam_box_pts = np.array(obb.get_box_points())
    bbp = np.round(beam_box_pts, decimals = 4)
    diff_list = []
    base_list = []
    for p in beam_box_pts:
        candidate = np.dot(beam_.R, beam_.dimensions) + p
        cnd = np.round(candidate, decimals = 4)
        if cnd in bbp:
            #beam_.basepoint = np.array(([p[0]],[p[1]],[p[2]]))
            base_list.append(np.array(([p[0]],[p[1]],[p[2]])))

        diff_list.append(candidate)

    #Quality Check - 2 : Multiple base-point candidates
    if len(base_list) == 1:
        beam_.basepoint = base_list[0]
    elif len(base_list) >1:
        #print("Make correction here")
        #pbs_b = exchange.obb2PbsBeam(obb)
        #Here check if the boxes are equal

        for base in base_list:
            beam_.basepoint = base
            pbs_pts = np.array(beam_.get_corner_points()).reshape(8,3)
            obb_pts = np.array(obb.get_box_points())
            
            sum_distance = 0
            for p in pbs_pts:
                distances = np.array([geometry.getDistance(p,p2) for p2 in obb_pts])
                min_dist = min(distances)
                sum_distance += min_dist
            
            if sum_distance < 0.02:
                break
            else:
                #This is defected beam
                print("Missing Beam at ", obb.get_center())
                #pbs_b2 = exchange.obb2PbsBeam(obb)
                #pbs_pc = o3d.geometry.PointCloud()
                #pcd_b.points = o3d.utility.Vector3dVector(pbs_pts) 
                #hull, _ = pcd_b.compute_convex_hull()
                #o3d.visualization.draw([hull, obb])
            
    beam_.sigma0 = 0.001
    return beam_

def getBeamDict(pbsBeams):
    pbsBeamsDict = []
    for beam in pbsBeams:
        pbsBeamsDict.append({'beam_id': beam.id, 'beam_obj': beam})
    return pbsBeamsDict

def obb2Mesh(obb):
    obb_vertices = np.asarray(obb.get_box_points())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obb_vertices) 
    hull, _ = pcd.compute_convex_hull()
    return hull

def mesh2OBBs(mesh , src = "o3d"):
    if src == "o3d":
        face_count = 12
    elif src == "RSTAB":
        face_count = 16

    mesh_clustering = mesh.cluster_connected_triangles() # 0 -> cluster_idx of triangle 1 -> count of cluster
    mesh_cluster_idx = np.unique(np.array(mesh_clustering[0]))
    mesh_cluster_counts = np.array(mesh_clustering[1])

    valid_clusters = np.unique(mesh_cluster_idx[mesh_cluster_counts == face_count])
    #valid_clusters = np.unique(mesh_cluster_idx)

    mesh_triangles = np.asarray(mesh.triangles)
    mesh_vertices = np.asarray(mesh.vertices)
    #GLB to original coordinate system transformation
    if src == "RSTAB":
        mesh_vertices = np.vstack((mesh_vertices[:,0],-mesh_vertices[:,2],mesh_vertices[:,1])).transpose()

    beam_boxes = []
    for cluster_id in valid_clusters:

        triangles = mesh_triangles[np.array(mesh_clustering[0])==cluster_id]
        vertex_indices = np.unique(triangles)
        vertices = mesh_vertices[vertex_indices]
        box_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))

        if len(vertices):
            try:
                box = box_pts.get_minimal_oriented_bounding_box(robust=False) 
                if min(box.extent) > 0.05 and max(box.extent) > 0.2:
                    beam_boxes.append(box)
            except:
                continue
    beams = [Beam.obb2Beam(b) for b in beam_boxes]
    return beam_boxes


class PBS_GUI():
    _MIN_SEGMENT_AREA = 0.05
    _MAX_SEGMENT_AREA = 50
    _MIN_BEAM_WIDTH  = 0.1
    _MAX_BEAM_WIDTH  = 0.4
    _ALPHA_SHAPE_MODE = True
    _SIMPLIFY_DP  = False
    _SAVE = False
    _LOAD = False
    _DEBUG = False
    _MULTIPROCESSING = True
    _NORMALSOUT = False
    _MIN_SEGMENT_PTS = 500
    _FILENAME = "TEST"
    _MAX_JOINT_LEN = 0.15
    _MATERIALS = ["Nadelholz C24"]
    _USE_ELONGATE_MATERIAL = False
    _ELONGATE_MATERIAL = ["Nadelholz C30"]
    _ELONGATION_MODE = external.pbs_enums.ELONGATION_MODE.ELONGATE

def getBeamProcessor(pbsBeams):
    beamDict = getBeamDict(pbsBeams)
    gui = PBS_GUI()
    gui
    processor = external.pbs_processor.Processor(gui)
    processor.beams = beamDict
    return processor