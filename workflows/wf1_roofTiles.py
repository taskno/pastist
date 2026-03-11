import sys
import argparse
import yaml
from datetime import datetime
import os
from pathlib import Path
import numpy as np

import toolBox.database as database
import toolBox.exchange as exchange
import toolBox.geometry as geometry

import roof.RoofTile as RoofTile
import roof.Beam as Beam

import open3d as o3d
import ezdxf
import alphashape
from shapely import wkb
from shapely.geometry import Point
from sklearn.cluster import KMeans


def getPlanes(point_cloud, voxel_size, nr_planes=4, ransac_th = 0.15, ransac_n= 3, ransac_iter= 10000, visibility_top = True):

    pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)#0.2

    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    
    if visibility_top:
        radius = diameter * 1000
        
        obb_ = pcd.get_oriented_bounding_box()
        obb_.color = (1,1,0)
        
        aabb_ = pcd.get_axis_aligned_bounding_box()
        aabb_.color = (0,1,0)
        
        # Get top view
        camera = [obb_.center[0], obb_.center[1], obb_.center[2] + diameter]      
        _, pt_map = pcd.hidden_point_removal(camera, diameter * 200)       
        pcd = pcd.select_by_index(pt_map, invert=False)

    inlier_clouds = []
    inlier_planes = []
    colors = [[1.0, 1.0, 0], # Yellow
             [0.0, 1.0, 1.0], # Turquoise
             [0.0, 1.0, 0.0], # Green
             [1.0, 0.0, 0.0] # Red
             ]

    if nr_planes > 4:
        import random
        for i in range(nr_planes-4):
            random.uniform(0, 1)
            colors.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])


    outlier_cloud = pcd
    for i in range(nr_planes):
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=ransac_th,
                                                 ransac_n=ransac_n,
                                                 num_iterations=ransac_iter)
        
        [a, b, c, d] = plane_model
        print(f"Plane equation {i}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = outlier_cloud.select_by_index(inliers)
        inlier_cloud.paint_uniform_color(colors[i])
        inlier_clouds.append(inlier_cloud)
        inlier_planes.append(plane_model)
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

    return inlier_clouds, inlier_planes

def getAlphaShape(pts3D, plane):
    pts2D = geometry.project3DPointsToPlane2D(np.array(pts3D), plane)
    alpha_shape = alphashape.alphashape(pts2D, 1.)
    return alpha_shape

def createProcessFolder(begin, pcd):
    process_dir = os.path.dirname(os.path.abspath(pcd))
    beginTag = "stage1_" + str(begin.strftime("%Y%m%d_%H%M%S%f"))
    processFolder = os.path.join(process_dir, beginTag)
    os.mkdir(processFolder)
    return processFolder

def saveLines2DXF(lines, outputPath, color, id):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for l in lines:
        msp.add_line(l.axis[0], l.axis[1])
    
    linesW = msp.query('LINE')
    
    for k,lW in enumerate(linesW):
        lW.dxf.color = color
    doc.saveas(outputPath + "/plane_lyr_" + str(id) + ".dxf")

def main(config_path):
    start = datetime.now()
    #Read parameters
    with open(config_path, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
   
    if config_data.__contains__('voxel_size'):
        voxel_size = config_data['voxel_size']
    if config_data.__contains__('nr_planes'):
        nr_planes = config_data['nr_planes']
    if config_data.__contains__('ransac_th'):
        ransac_th = config_data['ransac_th']
    if config_data.__contains__('show_roof_tiles'):
        show_roof_tiles = config_data['show_roof_tiles']
    if config_data.__contains__('db_rcp'):
        db_rcp = config_data['db_rcp']
    if config_data.__contains__('create_roof_tiles'):
        create_roof_tiles = config_data['create_roof_tiles']
    if config_data.__contains__('angle_th'):
        angle_th = config_data['angle_th']


    #Create a process folder to store sub-results(dxfs)
    process_folder = createProcessFolder(start, db_rcp)

    #Read roof cover point cloud
    pcd = exchange.getPCD(db_rcp)

    roof_db = database.modelDatabase(config_path)
    if create_roof_tiles:
        #Get n planar point clouds
        plane_points, plane_params = getPlanes(pcd, voxel_size, nr_planes, ransac_th, visibility_top=False)
        
        if show_roof_tiles:
            o3d.visualization.draw(plane_points)
            print("Keep going ? (y/n)")
            keep_going = str(input())
            if keep_going != "y":
                print("Process stops.")
                sys.exit(0)
            else:
                print("Process is going on with %s planes"  % nr_planes)
        
        #Get alpha_shape of large planes
        alpha_shapes = [ getAlphaShape(pts.points, plane_params[i]) for i, pts in enumerate(plane_points)]
        roof_tiles = [ RoofTile.RoofTile(i+1, plane_params[i], a_s)  for i, a_s in enumerate(alpha_shapes)]
        
        #Insert roof tiles to DB       
        roof_db.fillRoofTileTable(roof_tiles)

    else:
        #CASE: Already have roof tiles on db
        #Fetch roofTiles from db
        roof_tile_records = roof_db.getRoofTiles()
        roof_tiles =  [RoofTile.RoofTile(id=r['id'], plane=[float(r['plane_a']), float(r['plane_b']), float(r['plane_c']), float(r['plane_d'])], alpha_shape2d = None) for r in roof_tile_records]
        plane_params = [t.plane for t in roof_tiles]


    #Fetch data from beam table
    beam_records = roof_db.getBeams(["id", "axis_start", "axis_end", "nx", "ny", "nz", "width", "height", "length"])
    beam_attr = [(r['id'], wkb.loads(r['axis_start'], hex=True).coords[:][0], wkb.loads(r['axis_end'], hex=True).coords[:][0], 
                  r['nx'], r['ny'], r['nz'], r['width'], r['height'], r['length']) for r in beam_records]
    beams = [Beam.Beam(id=r[0], axis=[r[1], r[2]], unit_vector=[float(r[3]), float(r[4]), float(r[5])],width=r[6], height=r[7], length=r[8]) for r in beam_attr]

    #Clean relevant columns on beam table
    roof_db.connect(True)
    roof_db.cursor.execute("update beam set comment = null where comment like 'stage1_%'")
    roof_db.cursor.execute("update beam set roof_tile_id = null")
    roof_db.closeSession()

    #Compare beams-planes
    plane_beam_match = []
    for i, beam in enumerate(beams):
        beam_points = [beam.axis[0], beam.axis[1]]
        beam_center = np.mean(beam_points, axis = 0)
        dist_list = []

        for j, plane in enumerate(plane_params):
            dist_list.append(np.abs(geometry.getPoint2PlaneDistance(beam_center, plane)))

        closest_plane_id = np.argmin(dist_list)
        if dist_list[closest_plane_id] < 0.50:
            plane_beam_match.append([closest_plane_id, i, dist_list[closest_plane_id]])

    #Elemination of irrelevant beams (by direction)  
    for i, plane in enumerate(plane_params):
       plane_beam_match = np. array(plane_beam_match)
       inliers = np.where(plane_beam_match[:,0] == i)[0]
       beam_ids = plane_beam_match[inliers][:,1]
       #candidate_beams = [beam for i, beam in enumerate(dxfBeamAxes) if i in beam_ids]
       
       #Eliminate by max width of cross section (check on distribution)
       candidate_max_widths = np.array([(i, beam.height) for i, beam in enumerate(beams) if i in beam_ids]) # beam.height is 2nd largest size of obb
       beam_size_outliers = candidate_max_widths[np.where(abs(candidate_max_widths[:,1] - np.mean(candidate_max_widths[:,1])) > 2 * np.std(candidate_max_widths[:,1]))][:,0]
       #Eliminate by beam - plane distance
       plane_dist_outliers = beam_ids[np.where(abs(plane_beam_match[inliers][:,2] - np.mean(plane_beam_match[inliers][:,2])) > 2 * np.std(plane_beam_match[inliers][:,2]))]
       
       # If both conditions fits, exxclude beam from list
       size_dist_outliers= np.intersect1d(beam_size_outliers, plane_dist_outliers)
       # Add comment on beam object
       if len(size_dist_outliers) > 0:
           roof_db.connect(True)
           for j, b in enumerate(beams):
               if j in size_dist_outliers:
                   b.comment = "stage1_outlier"
                   roof_db.cursor.execute("update beam set comment = '" + b.comment + "' where id = " + str(b.id))
           roof_db.closeSession()

       beam_ids_after = np.in1d(beam_ids, size_dist_outliers)
       beam_ids_after = np.in1d(beam_ids, size_dist_outliers)
       beam_ids2 = beam_ids[~beam_ids_after]
       candidate_beams = [beam for j, beam in enumerate(beams) if j in beam_ids2]

       #beam_vecs = linesToOrientedUnitVecs(candidate_beams)
       beam_vecs = [ b.unit_vector for b in candidate_beams]

       kmeans = KMeans(n_clusters=4)
       kmeans = kmeans.fit(beam_vecs)
       labels = kmeans.predict(beam_vecs)

       #Get dominant direction as RoofCoverBeam
       km_unique, km_idx, km_counts = np.unique(kmeans.labels_,return_index=True,return_counts=True)
       ref_group_uvec = kmeans.cluster_centers_[np.argmax(km_counts)]

       #TODO loop optimisation
       angles = [geometry.getAngleBetweenVectors(vec,ref_group_uvec) for vec in beam_vecs]
       beam_ids2 = np.where(np.array(angles) <angle_th)[0]
       #beam_ids = np.where(labels==0)[0]

       # angle_th is around 5-10 degree, use very large to skip this thresholding!!

       if angle_th == 360:
           #a special case to exclude dominant direction thresholding!!           
           angles = [geometry.getAngleBetweenVectors(vec,plane[:3]) for vec in beam_vecs]
           beam_ids2_1 = np.where(np.array(angles) < 95)[0]
           beam_ids2_2 = np.where(np.array(angles) > 85)[0]
           beam_ids2 = np.intersect1d(beam_ids2_1, beam_ids2_2)

       write_beams = [beam for j, beam in enumerate(candidate_beams) if j in beam_ids2]
       
       #Update beam-roof_tile relation on db
       roof_db.connect(True)
       for b in write_beams:
           b.roof_tile_id = i + 1 # Plane id on db
           roof_db.cursor.execute("update beam set roof_tile_id = " + str(b.roof_tile_id) + " where id = " + str(b.id))
       roof_db.closeSession()

       saveLines2DXF(write_beams, process_folder, i, i) #Save outliers in the next dxf file

    ignored_beams = [b for b in beams if b.comment == "stage1_outlier"]
    saveLines2DXF(ignored_beams, process_folder, i+1, i+1)
    end = datetime.now()
    print("Roof tile - beam relations defined :\t", (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Roof Tile Transfer")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)