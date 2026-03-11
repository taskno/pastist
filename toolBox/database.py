import argparse
import yaml
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import toolBox.exchange as exchangeOps
from sklearn.neighbors import KDTree
import open3d as o3d
import ezdxf
import toolBox.geometry as geometryProcess

import roof.Beam as Beam

class modelDatabase:
    def __init__(self, config_file):
        self.db_name = None
        self.conn = None
        self.cursor = None

        self.db_user = None
        self.db_pass = None
        self.db_host = None
        self.db_port = None

        self.db_rcp = None #Roof cover points
        self.db_odm1 = None
        self.db_dxf = None
        
        #Read the config file
        #config_data = yaml.safe_load(config_file)
        with open(config_file, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
   
        if config_data.__contains__('db_user'):
            self.db_user = config_data['db_user']
        if config_data.__contains__('db_pass'):
            self.db_pass = config_data['db_pass']
        if config_data.__contains__('db_host'):
            self.db_host = config_data['db_host']
        if config_data.__contains__('db_port'):
            self.db_port = config_data['db_port']
        if config_data.__contains__('db_name'):
            self.db_name = config_data['db_name']

        if config_data.__contains__('db_rcp'):
            self.db_rcp = config_data['db_rcp']
        if config_data.__contains__('db_odm1'):
            self.db_odm1 = config_data['db_odm1']
        if config_data.__contains__('db_dxf'):
            self.db_dxf = config_data['db_dxf']
    

    def connect(self, defined_db=False):
        if (self.db_user is not None and 
            self.db_pass is not None and
            self.db_host is not None and
            self.db_port is not None):
            
            if defined_db:
                db_name = self.db_name
            else:
                db_name = None
            conn = psycopg2.connect(user = self.db_user, 
                            password = self.db_pass, 
                            host = self.db_host, 
                            port = self.db_port,
                            database = db_name)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            self.conn = conn
            self.cursor = cursor

    def closeSession(self):
        if self.conn:
            self.cursor.close()
            self.conn.close()

    def creeateDB(self):
        self.connect()            
        #Create the database
        sqlCreateDB = "create database " + self.db_name + ";"
        self.cursor.execute(sqlCreateDB)
        self.closeSession()
            
    def createTables(self):
        #Connect to the created db
        self.connect(True)
        #Create postgis extension
        self.cursor.execute("create extension postgis;")
        #Create roof model tables
        self.createBeamTable()
        self.createBeamGroupTable()
        self.createJointTable()
        self.createRafterTable()
        self.createRoofTileTable()
        self.createBeamNewTable()
        self.createJointNewTable()
        self.createClusterTable()
        self.closeSession()

    def createBeamTable(self):
        sqlCreateTable = 'create table beam (' \
            'id serial,'\
            'roof_tile_id int,'\
            'rafter_id int,'\
            'truss_id int,'\
            'group_id int,'\
            'cluster_id int,'\
            'merge_id int,'\
            'nx numeric(40,20),'\
            'ny numeric(40,20),'\
            'nz numeric(40,20),'\
            'width numeric(40,20),'\
            'height numeric(40,20),'\
            'length numeric(40,20),'\
            'comment varchar(200)'\
            ');'

        self.cursor.execute(sqlCreateTable)

        # Geometric columns p1-8 as beam corners
        for i in range(8):
            po_name = i.__str__()
            col_name = "p" + po_name
            sql = "select AddGeometryColumn('beam', '" + col_name + "', 0, 'POINT',3 );"
            self.cursor.execute(sql)
        #beam-axes points in oriented order
        self.cursor.execute("select AddGeometryColumn('beam', 'axis_start', 0, 'POINT',3 );")
        self.cursor.execute("select AddGeometryColumn('beam', 'axis_end', 0, 'POINT',3 );")

    def createBeamGroupTable(self):
        sqlCreateTable = 'create table beam_group (' \
            'id serial,'\
            'name varchar(100),'\
            'nx_avg numeric(40,20),'\
            'ny_avg numeric(40,20),'\
            'nz_avg numeric(40,20),'\
            'width_avg numeric(40,20),'\
            'height_avg numeric(40,20),'\
            'length_avg numeric(40,20),'\
            'plane_a numeric(40,20),'\
            'plane_b numeric(40,20),'\
            'plane_c numeric(40,20),'\
            'plane_d numeric(40,20),'\
            'group_type varchar(100)'\
            ');'

        self.cursor.execute(sqlCreateTable)
    
    def createJointTable(self):
        sqlCreateTable = 'create table joint (' \
            'id serial,'\
            'b1_id int,'\
            'b2_id int,'\
            'b1_position numeric(40,20),'\
            'b2_position numeric(40,20),'\
            'joint_type varchar(200),'\
            'comment varchar(200)'\
            ');'
        self.cursor.execute(sqlCreateTable)
        sql1 = "select AddGeometryColumn('joint', 'p_b1', 0, 'POINT',3 );"
        sql2 = "select AddGeometryColumn('joint', 'p_b2', 0, 'POINT',3 );"
        self.cursor.execute(sql1)
        self.cursor.execute(sql2)

    def createRafterTable(self):
        sqlCreateTable = 'create table rafter (' \
            'id serial,'\
            'joint_id int,'\
            'b1_id int,'\
            'b2_id int,'\
            'plane_a numeric(40,20),'\
            'plane_b numeric(40,20),'\
            'plane_c numeric(40,20),'\
            'plane_d numeric(40,20),'\
            'rafter_type varchar(100),'\
            'truss_type varchar(100)'\
            ');'
        self.cursor.execute(sqlCreateTable)
        sql1 = "select AddGeometryColumn('rafter', 'chull2d', 0, 'POLYGON',2 );"
        sql2 = "select AddGeometryColumn('rafter', 'alphashape2d', 0, 'MULTIPOLYGON',2 );"
        self.cursor.execute(sql1)
        self.cursor.execute(sql2)

    def createRoofTileTable(self):
        sqlCreateTable = 'create table roof_tile (' \
            'id serial,'\
            'plane_a numeric(40,20),'\
            'plane_b numeric(40,20),'\
            'plane_c numeric(40,20),'\
            'plane_d numeric(40,20)'\
            ');'
        self.cursor.execute(sqlCreateTable)
        sql1 = "select AddGeometryColumn('roof_tile', 'chull2d', 0, 'POLYGON',2 );"
        sql2 = "select AddGeometryColumn('roof_tile', 'alphashape2d', 0, 'MULTIPOLYGON',2 );"
        self.cursor.execute(sql1)
        self.cursor.execute(sql2)

    def createBeamNewTable(self):
        #Refined beams stored here
        sqlCreateTable = 'create table beam_new (' \
            'id serial,'\
            'old_id int,' \
            'roof_tile_id int,'\
            'rafter_id int,'\
            'truss_id int,'\
            'group_id int,'\
            'cluster_id int,'\
            'nx numeric(40,20),'\
            'ny numeric(40,20),'\
            'nz numeric(40,20),'\
            'width numeric(40,20),'\
            'height numeric(40,20),'\
            'length numeric(40,20),'\
            'comment varchar(200)'\
            ');'

        self.cursor.execute(sqlCreateTable)

        # Geometric columns p1-8 as beam corners
        for i in range(8):
            po_name = i.__str__()
            col_name = "p" + po_name
            sql = "select AddGeometryColumn('beam_new', '" + col_name + "', 0, 'POINT',3 );"
            self.cursor.execute(sql)
        #beam-axes points in oriented order
        self.cursor.execute("select AddGeometryColumn('beam_new', 'axis_start', 0, 'POINT',3 );")
        self.cursor.execute("select AddGeometryColumn('beam_new', 'axis_end', 0, 'POINT',3 );")
   
    def createJointNewTable(self):
        #Joints after refined beams created
        sqlCreateTable = 'create table joint_new (' \
            'id serial,'\
            'b1_id int,'\
            'b2_id int,'\
            'b1_position numeric(40,20),'\
            'b2_position numeric(40,20),'\
            'joint_type varchar(200),'\
            'comment varchar(200)'\
            ');'
        self.cursor.execute(sqlCreateTable)
        sql1 = "select AddGeometryColumn('joint_new', 'p_b1', 0, 'POINT',3 );"
        sql2 = "select AddGeometryColumn('joint_new', 'p_b2', 0, 'POINT',3 );"
        self.cursor.execute(sql1)
        self.cursor.execute(sql2)

    def createClusterTable(self):
        #Kmeans beam cluster centers
        sqlCreateTable = 'create table cluster (' \
            'id int,'\
            'nx numeric(40,20),'\
            'ny numeric(40,20),'\
            'nz numeric(40,20),'\
            'comment varchar(200)'\
            ');'
        self.cursor.execute(sqlCreateTable)


    def fillTablesFromDXF(self,beams=True, joints=True):
        self.connect(True)
        dxf_dict = exchangeOps.readBeamsDXFOriented(self.db_dxf)
        values = []

        for i in range(len(dxf_dict["Beams"])):
            axis_start = np.asarray(dxf_dict["Beam-Axes"][i].dxf.start.xyz)
            axis_end = np.asarray(dxf_dict["Beam-Axes"][i].dxf.end.xyz)
            faces = exchangeOps.getCuboidFaces(dxf_dict["Beams"][i])

            #Define start(bottom) and end(top) points of cuboid
            face0_mean = np.mean(faces[0], axis =0)
            d1 = face0_mean - axis_start
            if np.sqrt(np.dot(d1,d1)) < 0.00001:
                #face[0] = axis_start, face[5] = axis_end
                cuboid_start = faces[0]
                cuboid_end = faces[5]
            else:
                d2 = face0_mean - axis_end
                if np.sqrt(np.dot(d2,d2)) < 0.00001:
                    cuboid_start = faces[5]
                    cuboid_end = faces[0]
            
            cuboid_pts = [*faces[0], *faces[5]]
            obb = o3d.geometry.OrientedBoundingBox()
            obb = obb.create_from_points(points=o3d.utility.Vector3dVector(cuboid_pts))

            values.append((dxf_dict["Beam-Orientation"][i][0],dxf_dict["Beam-Orientation"][i][1], dxf_dict["Beam-Orientation"][i][2],
                           obb.extent[2],obb.extent[1],obb.extent[0],
                           "'POINT Z (" + str(cuboid_start[0][0]) + " " + str(cuboid_start[0][1]) + " " + str(cuboid_start[0][2]) + ")'::geometry",
                           "'POINT Z (" + str(cuboid_start[1][0]) + " " + str(cuboid_start[1][1]) + " " + str(cuboid_start[1][2]) + ")'::geometry",
                           "'POINT Z (" + str(cuboid_start[2][0]) + " " + str(cuboid_start[2][1]) + " " + str(cuboid_start[2][2]) + ")'::geometry", 
                           "'POINT Z (" + str(cuboid_start[3][0]) + " " + str(cuboid_start[3][1]) + " " + str(cuboid_start[3][2]) + ")'::geometry", 
                           "'POINT Z (" + str(cuboid_end[0][0]) + " " + str(cuboid_end[0][1]) + " " + str(cuboid_end[0][2]) + ")'::geometry",
                           "'POINT Z (" + str(cuboid_end[1][0]) + " " + str(cuboid_end[1][1]) + " " + str(cuboid_end[1][2]) + ")'::geometry",
                           "'POINT Z (" + str(cuboid_end[2][0]) + " " + str(cuboid_end[2][1]) + " " + str(cuboid_end[2][2]) + ")'::geometry",
                           "'POINT Z (" + str(cuboid_end[3][0]) + " " + str(cuboid_end[3][1]) + " " + str(cuboid_end[3][2]) + ")'::geometry",
                           "'POINT Z (" + str(axis_start[0]) + " " + str(axis_start[1]) + " " + str(axis_start[2]) + ")'::geometry",
                           "'POINT Z (" + str(axis_end[0]) + " " + str(axis_end[1]) + " " + str(axis_end[2]) + ")'::geometry"
                           ))

        if self.conn.closed ==1 and self.cursor.closed:
            self.connect(True)
        #Insert to the beam table
        for v in values:
            insert_sql = "insert into beam(nx,ny,nz,width,height,length,p0,p1,p2,p3,p4,p5,p6,p7,axis_start,axis_end) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s )" % v
            self.cursor.execute(insert_sql)

        #Filling the joint table
        values_joint = []
        for i, joint in enumerate(dxf_dict["Joints"]):
            b1_id = None #refers to joint start
            b2_id = None #refers to joint end
            b1_position = None #refers to distance ratio from b1.start range : [0-1]
            b2_position = None #refers to distance ratio from b1.start range : [0-1]
            p_b1 = None #refers to point on beam1
            p_b2 = None #refers to point on beam2

            for b, beam in enumerate(dxf_dict["Beam-Axes"]):
                intersection = ezdxf.math.intersection_line_line_3d((beam.dxf.start, beam.dxf.end),(joint.dxf.start,joint.dxf.end), virtual=True, abs_tol=1e-06)
                if intersection is not None:
                    d1 = geometryProcess.getDistance(intersection, joint.dxf.start)
                    d2 = geometryProcess.getDistance(intersection, joint.dxf.end)           
                    #d = np.min((d1,d2))
                    test1 = geometryProcess.isPointOnLineSegment3D(joint.dxf.start, (beam.dxf.start, beam.dxf.end))
                    test2 = geometryProcess.isPointOnLineSegment3D(joint.dxf.end, (beam.dxf.start, beam.dxf.end))

                    if d1 < 0.000001:
                        if b1_id is None:
                            b1_id = b + 1 # database id =  idx + 1
                            d_bs_int = geometryProcess.getDistance(intersection, beam.dxf.start)
                            d_bs_be = geometryProcess.getDistance(beam.dxf.end, beam.dxf.start)

                            b1_position =  d_bs_int / d_bs_be
                            p_b1 = "'POINT Z (" + str(intersection[0]) + " " + str(intersection[1]) + " " + str(intersection[2]) + ")'::geometry"
                      
                    if d2 < 0.000001:
                        if b2_id is None:
                            b2_id = b + 1 # database id =  idx + 1
                            d_bs_int = geometryProcess.getDistance(intersection, beam.dxf.start)
                            d_bs_be = geometryProcess.getDistance( beam.dxf.end, beam.dxf.start)

                            b2_position =  d_bs_int / d_bs_be
                            p_b2 = "'POINT Z (" + str(intersection[0]) + " " + str(intersection[1]) + " " + str(intersection[2]) + ")'::geometry"
                    
                    if b2_id is not None and b1_id is not None:
                        values = (b1_id, b2_id, b1_position, b2_position, p_b1, p_b2)
                        values_joint.append(values)
                        break

        #Insert to thw joint table
        for v in values_joint:
            insert_sql = "insert into joint(b1_id,b2_id,b1_position,b2_position,p_b1,p_b2) values (%s, %s, %s, %s, %s, %s)" % v
            self.cursor.execute(insert_sql)
        self.closeSession()

    def fillRoofTileTable(self,roof_tiles):
        self.connect(True)
        self.cursor.execute("delete from roof_tile;")
        #TODO check if alphashape polygon before casting to multipolygon (st_multi....)
        for tile in roof_tiles:
            v = (tile.id, tile.plane[0], tile.plane[1], tile.plane[2], tile.plane[3], 
                 "ST_GeomFromText('" + tile.alpha_shape2d.convex_hull.wkt + "',0)", 
                 "st_multi(ST_GeomFromText('" + tile.alpha_shape2d.wkt + "',0))")
            insert_sql = "insert into roof_tile(id, plane_a, plane_b, plane_c, plane_d, chull2d, alphashape2d) values (%s, %s, %s, %s, %s, %s, %s)" % v
            self.cursor.execute(insert_sql)
        self.closeSession()

    def fillBeamNewTable(self, beams):
        values = []

        for b in beams:
            values.append(("null" if b.old_id is None else b.old_id,
                           "'"+ b.comment + "'",
                           "null" if b.roof_tile_id is None else b.roof_tile_id,
                           "null" if b.rafter_id is None else b.rafter_id,
                           "null" if b.group_id is None else b.group_id,
                           b.unit_vector[0], b.unit_vector[1], b.unit_vector[2],
                           b.width, b.height, b.length,
                    "'POINT Z (" + str(b.vertices[0][0]) + " " + str(b.vertices[0][1]) + " " + str(b.vertices[0][2]) + ")'::geometry",
                    "'POINT Z (" + str(b.vertices[1][0]) + " " + str(b.vertices[1][1]) + " " + str(b.vertices[1][2]) + ")'::geometry",
                    "'POINT Z (" + str(b.vertices[2][0]) + " " + str(b.vertices[2][1]) + " " + str(b.vertices[2][2]) + ")'::geometry", 
                    "'POINT Z (" + str(b.vertices[3][0]) + " " + str(b.vertices[3][1]) + " " + str(b.vertices[3][2]) + ")'::geometry", 
                    "'POINT Z (" + str(b.vertices[4][0]) + " " + str(b.vertices[4][1]) + " " + str(b.vertices[4][2]) + ")'::geometry",
                    "'POINT Z (" + str(b.vertices[5][0]) + " " + str(b.vertices[5][1]) + " " + str(b.vertices[5][2]) + ")'::geometry",
                    "'POINT Z (" + str(b.vertices[6][0]) + " " + str(b.vertices[6][1]) + " " + str(b.vertices[6][2]) + ")'::geometry",
                    "'POINT Z (" + str(b.vertices[7][0]) + " " + str(b.vertices[7][1]) + " " + str(b.vertices[7][2]) + ")'::geometry",
                    "'POINT Z (" + str(b.axis[0][0]) + " " + str(b.axis[0][1]) + " " + str(b.axis[0][2]) + ")'::geometry",
                    "'POINT Z (" + str(b.axis[1][0]) + " " + str(b.axis[1][1]) + " " + str(b.axis[1][2]) + ")'::geometry"
                    ))

        if self.conn.closed ==1 and self.cursor.closed:
            self.connect(True)
        #Insert to the beam table
        for v in values:
            insert_sql = "insert into beam_new(old_id,comment,roof_tile_id,rafter_id,group_id,nx,ny,nz,width,height,length,p0,p1,p2,p3,p4,p5,p6,p7,axis_start,axis_end) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s )" % v
            self.cursor.execute(insert_sql)
        self.closeSession()

    def fillClusterTable(self, clusters):
        if self.conn.closed ==1 and self.cursor.closed:
            self.connect(True)
        values = []
        for i,c in enumerate(clusters):
            values.append((i + 1, c[0],c[1],c[2]))

        for v in values:
            insert_sql = "insert into cluster(id, nx, ny, nz) values (%s, %s, %s, %s)" % v
            self.cursor.execute(insert_sql)
        self.closeSession()

    def addBeamGroupTable(self, beamGroup):
        #Insert single BeamGroup object!!!!
        if self.conn.closed ==1 and self.cursor.closed:
            self.connect(True)
        values = (beamGroup.id, "'"+beamGroup.name+"'", 
                  beamGroup.optimal_uvec[0], beamGroup.optimal_uvec[1], beamGroup.optimal_uvec[2],
                  beamGroup.avg_width, beamGroup.avg_height, beamGroup.avg_length,
                  beamGroup.optimal_plane[0], beamGroup.optimal_plane[1], beamGroup.optimal_plane[2], beamGroup.optimal_plane[3])
        insert_sql = "insert into beam_group(id, name, nx_avg, ny_avg, nz_avg, width_avg, height_avg, length_avg, plane_a, plane_b, plane_c, plane_d) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % values
        self.cursor.execute(insert_sql)
        self.closeSession()

    def getBeams(self, columns=["id", "axis_start", "axis_end"], condition = "true;"):

        self.connect(True)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cols = ""
        for col in columns:
            cols += f'b.{col},'
            #cols += col + ",b."
        select_query = "select " + cols.rstrip(cols[-1]) + " from beam b" + " where " + condition
        
        self.cursor.execute(select_query)
        colnames = [d[0] for d in self.cursor.description]
        records = self.cursor.fetchall()
        self.closeSession()
        return records

    def getRoofTiles(self, columns=["id", "plane_a", "plane_b", "plane_c", "plane_d"], condition = "true;"):

        self.connect(True)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cols = ""
        for col in columns:
            cols += f'r.{col},'
        select_query = "select " + cols.rstrip(cols[-1]) + " from roof_tile r" + " where " + condition
        
        self.cursor.execute(select_query)
        colnames = [d[0] for d in self.cursor.description]
        records = self.cursor.fetchall()
        self.closeSession()
        return records
    def getRafters(self, columns=["id", "b1_id", "b2_id", "joint_id", "plane_a", "plane_b", "plane_c", "plane_d", "rafter_type", "chull2d"], condition = "true;"):

        self.connect(True)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cols = ""
        for col in columns:
            cols += f'r.{col},'
        select_query = "select " + cols.rstrip(cols[-1]) + " from rafter r" + " where " + condition
        
        self.cursor.execute(select_query)
        colnames = [d[0] for d in self.cursor.description]
        records = self.cursor.fetchall()
        self.closeSession()
        return records

    def getBeamGroups(self, columns=["id", "name", "nx_avg","ny_avg", "nz_avg","width_avg", "height_avg", "plane_a", "plane_b", "plane_c", "plane_d"], condition = "true;"):

        self.connect(True)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cols = ""
        for col in columns:
            cols += f'r.{col},'
        select_query = "select " + cols.rstrip(cols[-1]) + " from beam_group r" + " where " + condition
        
        self.cursor.execute(select_query)
        colnames = [d[0] for d in self.cursor.description]
        records = self.cursor.fetchall()
        self.closeSession()
        return records


    def getNewBeams(self, columns=["id", "axis_start", "axis_end"], condition = "true;"):

        self.connect(True)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cols = ""
        for col in columns:
            cols += f'b.{col},'
            #cols += col + ",b."
        select_query = "select " + cols.rstrip(cols[-1]) + " from beam_new b" + " where " + condition
        
        self.cursor.execute(select_query)
        colnames = [d[0] for d in self.cursor.description]
        records = self.cursor.fetchall()
        self.closeSession()
        return records

    def updataNewBeamClusters(self):
        self.connect(True)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        select_query = "select roof_tile_id, cluster_id, count(id) as cnt from beam where roof_tile_id is not null group by roof_tile_id, cluster_id;"
        self.cursor.execute(select_query)
        records = self.cursor.fetchall()

        rt_cluster_match = np.array([(r['roof_tile_id'], r['cluster_id']) for r in records])

        #TODO: one roof_tile -> multiple cluster_id case (fix this case later)
        for match in rt_cluster_match:
            roof_tile_id = match[0]
            cluster_id = match[1]

            sql_update = "update beam_new set cluster_id = " + str(cluster_id) + " where roof_tile_id = " + str(roof_tile_id)
            self.cursor.execute(sql_update)
        self.closeSession()

