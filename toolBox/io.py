from pathlib import Path
import open3d as o3d
import numpy as np
import laspy

def readPointCloud(path):
    """
    Point cloud reader for PLY/LAS/LAZ formats.
    Returns Open3D PointCloud object
    """
    extension = Path(path).suffix

    if extension in ('.las', '.laz'):
        try:
            coords = laspy.read(path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords.xyz)                       
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

def writePointCloudLAS(point_cloud, out_file_name):
    """
    Point cloud reader for PLY/LAS/LAZ formats.
    Returns Open3D PointCloud object
    """
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(point_cloud, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])
    
    with laspy.open(out_file_name, mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(point_cloud.shape[0], header=header)
        point_record.x = point_cloud[:, 0]
        point_record.y = point_cloud[:, 1]
        point_record.z = point_cloud[:, 2]   
        writer.write_points(point_record)

def writePLY(path, points, normals, colors, labels):
    """
    PLY exporter with Integer Segment IDs.
    Uses float32 for geometry and uint8/ for attributes.
    """
    num_points = points.shape[0]
    
    # Structure: 24 bytes (coords/normals) + 3 bytes (colors) + 4 bytes (int label) = 31 bytes/vert
    dtype = [
        ('coords', 'f4', 3),     # x, y, z
        ('normals', 'f4', 3),    # nx, ny, nz
        ('red', 'u1'),           # r
        ('green', 'u1'),         # g
        ('blue', 'u1'),          # b
        ('segmentid', 'i4')      # i4 = 32-bit signed integer
    ]
    
    vertex_data = np.empty(num_points, dtype=dtype)
    
    vertex_data['coords'] = points.astype(np.float32)
    vertex_data['normals'] = normals.astype(np.float32)
    
    c_array = colors.reshape(-1, 3)
    vertex_data['red'] = c_array[:, 0].astype(np.uint8)
    vertex_data['green'] = c_array[:, 1].astype(np.uint8)
    vertex_data['blue'] = c_array[:, 2].astype(np.uint8)
    
    vertex_data['segmentid'] = labels.reshape(-1).astype(np.int32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int segmentid\n"
        "end_header\n"
    )

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertex_data.tobytes())

def readPLY(path):
    """
    PLY reader.
    Parses the header to find the vertex count, then reads binary data in one block.
    """
    with open(path, 'rb') as f:
        header_lines = []
        num_points = 0
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line.startswith('element vertex'):
                num_points = int(line.split()[-1])
            if line == 'end_header':
                break
        
        # 3x float32 (xyz), 3x float32 (normals), 3x uint8 (rgb), 1x int32 (label)
        # Total bytes per vertex: 12 + 12 + 3 + 4 = 31
        dt = np.dtype([
            ('points', 'f4', (3,)),
            ('normals', 'f4', (3,)),
            ('colors', 'u1', (3,)),
            ('segmentid', 'i4')
        ])
        
        # fromfile for maximum speed
        data = np.fromfile(f, dtype=dt, count=num_points)
        
    return {
        'points': data['points'],
        'normals': data['normals'],
        'colors': data['colors'],
        'segmentid': data['segmentid']
    }
