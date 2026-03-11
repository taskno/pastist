import numpy as np
import copy

#Vector operations
def getUnitVector(v):
    v = np.asarray(v)
    return v / (np.sqrt(np.dot(v,v)))

def getAngleBetweenVectors(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    dot_val = np.clip(np.dot(v1,v2), -1, 1)
    return np.arccos(dot_val) * 180. / np.pi

def getBisectorVector(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    diff = v1 - v2
    theta = getAngleBetweenVectors(v1, v2) * np.pi /180
    bisector = diff / np.cos(theta / 2.)
    return getUnitVector(bisector)

def orientNormalVector(point, normal, ref_point, towards_ref=True):
    nrm = copy.copy(normal)
    if towards_ref:
        ref_vector = ref_point - point
    else:
        ref_vector = point - ref_point
    if np.dot(ref_vector, nrm) < 0.:
        nrm *= -1
    return nrm

def rotateVector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)    
    # Rodrigues rotation
    rotated_vector = vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle) + axis * np.dot(axis, vector) * (1 - np.cos(angle))   
    return rotated_vector

#Plane-Point operations
def getPlane2PlaneDistance(p1, p2):
    #Parallel plane distance, assumes n1 = n2
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return np.abs(p2[3] - p1[3]) / (np.sqrt(np.dot(p1[:3],p2[:3])))
    
def getPoint2PlaneDistance(point, plane):
    point = np.asarray(point)
    plane = np.asarray(plane)
    point = np.append(point, 1)
    return np.dot(plane, point)

def getPoints2PlaneDistances(points, plane):
    points = np.array(points)
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    plane = np.array(plane)
    return np.dot(points, plane)

def project3DPointToPlane(point, plane):
    point = np.asarray(point)
    plane = np.asarray(plane)
    d = getPoint2PlaneDistance(point, plane)
    return point - d * plane[:3]

def project3DPointToPlane2D(point, plane):
    point = np.asarray(point)
    plane = np.asarray(plane)
    #Projection of origin (0,0,0)
    p_o = project3DPointToPlane((0.,0.,0.), plane)    
    #Define x axis of 2D coordinate system
    u = project3DPointToPlane((1.,0.,0.), plane)   
    #Get projection of x axis
    u = getUnitVector(u - p_o)
    #y axis is orthogonal to x and normal vector of plane
    v = np.cross(plane[:3], u)
    #Get 2D coordinates along x and y axes
    x = np.dot((point - p_o), u)
    y = np.dot((point - p_o), v)
    #Test for back projection
    #s = np.dot(plane[:3], (point - p_o))
    #Prj3D = p_o + x * u + y * v + s * plane[:3]
    return np.asarray((x,y))

def project3DPointsToPlane2D(points, plane):
    points = np.asarray(points)
    plane = np.asarray(plane)
    #Projection of origin (0,0,0)
    p_o = project3DPointToPlane((0.,0.,0.), plane)   
    #Define x axis of 2D coordinate system
    u = project3DPointToPlane((1.,0.,0.), plane)    
    #Get projection of x axis
    u = getUnitVector(u - p_o)
    #y axis is orthogonal to x and normal vector of plane
    v = np.cross(plane[:3], u)
    #Get 2D coordinates along x and y axes
    pts_n = points - p_o
    x = np.dot(pts_n, u)
    y = np.dot(pts_n, v)
    #Test for back projection
    #s = np.dot(plane[:3], (point - p_o))
    #Prj3D = p_o + x * u + y * v + s * plane[:3]
    return np.transpose(np.vstack((x,y)))

def reproject2DPointToPlane3D(point, plane):
    point = np.append(np.asarray(point), 1.)
    plane = np.asarray(plane)
    #Projection of origin (0,0,0)
    p_o = project3DPointToPlane((0.,0.,0.), plane)   
    #Define x axis of 2D coordinate system
    u = project3DPointToPlane((1.,0.,0.), plane)   
    #Get projection of x axis
    u = getUnitVector(u - p_o)
    #y axis is orthogonal to x and normal vector of plane
    v = np.cross(plane[:3], u)
    #3D point on plane
    P = p_o + point[0] * u + point[1] * v
    return P

def getPlaneLS(p):
        p_mean = np.mean(p, axis=0)
        p_nrm = p - p_mean
        N = np.dot(p_nrm.T, p_nrm)

        ew, ev = np.linalg.eig(N)
        ew_sorted = np.sort(ew) #eigen values
        ev_sorted = ev[:, ew.argsort()] #eigen vectors

        n = np.asarray([ev_sorted[:, 0]]) # n = [a, b, c]             
        d = - np.dot(n[0],  p_mean) / np.linalg.norm(n[0])

        p0 = -d * n
        error = np.dot((p - p0), ev_sorted[:, 0])
        err =  np.abs(error)
        rmse = np.sqrt(np.mean(err**2))

        plane_params = np.append(n[0], (d,rmse))
        return [plane_params, ev_sorted, ew_sorted]

def getParallelPlanes(plane, d):
    #Plane as (a,b,c,d)
    plane = np.asarray(plane)
    val = np.sqrt(np.dot(plane[:3],plane[:3])) * d
    d1 = val + plane[3]
    d2 = -val + plane[3]
    return((np.append(plane[:3], d1), np.append(plane[:3], d2)))

#2D Line operations
def getLineEquation2D(p1,p2):
    # Line model: y = m*x + b
    # p1 = (x1,y1), p2 = (x2,y2)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    if p1[0] != p2[0]:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - (m * p1[0])
        return (m, b)
    else:
        return (None, None)

def getPointOnLine2D(line, point):
    #line = (m, b) where y = m*x + b
    return (point[0], (line[0] * point[0]) + line[1])

def projectPointToLine2D(line, point):
    # Line model: y = m*x + b
    # point = (x1,y1)
    m2 = -1. / line[0]
    b2 = point[1] - point[0] * m2
    x = (b2 - line[1]) / (line[0] - m2)
    y = line[0] * x + line[1]
    return (x,y)

def getAngleBetweenLines2D(line1, line2):
    return np.arctan( (line1[0] - line2[0]) / (1 + (line1[0] * line2[0]))) * 180 /np.pi

def getPoint2LineDistance(line, point):
    # Linear model is y = mx + b
    a = line[0] # m
    b = -1.
    c = line[1]
    line = np.asarray((a,b,c))
    point = np.asarray(point)
    dist = abs(np.dot(line[:2], point) + c) /  (np.sqrt(np.dot(line[:2],line[:2])))
    return dist

def getPoint2LineSegmentDistance2D(line, point):
    #line = (p1,p2) form
    d=np.cross(line[1]-line[0],point-line[0])/np.linalg.norm(line[1]-line[0])
    return d

def getLineIntersection(l1, l2):
    x = (l2[1] - l1[1]) / (l1[0] - l2[0])
    y = l1[0] * x + l1[1]
    return np.asarray((x,y))

def getParallelLines(l, d):
    #Line as (m,b) where y = mx+b
    l = np.asarray(l)
    val = np.sqrt(1 + l[0]**2) * d
    b1 = val + l[1]
    b2 = -val + l[1]
    return((l[0], b1), (l[0], b2))

def isPointOnLineSegment(p, l):
    #2D line segment- point check
    # line segment (start(x,y), end(x,y))

    d_sp = np.sqrt((l[0][0] - p[0])**2 + (l[0][1] - p[1])**2)
    d_ep = np.sqrt((l[1][0] - p[0])**2 + (l[1][1] - p[1])**2)

    d_se = np.sqrt((l[0][0] - l[1][0])**2 + (l[0][1] - l[1][1])**2)

    if d_sp + d_ep == d_se:
        return True
    else:
        return False

def fitLine3D(points):
    points = np.asarray(points)
    pts_mean = points.mean(axis=0)
    uu, dd, vv = np.linalg.svd(points - pts_mean)
    return(pts_mean, vv[0])


def projectPoint2Vector3D(v_start, v_end, p):
    v_start = np.asarray(v_start)
    v_end = np.asarray(v_end)
    p = np.asarray(p)
    u = (v_end - v_start) / getDistance(v_end, v_start)
    v = p - v_start
    t = np.dot(v, u)
    p_proj = v_start + t * u
    return p_proj

def getPoint2VectorDistance3D(v_start, v_end, p):
    v_start = np.asarray(v_start)
    v_end = np.asarray(v_end)
    p = np.asarray(p)
    u = (v_end - v_start) / getDistance(v_end, v_start)
    v = p - v_start
    t = np.dot(v, u)
    p_proj = v_start + t * u
    dist = getDistance(p, p_proj)
    return dist

def getDistance(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    d = p1 - p2
    d = np.sqrt(np.dot(d,d))
    return d

def getSignedDistance(p0, p1, p):
    p0 = np.asarray(p0).flatten() #start point of ref vector
    p1 = np.asarray(p1).flatten() #end point if ref vector
    p = np.asarray(p).flatten() # search point to compute signed distance to p0
    d = np.dot(p - p0, p1 - p0)
    len_i = getDistance(p0, p1)
    d_signed = d / len_i
    return d_signed

def intersectionLinePlane3D(plane, p0, p1):
    plane = np.asarray(plane)
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    n = plane[:3]
    v0 = project3DPointToPlane(p0,plane) # any point on plane   
    w = p0 - v0
    u = p1 - p0
    N = -np.dot(n, w)
    D = np.dot(n, u)
    sI = N / D
    intersection = p0+ sI*u
    return intersection

def intersection_point(plane1_params, plane2_params, plane3_params):
    def plane_params_to_vector(plane_params):
        # Given the coefficients of the plane equation ax + by + cz + d = 0
        # Returns the normal vector and a point on the plane
        a, b, c, d = plane_params
        normal_vector = np.array([a, b, c])
        point_on_plane = -d * normal_vector / np.linalg.norm(normal_vector)**2
        return normal_vector, point_on_plane
    # Convert plane parameters to normal vectors and points
    plane1 = plane_params_to_vector(plane1_params)
    plane2 = plane_params_to_vector(plane2_params)
    plane3 = plane_params_to_vector(plane3_params)

    # Extracting normal vectors and points from the planes
    normal1, point1 = plane1
    normal2, point2 = plane2
    normal3, point3 = plane3

    # Forming coefficient matrix and constants vector for the system of equations
    A = np.vstack((normal1, normal2, normal3))
    b = np.array([np.dot(normal1, point1), np.dot(normal2, point2), np.dot(normal3, point3)])

    # Solving the system of equations
    try:
        intersection = np.linalg.solve(A, b)
        return intersection
    except np.linalg.LinAlgError:
        # If the system is singular (planes are parallel or coincident), return None
        return None

def get_segment_to_segment_connector(p1, p2, p3, p4):
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = np.dot(u.T, u).flatten()[0] # always >= 0
    b = np.dot(u.T, v).flatten()[0]
    c = np.dot(v.T, v).flatten()[0] # always >= 0
    d = np.dot(u.T, w).flatten()[0]
    e = np.dot(v.T, w).flatten()[0]

    D = a * c - b * b # always >= 0
    sc=sN=sD = D # sc = sN / sD, default sD = D >= 0
    tc=tN=tD = D # tc = tN / tD, default tD = D >= 0

    # compute the line parameters of the two closest points
    if np.isclose(D,0,1.e-8): # D is close to 0 -> the lines are almost parallel
        sN = 0.0 # force using point P0 on segment S1
        sD = 1.0 # to prevent possible division by 0.0 later
        tN = e
        tD = c
    else: # get the closest points on the infinite lines
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if (sN < 0.0): # sc < 0 = > the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif (sN > sD) : # sc > 1  = > the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c

    if (tN < 0.0): # tc < 0 = > the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if (-d < 0.0):
            sN = 0.0
        elif (-d > a):
            sN = sD
        else:
            sN = -d
            sD = a

    elif (tN > tD): # tc > 1  = > the t=1 edge is visible
        tN = tD
        # recompute sc for this edge
        if ((-d + b) < 0.0):
            sN = 0
        elif ((-d + b) > a):
            sN = sD
        else:
            sN = (-d +  b)
            sD = a

    # finally do the division to get sc and tc
    sc = 0.0 if np.isclose(abs(sN), 0, 1.e-8) else sN / sD
    tc = 0.0 if np.isclose(abs(tN), 0, 1.e-8) else tN / tD

    Pa = p1 + sc * u
    Pb = p3 + tc * v

    # get the difference of the two closest points Vector
    dP = w + (sc * u) - (tc * v) # =  S1(sc) - S2(tc)
    min_dist = np.linalg.norm(dP)  # calculate the closest distance

    return (Pa, Pb)

