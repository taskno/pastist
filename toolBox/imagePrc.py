import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage import feature
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
import toolBox.geometry as geometry

def getImageFromPoints(pts_2D, scale=1):
    
    x_min = np.min(pts_2D[:,0])
    y_min = np.min(pts_2D[:,1])
    x_max = np.max(pts_2D[:,0])
    y_max = np.max(pts_2D[:,1])

    col_size = x_max - x_min
    row_size = y_max - y_min
    
    pixel_size = 50 if np.max(np.array((col_size, row_size))) > 4 else 100
    pixel_size *= scale
    
    cols = round(col_size * pixel_size)
    rows = round(row_size * pixel_size)
      
    x_nrm  =  pts_2D[:,0] - np.min(pts_2D[:,0])
    y_nrm  =  pts_2D[:,1] - np.min(pts_2D[:,1])
        
    matrix, y_, x_ = np.histogram2d(-y_nrm, x_nrm, bins=[rows,cols])

    return matrix, pixel_size, (x_min, y_min, x_max, y_max)

def cannyEdges(image, sigma = 3):
    edges = feature.canny(image, sigma = sigma)
    edges = edges.astype(np.uint8)
    edges*=255
    return edges

def lineFit(image, show_result= False):
    # Classic straight-line Hough transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)
    
    if show_result:
        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()
        
        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()
        
        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                  np.rad2deg(theta[-1] + angle_step),
                  d[-1] + d_step, d[0] - d_step]
        ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')
        
        ax[2].imshow(image, cmap=cm.gray)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

    lines = [] # reference pixel coordinates (i,j) and slope
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=5, min_angle= 1)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope=np.tan(angle + np.pi/2)
        lines.append((x0, y0, slope))
        
        if show_result:
            ax[2].axline((x0, y0), slope=slope)        
        
    if show_result:     
        plt.tight_layout()
        plt.show()
    return lines

def image2CartesianCoordinates(image_coor, X_min, Y_max, pixel_size):
    X = X_min + image_coor[0] / float(pixel_size)
    Y = Y_max - image_coor[1] / float(pixel_size)
    return (X, Y)

def cartesian2ImageCoordinates(coor, X_min, Y_max, pixel_size):
    x = int((coor[0] - X_min) * pixel_size)
    y = int((Y_max - coor[1]) * pixel_size)
    return (x, y)

def getHoughLinesFrom2DPts(pts_2D):
    matrix0, pixel_size, extend = getImageFromPoints(pts_2D)
    matrix = matrix0 <= 0

    edges = cannyEdges(matrix)
    lines = lineFit(edges,show_result=False)

    line_eq = []
    for l in lines:
        x, y = image2CartesianCoordinates((l[0],l[1]), extend[0], extend[3], pixel_size)
        m = -l[2]
        b = y - m * x
        line_eq.append((m, b))

    return line_eq

def getLineMatchRatio(img, lines):
    mask = np.zeros((img.shape[0],img.shape[1],1), np.uint8)

    vertices = []
    for l in lines:
        vertices.append((l[0],l[1]))
        vertices.append((l[2],l[3]))

    hull = cv2.convexHull(np.array(vertices,dtype='float32'))
    hull = [np.array(hull).reshape((-1,1,2)).astype(np.int32)]

    cv2.drawContours(mask, contours=hull, 
                     contourIdx = 0,                     
                     color=(255), thickness=-1)

    intersection = cv2.bitwise_and(img, mask)

    mask_cnt = cv2.countNonZero(mask)
    logic_cnt = cv2.countNonZero(intersection)

    ratio = logic_cnt / mask_cnt
    return ratio

def getLineSegments(image):
    #blur_img = cv2.GaussianBlur(image, (3,3), 0)    
    image_bgr = cv2.merge((image,image,image))
    image_bgr = np.copy(image)
  
    fld = cv2.ximgproc.createFastLineDetector(length_threshold = 5, do_merge=True)
    lines = fld.detect(image)# #blur_img
    result_img = fld.drawSegments(image_bgr,lines, False, (0,0,255), 2)
    return lines

def getRandomColors(nr_colors, eight_bit = False):
    colors = [[1.0, 1.0, 0], # Yellow
    [0.0, 1.0, 1.0], # Turquoise
    [0.0, 1.0, 0.0], # Green
    [1.0, 0.0, 0.0], # Red
    [0.0, 0.0, 1.0], # Blue
    [0.5, 0.0, 1.0], # color6
    [0.1, 0.5, 0.0] # color7
    ]

    if nr_colors > 7:
        import random
        for i in range(nr_colors - 7):
            random.uniform(0, 1)
            colors.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

    colors = np.array(colors)
    if eight_bit:
        colors *= 255
        colors = colors.astype(int)
    return colors

def linesToRects(image, lines, show_results = False, same_side_th=3, opposite_side_th=20, min_ratio=0.8):

    lines = lines.astype(int)
    line_lengths = np.array([ geometry.getDistance( (l[0][0], l[0][1]),(l[0][2], l[0][3])) for l in lines])
    line_vecs = np.array([ geometry.getUnitVector(np.array(((l[0][2], l[0][3]))) - 
                                                  np.array(((l[0][0], l[0][1])))) for l in lines ])
    #angles_vertical = [geometry.getAngleBetweenVectors((0.,1.), np.abs(v))  for v in line_vecs]
    line_equations = np.array([geometry.getLineEquation2D( (l[0][0], l[0][1]),
                                                          (l[0][2], l[0][3])) for l in lines])

    #Match ratio test
    #l1 = lines[9][0]
    #l2 = lines[1][0]
    #lines_test = [l1,l2]
    #getLineMatchRatio(image, lines_test)

    # Check vertical lines first
    angles_2 = np.array([ 90. if eq[0] is None else np.arctan(eq[0]) * 180. /np.pi for eq in line_equations])
    angle_ = angles_2[..., np.newaxis]

    if len(lines)<5:
        kmeans = KMeans(n_clusters=len(lines))
        kmeans = kmeans.fit(angle_)
        cluster = kmeans.labels_
    else:
        kmeans = KMeans(n_clusters=5)
        kmeans = kmeans.fit(angle_)
        cluster = kmeans.labels_

    centers = 90 - np.abs(kmeans.cluster_centers_)
    merge_idx = np.argsort(centers.flatten())[:2]

    cluster[np.where(cluster == np.max(merge_idx))] = np.min(merge_idx)
    colors = getRandomColors(max(cluster)+1,True)
    
    dist_inliers = np.where(line_lengths >10)[0]
    if show_results:
        image_bgr = cv2.merge((image,image,image))
        for i,line in enumerate(lines):
            if i in dist_inliers:
                l = line[0]
                x1, y1, x2, y2 = l
                color = colors[cluster[i]]
                result_img2 = cv2.arrowedLine(image_bgr, (x1,y1), (x2,y2), (int(color[0]), int(color[1]), int(color[2])), 2)
        
        cv2.imshow('Lines', result_img2)
        cv2.waitKey()

    clusters = np.unique(cluster)

    rect_multi_points = []

    for cls in clusters:
        cls_idx = np.argwhere(cluster == cls)
        dist_idx = np.argwhere(line_lengths > 10.)
        idx_inter = np.intersect1d(cls_idx, dist_idx)
        idx = idx_inter[np.argsort(-line_lengths[idx_inter])] #id of all lines in cluster (except outlier small ones)
        idx_status = np.zeros(idx.shape, dtype = int) # shows processed/non-processed
     
        lines_cls = lines[idx]
        line_lengths_cls = line_lengths[idx]
        line_vecs_cls = line_vecs[idx] # opposite directions are candidate to be opposite side

        for j,id in enumerate(idx):
            if idx_status[j] == 0:
                line_ref_idx = j
                idx_status[j] = 1
            else:
                continue

            line_ref = lines[idx[line_ref_idx]][0]

            distances = np.array([ geometry.getPoint2LineSegmentDistance2D((line_ref[:2], line_ref[2:]),
                                                                  (np.mean((line[0][0], line[0][2])) , 
                                                                   np.mean((line[0][1], line[0][3])))) for line in lines_cls ])
            
            if show_results:
                test_img = cv2.merge((image,image,image))
                test_img = cv2.arrowedLine(test_img, (line_ref[0],line_ref[1]), (line_ref[2],line_ref[3]), 
                                           (int(color[0]), int(color[1]),int(color[2])), 2)
                for i, line in enumerate(lines_cls):
                    x = np.mean((line[0][0], line[0][2]))
                    y = np.mean((line[0][1], line[0][3]))
                    test_img = cv2.arrowedLine(test_img, (line[0][0],line[0][1]),
                                              (line[0][2],line[0][3]), (0,0,255), 2)
                    cv2.putText(test_img,str(distances[i]),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
                cv2.imshow("Distances", test_img)
                cv2.waitKey()
            
            sides = np.array([1 if np.dot(vec, line_vecs_cls[line_ref_idx])>=0 else 0  
                              for vec in line_vecs_cls]) # 1: same direction 0: opposite direction
            ratios = np.array([0 if sides[i] == 1 or idx_status[i] == 1 else getLineMatchRatio(image, (line_ref,line[0])) 
                               for i, line in enumerate(lines_cls) ])
                  
            opposite_idx1 = np.argwhere(ratios>min_ratio)
            opposite_idx2 = np.argwhere(abs(distances)<opposite_side_th)
            opposite_idx = np.intersect1d(opposite_idx1, opposite_idx2)
            opposite_idx = opposite_idx[:, np.newaxis]

            if len(opposite_idx) == 0:
                continue

            idx_status[opposite_idx] = 1

            oppo_lines = lines_cls[opposite_idx]
            oppo_lenghts = line_lengths_cls[opposite_idx]
            
            oppo_ref_line = oppo_lines[np.argmax(oppo_lenghts)][0][0]

            same_orient_idx = np.argwhere(sides == 1)
            closed_idx = np.argwhere(abs(distances) < same_side_th)
            
            same_side_idx = np.intersect1d(same_orient_idx, closed_idx)

            side_candidate_lines = lines_cls[same_side_idx]

            ratios2 = np.array([ getLineMatchRatio(image, (line_ref,oppo_ref_line,line[0])) 
                               for i, line in enumerate(side_candidate_lines) ])

            same_side_inlier_idx = same_side_idx[np.argwhere(ratios2>min_ratio)]
            same_side_inlier_lines = lines_cls[same_side_inlier_idx]
            idx_status[same_side_inlier_idx] = 1

            
            side1 = same_side_inlier_lines.flatten() #this includes line_ref
            side2 = oppo_lines.flatten()

            all_pts = np.hstack((side1, side2))

            multi_pts = np.reshape(all_pts, (int(len(all_pts)/2),2))
            rect_multi_points.append(multi_pts)

            if show_results:
                oppo_img = cv2.merge((image,image,image))
                oppo_img = cv2.arrowedLine(oppo_img, (line_ref[0],line_ref[1]), (line_ref[2],line_ref[3]), 
                                           (255,0,0), 2)
                for i,line in enumerate(oppo_lines):
                    l = line[0][0]
                    x1, y1, x2, y2 = l
                    color = (0,255,0)
                    oppo_img = cv2.arrowedLine(oppo_img, (x1,y1), (x2,y2), color, 2)
                
                cv2.imshow("Opposite", oppo_img)
                cv2.waitKey()
    boxes = [ cv2.boxPoints(cv2.minAreaRect(pts)) for pts in rect_multi_points]
    rect_axes = rectangles2Lines(boxes)

    #box singularization
    merge_list = []
    for i,box in enumerate(boxes):

        cond1 = True
        if len(merge_list):
            if i in np.hstack(merge_list):
                cond1 = False

        if cond1:
            matches = []
            ref_img = np.zeros((image.shape[0],image.shape[1],1), np.uint8)
            box = box.astype(np.int32)
            cv2.drawContours(ref_img, [box],0,255,-1)
            
            ref_line = rect_axes[i]
            ref_vec = geometry.getUnitVector(ref_line[1] - ref_line[0])
            
            for j,box2 in enumerate(boxes):
                cond2 = True
                if len(merge_list):
                    if j in np.hstack(merge_list):
                        cond2 = False

                if j>i and cond2:
                    trg_img = np.zeros((image.shape[0],image.shape[1],1), np.uint8)
                    box2 = box2.astype(np.int32)
                    cv2.drawContours(trg_img, [box2],0,255,-1)
                    
                    trg_line = rect_axes[j]
                    trg_vec = geometry.getUnitVector(trg_line[1] - trg_line[0])
            
                    alpha = geometry.getAngleBetweenVectors(ref_vec, trg_vec)
            
                    if alpha < 5.:
                        #Check intersection
                        inter_image = cv2.bitwise_and(ref_img, trg_img)
                        inter_cnt = cv2.countNonZero(inter_image)
                        if inter_cnt >10:
                            if len(matches):
                                matches.append(j)
                            else:
                                matches.append(i)
                                matches.append(j)
            if len(matches):
                merge_list.append(np.array(matches))

    merge_img = cv2.merge((image,image,image))

    merged_boxes = []
    merged_mpts = []
    for idx in merge_list:
        merge_mask = np.zeros((image.shape[0],image.shape[1],1), np.uint8)
        merge_mpts = []
        for id in idx:
            box = boxes[id].astype(np.int32)
            [merge_mpts.append(p) for p in box]
            cv2.drawContours(merge_mask, [box],0,255,-1)
            cv2.drawContours(merge_img, [box],0,(255,0,0),-1)
        mask_inter = cv2.bitwise_and(image, merge_mask)
       
        new_box = cv2.boxPoints(cv2.minAreaRect(np.array(merge_mpts)))
        merged_boxes.append(new_box)
        merged_mpts.append(merge_mpts)

    outliers = np.hstack(merge_list) if len(merge_list) else np.empty(shape=[0, 1],dtype=np.uint8)
    if len(merge_list):
        rect_multi_points2 = [r for i,r in enumerate(rect_multi_points) if i not in outliers]
        boxes2 = [b for i,b in enumerate(boxes) if i not in outliers]

        for i, mpts in enumerate(merged_mpts):
            rect_multi_points2.append(np.array(mpts))
            boxes2.append(np.array(merged_boxes[i]))

        rect_multi_points = rect_multi_points2
        boxes = boxes2
    return rect_multi_points, boxes

def rectangles2Lines(rectangles):
    lines = []
    for r in rectangles:
        d = np.array([geometry.getDistance(v, r[0]) for i,v in enumerate(r) if i >0])
        idx = np.array([0,1,2,3])
        v1 = np.mean((r[0], r[np.argmin(d)+1]), axis = 0)
        opposites = [i for i in idx if i not in (0, np.argmin(d)+1)]
        v2 = np.mean((r[opposites[0]], r[opposites[1]]), axis = 0)
        lines.append(np.array((v1,v2)))
    return np.array(lines)

def getImageRectListOverlapRatio(img, group_rects_image_cs):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)

    mask = np.zeros((img.shape[0],img.shape[1],1), np.uint8)
    for i,gr in enumerate(group_rects_image_cs):      
        r = cv2.minAreaRect(np.array(group_rects_image_cs[i]))
        b = cv2.boxPoints(r)
        b = b.astype(np.int32)
        cv2.drawContours(mask, contours=[b], 
                     contourIdx = 0,
                     color=(255), thickness=-1)
    overlap_img = cv2.bitwise_and(img, mask)
    mod_cnt = cv2.countNonZero(mask)
    overlap_cnt = cv2.countNonZero(overlap_img)

    if mod_cnt == 0:
        return 0
    else:
        overlap_ratio = float(overlap_cnt) / float(mod_cnt)
        return overlap_ratio

def getMBRStats(img):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)

    cnt = np.argwhere(img>0.)
    cnt = np.vstack((cnt.transpose()[1], cnt.transpose()[0])).transpose()

    mbr = np.zeros((img.shape[0],img.shape[1],1), np.uint8)  
    r = cv2.minAreaRect(np.array(cnt))
    b = cv2.boxPoints(r)
    #b = np.int0(b)

    mbr_size = r[1]
    if min(mbr_size) == 0:
        return 0
    else:
        overlap_ratio = float(len(cnt)) / float( mbr_size[0] * mbr_size[1])
    return {"mbr_pixels": b, "mbr_size": mbr_size, "fArea": overlap_ratio}