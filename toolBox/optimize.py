import numpy as np
import copy
from sklearn.decomposition import PCA
from scipy  import optimize
from scipy.optimize import least_squares

import open3d as o3d
import toolBox.geometry as geometry
import roof.Beam as Beam
import sklearn
#from sklearn import metrics
from sklearn import linear_model

#Fitting operations
def fitRectangle2D(points, search_rect = None):
    def rectangleSSD(params, points):
        #Sum of squared distance of points to one of closest 4 lines of the rectangle
        a,b,c,d,e = params
        x = points[:,0]
        y = points[:,1]
        denom = (b**2 + 1)
        d1 = np.square(y - a - b * x) / denom
        d2 = np.square(y - c - b * x) / denom
        d3 = np.square(b*(y - d) + x) / denom
        d4 = np.square(b*(y - e) + x) / denom
        d = np.min((d1,d2,d3,d4), axis = 0)
        return np.sum(d)
    
    def getStartingParameters(points):
        x_c = (np.max(points[:,0]) - np.min(points[:,0])) /2 + np.min(points[:,0])
        y_c = (np.max(points[:,1]) - np.min(points[:,1])) /2 + np.min(points[:,1])

        x_dif = points[:,0] - x_c
        y_dif = points[:,1] - y_c

        theta = np.arctan(y_dif / x_dif) *180 / np.pi
        r = np.sqrt(x_dif **2 + y_dif**2)

    #getStartingParameters(points)
    if search_rect is None:
        pca = PCA(n_components=2).fit(points)
        #Starting parameters of the rectangle from PCA
        b = pca.components_[0][1] / pca.components_[0][0] #slope of the first vector
        a = points[np.argmax(points[:,0])][1]- (b * np.max(points[:,0]))
        c = points[np.argmin(points[:,0])][1]- (b * np.min(points[:,0]))
        d = np.max(points[:,1]) + (points[np.argmax(points[:,1])][0] / b)
        e = np.min(points[:,1]) + (points[np.argmin(points[:,1])][0] / b)
        
        test_parameters = [a,b,c,d,e]   
        test_result = rectangleSSD(test_parameters, points)

    else:
        dist = np.array([geometry.getDistance(p, p[0]) for i,p in enumerate(search_rect)])

        idx = np.argsort(dist)
        v1 = search_rect[idx[0]]
        v2 = search_rect[idx[1]]
        v4 = search_rect[idx[2]]
        v3 = search_rect[idx[3]]

        b, a = geometry.getLineEquation2D(v2,v1)
        c = v4[1] - b * v4[0] #v3 or v4
        d = v1[1] + v1[0] / b #v1 or v4
        e = v3[1] + v3[0] / b #v2 or v3

        test_parameters = [a,b,c,d,e]
        test_result = rectangleSSD(test_parameters, points)

    #Minimization of the sum of squared distances
    #best_parameters2 = optimize.minimize(rectangleSSD, test_parameters, args=points, method="SLSQP")  
    best_parameters = least_squares(rectangleSSD, test_parameters, args=([points]))

    #Line parameters of the best fitting rectangle
    a_1,b_1,c_1,d_1,e_1 = best_parameters.x
    
    line1 = (b_1, a_1)
    line2 = (b_1, c_1)
    line3 = (-1./b_1, d_1)
    line4 = (-1./b_1, e_1)
    
    #Corner points of the rectangle as intersection of perpendicular lines
    p1 = geometry.getLineIntersection(line1,line3)
    p2 = geometry.getLineIntersection(line1,line4)
    p3 = geometry.getLineIntersection(line2,line4)
    p4 = geometry.getLineIntersection(line2,line3)
    
    rectangle_points = (p1,p2,p3,p4)
    fitting_success = best_parameters.success
    rmse = np.sqrt(best_parameters.fun)

    return rectangle_points, fitting_success, rmse

#Beam search based on template

def nLineFitRansac(X, y):
    MIN_SAMPLES = 300
   
    idx = 0
    segments = np.empty(len(X), dtype = int)
    segments.fill(-1)

    # models are based on y= mX + b  --> (m, b)
    models = []
 
    while len(X) > MIN_SAMPLES:
        
        #ransac = sklearn.linear_model.RANSACRegressor(residual_threshold = 0.045)#, max_skips=500)   #min_samples=3
        ransac = linear_model.RANSACRegressor(residual_threshold = 0.15)#, max_skips=500)   #min_samples=3
        res = ransac.fit(X, y)
        count1 = np.count_nonzero(ransac.inlier_mask_)

        X2 = X.reshape(1,len(y))[0]
        y2 = y.reshape(-1,1)
        
        ransac2 = sklearn.linear_model.RANSACRegressor(residual_threshold = 0.15)#, max_skips=500) #loss='squared_loss'
        res2 = ransac2.fit(y2, X2)
        count2 = np.count_nonzero(ransac2.inlier_mask_)

        #Find best model
        RANSACmodel = 1
        inlier_mask = np.array(ransac.inlier_mask_)
        if count2 > count1:
            RANSACmodel = 2
            inlier_mask = np.array(ransac2.inlier_mask_)

        if idx == 0:
           trues = np.where(inlier_mask == True)[0]
           falses = np.where(inlier_mask == False)[0]
           segments[trues] = int(idx)
           nodes,  counts = np.unique(segments,  return_counts=True)

        else:
            tt = np.where(inlier_mask == True)[0]
            trues = falses[tt]
            ff = np.where(inlier_mask == False)[0]
            falses = falses[ff]
            segments[trues] = int(idx)
            nodes,  counts = np.unique(segments,  return_counts=True)

        # plot point cloud:
        xinlier = X[inlier_mask]
        yinlier = y[inlier_mask]

        #verticality = np.std(xinlier)/np.std(yinlier)
        #score = ransac.score(xinlier, yinlier)

        #store linear models
        if RANSACmodel == 1:
            models.append((res.estimator_.coef_[0], res.estimator_.intercept_, RANSACmodel))
        else:
            models.append((1/res2.estimator_.coef_[0], -res2.estimator_.intercept_/res2.estimator_.coef_[0], RANSACmodel))

        idx += 1
       
        # Predict data of estimated models
        '''
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        line_y = ransac.predict(line_X)
        line_y_ransac = ransac.predict(line_X)
        plt.plot(line_X, line_y_ransac, color=color, linewidth=1, label='RANSAC regressor')
        '''
        # only keep the outliers:
        X = X[~inlier_mask]
        y = y[~inlier_mask]
       
    return segments, models