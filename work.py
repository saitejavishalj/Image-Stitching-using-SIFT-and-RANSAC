# -*- coding: utf-8 -*-
"""
Created on Tue Apr 07 14:53:29 2021

@author: Sai Teja Vishal J
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

image1 = cv2.imread("keble_a.jpg")
image2 = cv2.imread("keble_b.jpg")
image3 = cv2.imread("keble_c.jpg")

plt.imshow(image1)
plt.show()

##Using SIFT to make the feature detection invariant to scale and rotation
sift = cv2.SIFT_create()

##Finding the Keypoints and Descriptors of all the three images
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
kp3, des3 = sift.detectAndCompute(image3, None)

##Function to compute the Euclidean Distance between two points, used in finding the 
def eucildean_Distance(des_img1,des_img2):
    result = np.zeros((len(des_img2),len(des_img1)), np.float32)
    
    for i in range(len(des_img2)):
        des_img2_rows = np.array([des_img2[i]]*len(des_img1))
        subtraction_matrix = np.subtract(des_img1, des_img2_rows)
        square_matrix = np.square(subtraction_matrix)
        #returns an array of sums
        sum_row = square_matrix.sum(axis=1)
        result[i] = np.sqrt(sum_row)
    return result

img1_img2_dist = eucildean_Distance(des1, des2)

indices_of_sorted = np.argsort(img1_img2_dist, axis=1)[:,:2]
distance_matrix = np.zeros((indices_of_sorted.shape[0], indices_of_sorted.shape[1]))
distance_matrix = img1_img2_dist[np.arange(img1_img2_dist.shape[0])[:,None], indices_of_sorted]

##Validating the matches based on Lowe's ratio, which is 0.7. Eliminating the point, if it is almost good. 
##The distance between the points should be sufficiently different, then considered. 
#Reference: https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
def matches_SIFT(indices,dist_mat,keypoint1,keypoint2):
    m1 = []
    m2 = []
    indices_goodMatch = []
    sift_matches = []
    for i in range(len(indices)):
        if dist_mat[i][0] < 0.7 * dist_mat[i][1]:
            indices_goodMatch.append((i,indices[i][0]))
    print("Debugging")
    print(len(indices_goodMatch))
    print(len(sift_matches))
    for i in range(len(indices_goodMatch)-1):
        sift_matches.append((keypoint1[indices_goodMatch[i][0]],keypoint2[indices_goodMatch[i][1]]))
        m1.append(keypoint1[indices_goodMatch[i][0]])
        m2.append(keypoint2[indices_goodMatch[i][1]])
    
    return sift_matches,m1,m2

sift_matches, matches_img2, matches_img1 = matches_SIFT(indices_of_sorted, distance_matrix, kp2, kp1)

##On the 1st and 2nd image, point all the keypoints on the respective images using a function in OpenCV
image1_keypoints = cv2.drawKeypoints(image1, matches_img1, None)
plt.imshow(cv2.cvtColor(image1_keypoints, cv2.COLOR_BGR2RGB))
plt.show()
image2_keypoints = cv2.drawKeypoints(image2, matches_img2, None)
plt.imshow(cv2.cvtColor(image2_keypoints, cv2.COLOR_BGR2RGB))
plt.show()

##Concatenate both the images
image_combined_1_2 = np.concatenate((image1_keypoints,image2_keypoints),axis=1)
plt.imshow(cv2.cvtColor(image_combined_1_2, cv2.COLOR_BGR2RGB))
plt.show()

## drawing lines between good features between image 1 and image 2
points_right = []

for i in range(len(matches_img2)):
    points_left = np.int32(matches_img1[i].pt)
    points_right = np.int32(matches_img2[i].pt[0] + image1_keypoints.shape[1])
    combined_1_2_lines=cv2.line(image_combined_1_2,(points_left[0],points_left[1]),(points_right,np.int32(matches_img2[i].pt[1])),(0,255,0),thickness=1)

plt.imshow(cv2.cvtColor(combined_1_2_lines, cv2.COLOR_BGR2RGB))
plt.show()

def transform_as_homography(H_matrix, points):
    #perform padding of ones to pts and make them (x,y,1)
    # print(H_matrix,points)
    matrix_temp = np.ones((points.shape[0]+1 , points.shape[1]), np.float32)
    matrix_temp[:-1, :] = points
    H_multiplied = np.matmul(H_matrix,matrix_temp)
    #each column of transformed points is 1 transformed point (x,y,z)
    transformed = np.zeros((points.shape[0] , points.shape[1]), np.float32)
    #converting to heterogenousc(x/z,y/z) of (x,y,z)
    transformed[0, :] = np.divide(H_multiplied[0,:], H_multiplied[2, :])
    transformed[1, :] = np.divide(H_multiplied[1,:], H_multiplied[2, :])
    # print(transformed)
    return transformed

def getInlier_count(target_points, transformed_points, threshold):
    score = 0
    inlier_set = []
    #calculating error function
    err = np.square(np.subtract(target_points, transformed_points))
    err = np.sqrt(err.sum(axis=0))
    for i in range(len(err)):
        if(err[i]<threshold):
            inlier_set.append(i)
            score += 1
            
    return score, inlier_set

def matrix_keypoints(kp):
    matrix = np.zeros((2,len(kp)),np.float32)
    for i, j in enumerate(kp):
        matrix[0, i] = j.pt[0]
        matrix[1, i] = j.pt[1]
    return matrix

def matrix_formation(pt1, pt2):
    result = np.zeros((2,9), np.float32)
    result[0, :] = [-pt1[0], -pt1[1], -1, 0, 0, 0, pt1[0]*pt2[0], pt1[1]*pt2[0], pt2[0]]
    result[1, :] = [0, 0, 0, -pt1[0], -pt1[1], -1, pt1[0]*pt2[1], pt1[1]*pt2[1], pt2[1]]
#     print(lin)
    return result

##Computation of Homography using SVD, ref: https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
def computeH(img1_pts, img2_pts):
    A_matrix = np.zeros((2*len(img1_pts[0]),9), np.float32)
    for i in range(len(img1_pts[0])):
        A_matrix[2*i: 2*i+2, :] = matrix_formation(img1_pts[:,i], img2_pts[:,i])
    sigma, U, V_transpose = cv2.SVDecomp(A_matrix)
    
    #Here, we take the right singular vector (a column from V ) which corresponds to the smallest singular value sigma-9
    h_temp = V_transpose[len(V_transpose)-1,:] ##V_transpose is the right singular vector
    if (h_temp[len(h_temp)-1]!=0):
        h_temp = h_temp*(1/h_temp[len(h_temp)-1])
    H = np.reshape(h_temp, (3,3))
    return H

def RANSAC(matches,e,iterations):
    inlier_array_highest = []
    for m in range(iterations):
        random_matches = random.choices(matches, k=4)
        
        first_image_pts = matrix_keypoints([i[0] for i in random_matches])
        second_image_pts = matrix_keypoints([i[1] for i in random_matches])
        
        # compute Homography using the selected matches
        homography_matrix = computeH(second_image_pts,first_image_pts)
        second_image_keypoints = [j[1] for j in matches]
        
        # converting the keypoints into coordinates of 2*N matrix
        second_coords = matrix_keypoints(second_image_keypoints)
        second_transformed = transform_as_homography(homography_matrix, second_coords)
        eps = e
        first_img_keys = [i[0] for i in matches]
        first_coords = matrix_keypoints(first_img_keys)
        
        inlier_count, inlier_arr_computed = getInlier_count(first_coords,second_transformed,eps)
        
        ##Check for the maximum number of inliers here, and consider those indices
        if (inlier_count>len(inlier_array_highest)):
            inlier_array_highest = inlier_arr_computed
            
    percentage = len(inlier_array_highest)*100/len(matches)
    # print(inlier_array_highest)
    print("inlier percentage is : ", percentage)
    
    #Compute the final homography again based on RANSAC Algorithm
    final_matches = [matches[i] for i in inlier_array_highest]
    first_coords_final = matrix_keypoints([i[0] for i in final_matches])
    second_coords_final = matrix_keypoints([i[1] for i in final_matches])
    final_valueH = computeH(second_coords_final, first_coords_final)
    return final_matches, final_valueH

##Ransac Algorithm between images 1 and 2
final_matches, final_valueH= RANSAC(sift_matches, 5, 5000)

##Matching after estimating Homography using RANSAC
final_matches2 = []
final_matches1= []
for m in final_matches:
    final_matches2.append(m[0])
    final_matches1.append(m[1])

point_right = []
for i in range(len(final_matches2)):
    point_left = np.int32(final_matches1[i].pt)
    point_right = np.int32(final_matches2[i].pt[0] + image1_keypoints.shape[1])
    final_img_1_2=cv2.line(image_combined_1_2,(point_left[0],point_left[1]),(point_right,np.int32(final_matches2[i].pt[1])),(0,255,0),thickness=1)
    
# FinalImgmatch = drawMatches(img2, img1, BestMatchesFound)
plt.imshow(cv2.cvtColor(final_img_1_2, cv2.COLOR_BGR2RGB))
plt.show()

def warping(img1, img2, H):
    
    height1,width1 = img1.shape[:2]
    height2,width2 = img2.shape[:2]
    points_1 = np.float32([[0,0],[0,height1],[width1,height1],[width1,0]]).reshape(-1,1,2)
    points_2 = np.float32([[0,0],[0,height2],[width2,height2],[width2,0]]).reshape(-1,1,2)
    points_2_new = cv2.perspectiveTransform(points_2, H)
    combined_points = np.concatenate((points_1, points_2_new), axis=0)
    print(combined_points.shape)
    [x_minimum, y_minimum] = np.int32(combined_points.min(axis=0).ravel() - 0.5)
    [x_maximum, y_maximum] = np.int32(combined_points.max(axis=0).ravel() + 0.5)
    translation = [-x_minimum,-y_minimum]
    h_translate = np.array([[1,0,translation[0]],
                           [0,1,translation[1]],
                           [0,0,1]])
    output = cv2.warpPerspective(img2, h_translate.dot(H), (x_maximum-x_minimum, y_maximum-y_minimum))
    output[translation[1]:height1+translation[1],translation[0]:width1+translation[0]] = img1
    
    return output

image_warp_1_2 = warping(image2,image1,final_valueH)
print(image_warp_1_2.shape)
plt.imshow(cv2.cvtColor(image_warp_1_2, cv2.COLOR_BGR2RGB))
plt.show()

keypoints_12, des_12 = sift.detectAndCompute(image_warp_1_2, None)

image_12_3_dist = eucildean_Distance(des_12, des3)

indices_of_sorted_12_3 = np.argsort(image_12_3_dist, axis=1)[:,:2]
distance_matrix_12_3 = np.zeros((indices_of_sorted_12_3.shape[0], indices_of_sorted_12_3.shape[1]))
distance_matrix_12_3 = image_12_3_dist[np.arange(image_12_3_dist.shape[0])[:,None], indices_of_sorted_12_3]

sift_matches_12_3, match_img3, match_img12 = matches_SIFT(indices_of_sorted_12_3, distance_matrix_12_3, kp3, keypoints_12)


img12_keypoints_draw = cv2.drawKeypoints(image_warp_1_2, match_img12, None)
img3_keypoints_draw = cv2.drawKeypoints(image3, match_img3, None)
plt.imshow(cv2.cvtColor(img3_keypoints_draw, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(img12_keypoints_draw, cv2.COLOR_BGR2RGB))
plt.show()

##Resizing the warped image
if(img12_keypoints_draw.shape[0]!=img3_keypoints_draw.shape[0]):
    img12_keypoints_draw = cv2.resize(img12_keypoints_draw, (img12_keypoints_draw.shape[1], img3_keypoints_draw.shape[0]))
    
image_concat_12_3 = np.concatenate((img12_keypoints_draw, img3_keypoints_draw), axis=1)
plt.imshow(cv2.cvtColor(image_concat_12_3, cv2.COLOR_BGR2RGB))
plt.show()

points_right = []
for i in range(len(match_img3)):
    points_left = np.int32(match_img12[i].pt)
    points_right = np.int32(match_img3[i].pt[0] + img12_keypoints_draw.shape[1])
    image_12_3_lines=cv2.line(image_concat_12_3,(points_left[0],points_left[1]),(points_right,np.int32(match_img3[i].pt[1])),(0,255,0),thickness=1)

plt.imshow(cv2.cvtColor(image_12_3_lines, cv2.COLOR_BGR2RGB))
plt.show()

ransac_matches, final_valueH_12_3= RANSAC(sift_matches_12_3, 5, 5000)

#Feature Matching after estimating Homographies using RANSAC
correct_matches_3 = []
correct_matches_12_3 = []
for m in ransac_matches:
    correct_matches_3.append(m[0])
    correct_matches_12_3.append(m[1])

points_right = []
for i in range(len(correct_matches_3)):
    points_left = np.int32(correct_matches_12_3[i].pt)
    points_right = np.int32(correct_matches_3[i].pt[0] + img12_keypoints_draw.shape[1])
    image_12_3_lines=cv2.line(image_concat_12_3,(points_left[0],points_left[1]),(points_right,np.int32(correct_matches_3[i].pt[1])),(0,255,0),thickness=1)

plt.imshow(cv2.cvtColor(image_12_3_lines, cv2.COLOR_BGR2RGB))
plt.show()

contrcuted_image = warping(image3,image_warp_1_2,final_valueH_12_3)
print(image_warp_1_2.shape)
plt.imshow(cv2.cvtColor(contrcuted_image, cv2.COLOR_BGR2RGB))
plt.show()
