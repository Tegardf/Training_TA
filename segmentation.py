import cv2
import numpy as np

import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb
from skimage.segmentation import active_contour
from skimage import measure
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter,rectangle_perimeter



def ROI_segmentation(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray , (5,5), 0)
    kernelSharp = np.array([[0, -1, 0],[-1,5,-1],[0,-1,0]])
    imgBlur = cv2.filter2D(imgBlur, -1, kernelSharp)
    centerImg = [np.shape(img)[0]/2,np.shape(img)[1]/2]
    # detect Iris Area
    circles = cv2.HoughCircles(imgBlur, cv2.HOUGH_GRADIENT, 1, 90, param1 = 235, param2 = 4, minRadius = np.uint64(np.shape(img)[0]/3.5), maxRadius= np.uint64(np.shape(img)[0]/2.8))
    inner_circle = np.around(circles[0][0]).tolist()
    distPoint = [abs(centerImg[0]-inner_circle[0]),abs(centerImg[1]-inner_circle[1])]
    for c in circles[0,:]:
        if abs(distPoint[0])>abs(centerImg[0]-c[0]) and abs(distPoint[1])>abs(centerImg[1]-c[1]):
            inner_circle[0] = c[0]
            distPoint[0] = abs(centerImg[0]-c[0])
            inner_circle[1] = c[1]
            distPoint[1] = abs(centerImg[1]-c[1])
            inner_circle[2] = c[2]
    centerx = (inner_circle[0]+centerImg[0])/2.1
    centery = (inner_circle[1]+centerImg[1])/2.1
    inner_circle = (centerx,centery,inner_circle[2])
    inner_circle = np.uint64(inner_circle)
    x1,y1,x2,y2 = drawRec(img.shape,centerx,centery,inner_circle[2], 1.6)
    # cv2.circle(img,(inner_circle[0],inner_circle[1]),np.uint64(inner_circle[2]+2),(0,0,255),4)
    # cv2.rectangle(img,(x1 , y1),(x2,y2),(0,255,0),2)
    img_crop = img[(x1):(x2), (y1):(y2)]
    # print(img_crop.shape)s
    # cv2.imshow("before",img_crop)
    # cv2.imshow("hasil", img)
    img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    hsv_value = img_hsv[:,:,2]
    edges = canny(hsv_value, sigma=3, low_threshold= 6, high_threshold=40)
    hough_radii = np.arange(14,45,1)
    hough_res = hough_circle(edges,hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    xs1,ys1,xs2,ys2 = drawRec(img_crop.shape, cy[0],cx[0],radii[0])
    img_crop2 = img_crop[(xs1):(xs2), (ys1):(ys2)]
    # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
    # image = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    # circy, circx = circle_perimeter(cy[0], cx[0], radii[0], shape=image.shape)
    # image[circy, circx] = (220, 20, 20)
    # row,col = rectangle_perimeter(start=(xs1,ys1), end=(xs2,ys2))
    # image[row,col] = (255,255,0)
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax2.imshow(hsv_value)
    # ax3.imshow(edges)
    # ax4.imshow(img_crop2)
    # ax1.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # ax2.imshow(cv2.cvtColor(img_crop,cv2.COLOR_BGR2RGB))
    # ax3.imshow(cv2.cvtColor(img_crop2,cv2.COLOR_BGR2RGB))
    # ax4.imshow(image)
    # plt.show()
    img_crop2 = cv2.cvtColor(img_crop2 ,cv2.COLOR_BGR2GRAY)
    return img_crop2


def drawRec(imgShape,x,y,r,divide = 1):
    top_left_x = int(max(0, x-r))
    top_left_y = int(max(0, y-r))
    bottom_right_x = int(min(imgShape[1],x + r))
    bottom_left_y = int(min(imgShape[0],y + r))
    if divide > 1:
        Ax = bottom_right_x - top_left_x
        Ay = bottom_left_y - top_left_y
        Ax2 = Ax/divide
        Ay2 = Ay/divide
        top_left_x = np.int32(top_left_x + ((Ax - Ax2)/2)) 
        top_left_y = np.int32(top_left_y + ((Ay - Ay2)/2)) 
        bottom_right_x = np.int32(top_left_x + Ax2) 
        bottom_left_y = np.int32(top_left_y + Ay2) 

    return top_left_x,top_left_y,bottom_right_x,bottom_left_y

def filterCircle(circles, img_shape):
    if len(circles) == 1:
        return circles[0]
    return circles[0]


    # newCircles = []
    # for i in circles:
    #     if i[2] < img_shape[0]/3:
    #         newCircles = i
    # return newCircles

def ifFaraway(x,y,x_im, y_img, distance):
    distanceX = abs(x-x_im)
    distanceY = abs(y-y_img)
    if distanceX>=distance and distanceY>=distance:
        print('x and y is far')
        return x_im,y_img
    if distanceX>=distance:
        print('x is far')
        return x_im,y
    if distanceY>=distance:
        print('y is far')
        return x,y_img
    return x,y




