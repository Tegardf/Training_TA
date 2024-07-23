import cv2
from skimage.restoration import inpaint
import numpy as np

def removeFlare(imgBGR):
    # cv2.imshow("original",imgBGR)
    lab = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab img",lab)
    # cv2.waitKey(0)
    labPlane = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(13,13))
    # cv2.imshow("L space",labPlane[0])
    labPlane[0]=clahe.apply(labPlane[0])
    lab = cv2.merge(labPlane)
    # cv2.imshow("after clahe",lab)
    claheBgr=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    grayimg = cv2.cvtColor(claheBgr,cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(grayimg, 200,255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("masking",mask)
    result= cv2.inpaint(imgBGR,mask,0.2,cv2.INPAINT_NS)
    # cv2.imshow("result",result)
    # cv2.waitKey(0)
    return result

def claheFilterContrast(imgBGR):
    lab = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2LAB)
    labPlane=list(cv2.split(lab))
    # cv2.imshow("lab img",labPlane[0])
    

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
    labPlane[0]=clahe.apply(labPlane[0])
    lab = cv2.merge(labPlane)
    claheBgr=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

    claheBgr=cv2.GaussianBlur(claheBgr,(5,5),0)
    # cv2.imshow("result",cv2.cvtColor(claheBgr,cv2.COLOR_BGR2GRAY))
    # cv2.waitKey(0)
    return claheBgr

def removeGlare2(imgBGR):
    gray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

    # lab = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab img",lab)
    # labPlane = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(13,13))
    gray = clahe.apply(gray)
    # cv2.imshow("L space",labPlane[0])
    # labPlane[0]=clahe.apply(labPlane[0])
    # lab = cv2.merge(labPlane)
    # cv2.imshow("after clahe",lab)
    # claheBgr=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    # grayimg = cv2.cvtColor(claheBgr,cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 200,255, cv2.THRESH_BINARY)[1]
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    # cv2.imshow("masking",mask)
    # cv2.imshow("dilated masking",dilated_mask)
    # imgTest = cv2.bitwise_and(grayimg,cv2.bitwise_not(dilated_mask))
    # imgBGR2 = cv2.cvtColor(imgTest, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("remove mask in IMG",imgTest)
    # cv2.imshow("img BGR 2",imgBGR2)
    imgRes = inpaint.inpaint_biharmonic(imgBGR, dilated_mask, channel_axis= -1)
    imgRes = imgRes*255.0
    imgRes = imgRes.astype(np.uint8)
    # cv2.imshow("Skimage Inpaint", imgRes)
    # cv2.waitKey(0)
    # imgRes = cv2.cvtColor(imgRes,cv2.COLOR_BGR2GRAY)
    return imgRes


import os

def main():
    categories = ['katarak']
    # categories = ['normal']
    dataDir = 'Dataset/'
    # img_arr = []
    for i in categories:
        path = os.path.join(dataDir,i)
        for j in os.listdir(path):
            # img_name = str(j) + ".jpg"
            # img = cv2.imread(os.path.join(path,img_name))
            # cv2.imshow("images 1",img)
            img = cv2.imread(os.path.join(path,j))
            img = cv2.resize(img,(300,300),interpolation=cv2.INTER_CUBIC)
            # img = claheFilterContrast(img)
            # img = removeFlare(img)
            img = removeGlare2(img)
            # cv2.imshow("images 2",img)
            # cv2.waitKey(0)
    
    # for k in img_arr:
        # img = ROI_segmentation(k)
        

if __name__=="__main__":
    main()