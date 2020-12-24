import sys
import os
import glob
import numpy as np
import cv2
import scipy.misc
from matplotlib import pyplot as plt





def segmentation(img):
    img_erased = eraseMax(img,draw=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_erased)   

    ker = 169
    kernel = np.ones((ker,ker),np.uint8)
    blackhat = cv2.morphologyEx(img_clahe, cv2.MORPH_BLACKHAT, kernel)    

    threshold = 45
    ret, thresh = cv2.threshold(blackhat, threshold, 255, 0)
    cmask = get_cmask(img_clahe)

    mask = np.multiply(cmask,thresh).astype('uint8')
    median = cv2.medianBlur(mask,23)
    contour_mask = contourMask(median).astype('uint8')
    return contour_mask
def eraseMax(img,eraseLineCenter=0,eraseLineWidth=30,draw=False):
    sumpix0=np.sum(img,0)
    if draw:
        plt.plot(sumpix0)
        plt.title('Sum along axis=0')
        plt.xlabel('Column number')
        plt.ylabel('Sum of column')
    max_r2=np.int_(len(sumpix0)/3)+np.argmax(sumpix0[np.int_(len(sumpix0)/3):np.int_(len(sumpix0)*2/3)])
    cv2.line(img,(max_r2+eraseLineCenter,0),(max_r2+eraseLineCenter,512),0,eraseLineWidth)
    return img
def get_cmask(img, maxCorners=3800, qualityLevel=0.001, minDistance=1,Cradius=6):
    corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
    corners = np.int0(corners)
    cmask = np.zeros(img.shape)
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(cmask,(x,y),Cradius,1,-1)
    return cmask
def contourMask(image):
    contours ,hierc = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    area = np.zeros(len(contours))
    for j in range(len(contours)):
        cnt = contours[j]
        area[j] = cv2.contourArea(cnt)
    mask = np.zeros(image.shape)
    cv2.drawContours(mask, contours, np.argmax(area), (255), -1)#draw largest contour-usually right lung   
    temp = np.copy(area[np.argmax(area)])
    area[np.argmax(area)]=0
    if area[np.argmax(area)] > temp/10:#make sure 2nd largest contour is also lung, not 2 lungs connected
        cv2.drawContours(mask, contours, np.argmax(area), (255), -1)#draw second largest contour  
    contours.clear() 
    return mask



if __name__ == '__main__':
    filename=r'C:\Users\ultra\source\repos\IP__17123313_chest\datasets\train\positive\1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-002-fig3b.png'
    img = cv2.imread(filename,0) 
    print('Image shape is:', img.shape)
    contour_mask=segmentation(img)
    plt.imshow(contour_mask,cmap='gray')
    print(contour_mask)
    res = np.hstack((img,contour_mask)) #stacking images side-by-side
    surpise = contour_mask*img
    cv2.imshow('image',surpise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()