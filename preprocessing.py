import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment import segmentation
import os

#img_path=r'C:\Users\ultra\source\repos\IP__17123313_chest\datasets\train\positive\1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-002-fig3b.png'
#good_path=r'C:\Users\ultra\source\repos\IP__17123313_chest\datasets\test\negative\00002474_006.png'
#bad_path=r'C:\Users\ultra\source\repos\IP__17123313_chest\datasets\test\positive\85E52EB3-56E9-4D67-82DA-DEA247C82886.jpeg'

def pre_processing(img):
    img_rgb  =  cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #step 1 : resize
    img_1 = cv2.resize(img_rgb, (800,800),interpolation=cv2.INTER_AREA)

    #step 2 : historical equalisation
    img_2 = cv2.equalizeHist(img_1)

    #step 3 : cropping abdomen area
    img_3 = img_2

    #step 4 : Lung boundary
    crop = segmentation(img_3)
    img_4 = crop*img_3

    #step 5 : Thresholding
    _, img_result = cv2.threshold(img_4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_result




ROOT = r"C:/Users/ultra/source/repos/IP__17123313_chest/"
#INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR_positive = ROOT+'datasets/train/positive'
TEST_IMG_DIR_positive = ROOT+'datasets/test/positive'
VALI_IMG_DIR_positive = ROOT+'datasets/vali/positive'
TRAIN_IMG_DIR_negative = ROOT+'datasets/train/negative'
TEST_IMG_DIR_negative = ROOT+'datasets/test/negative'
VALI_IMG_DIR_negative = ROOT+'datasets/vali/negative'
OUTPUT_DIR = ROOT+'pos'



if __name__=='__main__':
    current_process  =  VALI_IMG_DIR_negative
    output_path = current_process.replace('datasets','datasets_post')
    ok = os.listdir(current_process)
    counter=1
    for i in ok:
        input = current_process+'/'+i
        print(counter)
        filename = output_path +r'/'+'{}.png'.format(counter)
        output = pre_processing(input)
        #print(filename, output)
        cv2.imwrite(filename, output)
        counter=counter+1



#res = np.hstack((img_1,img_result)) #stacking images side-by-side
#cv2.imshow('image',img_result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#equ=cv2.equalizeHist(img_gray)
#res = np.hstack((img_gray,equ)) #stacking images side-by-side
#cv2.imwrite('res.png',res)
#median = cv2.medianBlur(equ,5)
#cv2.imshow('image',res)