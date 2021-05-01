# Covid19-chest-classification-using-DL-image-processing
Covid19-chest-classification-using-DL-image-processing
1.Introduction
The COVID-19 pandemic has been gripping much of the world recently. Although there have been much efforts gone into developing affordable testing for the masses, it has been shown that the established and widely available chest x-rays (CXR) may be used as a screening criterion. There are many dedicated works by various individuals and organizations, publicly available chest x-rays of COVID-19 subjects are available for analytic usage. The 3rd Deep Learning and Artificial Intelligence Summer/Winter School (DLAI3) is glad to offer such an opportunity for aspiring researchers and hobbyists to test their skill in this area.

This challenge is a continuation from the CSC532 Machine Learning course projects from the School of Information Technology, King Mongkut's University of Technology Thonburi (KMUTT), which provided the mini-dataset and the dataset of annotated CXR images.

The idea for using the CXR data is to determine weather the patient admitted is positive or negative in covid-19. There is study found that one of the covid-19 symptom is having clouded formation in the lung and causes difficult in breathing. Since this is a binary output, we treat it as a classification problem. I combine both image processing technique and deep learning method in order to produce a more accurate and more performance outcome. Image processing technique can enhance the feature so that deep learning model can be learn faster.

Objective 1:
To identify the positivity of Covid-19 from the image of lung's x ray image.

Objective 2:
To evaluate the accuracy of the model classifying COVID-19 X-ray image

Data Source

The images data is a open source data provided by CSC532 Machine Learning course projects from the School of Information Technology, King Mongkut’s University of Technology Thonburi (KMUTT).

There are 250 images of chest X-ray images. Half of the data is Covid-19 positive anda nother half is Covid-19 negative. Their labels 1,0 is saved in a excel CSV format. Below attach original image from sources.

![image](https://user-images.githubusercontent.com/29944896/116771181-031f9480-aa7c-11eb-8fde-741996b3d997.png)

Analysis and Design

![image](https://user-images.githubusercontent.com/29944896/116771207-20ecf980-aa7c-11eb-86ac-6fb90ccc39c8.png)

The overall process can be categorised into 4 stages. The first step is to process the image before make detection of the lungs. I used to resize and histogram equalisation in this stage (refer figure 2 top 2 step). Next stage is detection of an object. I used image segmentation, normalization, morphology technique, corner detection, median blur filter and identify the position of object of interest (refer figure 3). Third stage, I perform to do feature extraction using deep learning method. The model I used is called Alexnet model which widely used in image classification. Last stage, the prediction outcome of the deep learning model will then compare to actual result to get the accuracy. 

![image](https://user-images.githubusercontent.com/29944896/116771214-2ea27f00-aa7c-11eb-992b-57f192e74ac5.png)

![image](https://user-images.githubusercontent.com/29944896/116771219-34986000-aa7c-11eb-8689-b89ac95b6358.png)

Figure above shown all stages.

In this project, I used Python programming language, with OpenCV and pytorch library. Let’s start with the original images for stage 1. Stage 1 mostly focus on contrast enhancement and resizing. Below start with a sample image.![image](https://user-images.githubusercontent.com/29944896/116771232-4843c680-aa7c-11eb-9dd9-78d98c30cd36.png)

The original images are then converting to square with dimension of 800x800 with interpolation = Inter-Area.
![image](https://user-images.githubusercontent.com/29944896/116771239-5560b580-aa7c-11eb-844f-a7c3e0f691f5.png)

Then we perform histogram equalization on the image above. Histogram equalization is used to enhance contrast. It is not necessary that contrast will always be increase in this. There may be some cases were histogram equalization can be worse. In that cases the contrast is decreased. This method usually increases the global contrast of many images, especially when the usable data of the image is represented by close contrast values. Through this adjustment, the intensities can be better distributed on the histogram. This allows for areas of lower local contrast to gain a higher contrast. Histogram equalization accomplishes this by effectively spreading out the most frequent intensity values.https://github.com/Ultraleow/Covid19-chest-classification-using-DL-image-processing The method is useful in images with backgrounds and foregrounds that are both bright or both dark. In particular, the method can lead to better views of bone structure in xray images, and to better detail in photographs that are over or under-exposed.

![image](https://user-images.githubusercontent.com/29944896/116771246-63163b00-aa7c-11eb-86b9-af12e76ca5e8.png)

After stage 1, we proceed to stage 2 which is segmentation of lung area. We are then removing the spinal cord, which can be found by calculating the maximum when summing along the first axis. And then we will get 3 maxima, while the largest for the spinal chord.

![image](https://user-images.githubusercontent.com/29944896/116771252-6d383980-aa7c-11eb-97e4-5e4c0afb6ebf.png)

And we get 3 maxima, the largest for the spinal chord.

After erasing we get:

![image](https://user-images.githubusercontent.com/29944896/116771255-775a3800-aa7c-11eb-8438-2776ecefd867.png)

Next step was to naromalize the images, and to bring out more detail, using CLAHE:

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

In image processing, normalization is a process that changes the range of pixel intensity values. Often, the motivation is to achieve consistency in dynamic range for a set of data, signals, or images to avoid mental distraction or fatigue.

![image](https://user-images.githubusercontent.com/29944896/116771264-8345fa00-aa7c-11eb-92db-b366be4e30bd.png)

Next we applied openCV's Blackhat kernel:
ker = 169
kernel = np.ones((ker,ker),np.uint8)
blackhat = cv2.morphologyEx(img_clahe, cv2.MORPH_BLACKHAT, kernel)https://github.com/Ultraleow/Covid19-chest-classification-using-DL-image-processing

According to Wikipedia, morphological operations rely only on the relative ordering of pixel values, not on their numerical values, and therefore are especially suited to the processing of binary images. A morphological operation on a binary image creates a new binary image in which the pixel has a non-zero value only if the test is successful at that location in the input image. Black top-hat filtering returns an image containing the objects or elements of the input image that are smaller than the structuring element and darker than their surroundings. In digital image processing, a morphological gradient is the difference between the dilation and the erosion of a given image.

![image](https://user-images.githubusercontent.com/29944896/116771274-948f0680-aa7c-11eb-94e1-c886d044caff.png)

In the next step we applied a threshold:
ret, thresh = cv2.threshold(blackhat, threshold, 255, 0)

![image](https://user-images.githubusercontent.com/29944896/116771276-9ce74180-aa7c-11eb-8fba-e6c6fcded422.png)

Next we used openCV's corner detection function on the original image, which was useful because the lungs have many small details which can be detected as corners:
def get_cmask(img, maxCorners=3800, qualityLevel=0.001, minDistance=1,Cradius=6):
corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
corners = np.int0(corners)
cmask = np.zeros(img.shape)
for corner in corners:
x,y = corner.ravel()
cv2.circle(cmask,(x,y),Cradius,1,-1)
return cmask

![image](https://user-images.githubusercontent.com/29944896/116771285-a83a6d00-aa7c-11eb-9905-a77183ee823b.png)

And so the corner mask gives us the lungs without the background.https://github.com/Ultraleow/Covid19-chest-classification-using-DL-image-processing
So in the next step we multiply the two previous results:

![image](https://user-images.githubusercontent.com/29944896/116771289-b12b3e80-aa7c-11eb-9d74-6ed534a90b1e.png)

Next, we used a median blur to clean the mask. The Median blur operation is like the other averaging methods. Here, the central element of the image is replaced by the median of all the pixels in the kernel area. This operation processes the edges while removing the noise.

![image](https://user-images.githubusercontent.com/29944896/116771295-bbe5d380-aa7c-11eb-847d-bb75113d1910.png)

Next we use openCV's contour finder to get only the 2 largest contours, which are the lungs:

def contourMask(image):
im2,contours,hierc = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SI
MPLE)
area = np.zeros(len(contours))
for j in range(len(contours)):
cnt = contours[j]
area[j] = cv2.contourArea(cnt)
mask = np.zeros(image.shape)
cv2.drawContours(mask, contours, np.argmax(area), (255), -1)#draw largest contour-usua
lly right lung
temp = np.copy(area[np.argmax(area)])
area[np.argmax(area)]=0
if area[np.argmax(area)] > temp/10:#make sure 2nd largest contour is also lung, not 2 lun
gs connected
cv2.drawContours(mask, contours, np.argmax(area), (255), -1)#draw second largest co
ntour
contours.clear()
return mask

![image](https://user-images.githubusercontent.com/29944896/116771299-cacc8600-aa7c-11eb-9820-ea6e97cc3d0c.png)

This is the end of stage 2. Stage 2 mainly produce only lung segmented image with the visualization of the lung segment. For human, we can identify the likeness of Covid-19 by observing both lungs is connected fully or not. However, we also can make this process automated which is by training a classification neural network to classify it. All images are processed with stage 1 and 2 save saved into datasets_post folder. I separated the data into 3 folders which are for training, testing and validation for neural network. Their ratio is 70% for training, 20% for testing and the remaining 10% is for validation. The classification model selected is Alexnet Model.

AlexNet is an incredibly powerful model capable of achieving high accuracies on very challenging datasets. However, removing any of the convolutional layers will drastically degrade AlexNet’s performance. AlexNet is a leading architecture for any object-detection task and may have huge applications in the computer vision sector of artificial intelligence problems. In the future, AlexNet may be adopted more than CNNs for image tasks. Below attached the architecture of AlexNet model.

![image](https://user-images.githubusercontent.com/29944896/116771309-dfa91980-aa7c-11eb-84fb-b6a0ed5459f7.png)

After stage 3, we must review out classification result. In stage 4, I made a confusion matrix
for visualization. The overall accuracy of the model is around 90%.

![image](https://user-images.githubusercontent.com/29944896/116771317-f0f22600-aa7c-11eb-995f-06eb1e2c745d.png)

Experiment results
The result will be separated into 2 parts, which are first 2 stages is image processing method, the last 2 stages is deep learning method.
The results of the image processing, which is also the input of the neural network model is 
attached as below:

Positive Covid-19:

![image](https://user-images.githubusercontent.com/29944896/116771326-fea7ab80-aa7c-11eb-9f1b-855e0a394101.png)

Negative Covid-19:

![image](https://user-images.githubusercontent.com/29944896/116771330-06ffe680-aa7d-11eb-8be7-0d8347a73b66.png)

The performance of this method looks good because the overall accuracy is 90% and above. The strength of this overall method is that, we can identify the likeness of Covid-19 even only image processing technique, this will allow medical or non-medical make more accurate judgement. Also, by adding classifier, the overall process is automated. The image processing technique let us to identify more details in the image by normalize and perform morphology on it. The expected results with only image processing technique is only 2 segments of lungs, black background and maybe broken noted in the segment of lungs (Covid-19 symptoms). By using the algorithms above, I managed to achieve to obtain this kind of features. The limitations might be the neural network because 90% is not yet excellent and have the risk of overfitting because less data is used. The future improvements will be using different classification model.

References
1. M. N. Saad, Z. Muda, N. S. Ashaari and H. A. Hamid, "Image segmentation for lung
region in chest X-ray images using edge detection and morphology," 2014 IEEE
International Conference on Control System, Computing and Engineering (ICCSCE
2014), Batu Ferringhi, 2014, pp. 46-51, doi: 10.1109/ICCSCE.2014.7072687.
2. Wei, J. (2020, September 25). AlexNet: The Architecture that Challenged CNNs.
Retrieved January 13, 2021, from https://towardsdatascience.com/alexnet-thearchitecture-that-challenged-cnns-e406d5297951
3. DLAI3 Hackathon. (n.d.). Retrieved January 13, 2021, from
https://www.kaggle.com/c/dlai3/data?select=datasets
4. Ilaiw. (n.d.). Ilaiw/CXR-lung-segmentation. Retrieved January 13, 2021, from
https://github.com/ilaiw/CXR-lung-segmentation
5. Morphological Image Processing. (n.d.). Retrieved January 13, 2021, from
https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessinghtml/topic4.htm#:~:text=Morphological%20image%20processing%20is%20a,of%20fe
atures%20in%20an%20image.&text=A%20morphological%20operation%20on%20a,
location%20in%20the%20input%20image.
6. Normalization (image processing). (2020, April 16). Retrieved January 13, 2021, from
https://en.wikipedia.org/wiki/Normalization_(image_processing)#:~:text=In%20image
%20processing%2C%20normalization%20is,due%20to%20glare%2C%20for%20exa
mple.&text=Often%2C%20the%20motivation%20is%20to,avoid%20mental%20distra
ction%20or%20fatigue.
7. Histogram equalization. (2020, December 08). Retrieved January 13, 2021, from
https://en.wikipedia.org/wiki/Histogram_equalization
8. Sharma, D. Raju and S. Ranjan, "Detection of pneumonia clouds in chest X-ray using
image processing approach," 2017 Nirma University International Conference on
Engineering (NUiCONE), Ahmedabad, 2017, pp. 1-4, doi:
10.1109/NUICONE.2017.8325607.

