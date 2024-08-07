# ADAS (Advanced Driver Assistance System) Project

### Overview
This project aims to develop an Advanced Driver Assistance System (ADAS) using various computer vision models. The system includes five models:

1-Drowsiness Detection

2-Face Verification

3-Lane Detection

4-Person Detection

5-Traffic Sign Detection




## Drowsiness Detection
This model consist of four models :
* Haar Cascade Classifiers
    * Face
    * Left Eye
    * Right Eye
* CNN model to detect the eye states
### Dataset
This dataset is just one part of The MRL Eye Dataset, the large-scale dataset of human eye images. It is prepared for classification tasks This dataset contains infrared images in low and high resolution, all captured in various lighting conditions and by different devices. The dataset is suitable for testing several features or trainable classifiers. In order to simplify the comparison of algorithms, the images are divided into several categories, which also makes them suitable for training and testing classifiers.
The full dataset is available here : http://mrl.cs.vsb.cz/eyedataset
Our dataset includes 84K images for both closed and open eye categories.
* Training images : 44800
* Validation images : 11200
* Test images : 28000
* Orginal size (86,86,3)
* Resized (64,64,1) -> smaller image size means less training time !
Dataset has been already balanced,i.e both categories have same num of images. Thus, we shall only look at Accuracy metric.

## Model Architecture Summary

| Layer Type             | Output Shape         | Parameters  |
|------------------------|-----------------------|-------------|
| Conv2D                 | (None, 60, 60, 32)    | 832         |
| Conv2D                 | (None, 56, 56, 32)    | 25,600      |
| BatchNormalization     | (None, 56, 56, 32)    | 128         |
| MaxPooling2D           | (None, 28, 28, 32)    | 0           |
| Dropout                | (None, 28, 28, 32)    | 0           |
| Conv2D                 | (None, 26, 26, 64)    | 18,496      |
| Conv2D                 | (None, 24, 24, 64)    | 36,864      |
| BatchNormalization     | (None, 24, 24, 64)    | 256         |
| MaxPooling2D           | (None, 12, 12, 64)    | 0           |
| Dropout                | (None, 12, 12, 64)    | 0           |
| Flatten                | (None, 9216)          | 0           |
| Dense                  | (None, 256)           | 2,359,296   |
| BatchNormalization     | (None, 256)           | 1,024       |
| Dense                  | (None, 128)           | 32,768      |
| Dense                  | (None, 84)            | 10,752      |
| BatchNormalization     | (None, 84)            | 336         |
| Dropout                | (None, 84)            | 0           |
| Dense (Output)         | (None, 1)             | 85          |

**Total Parameters:** 2,486,437 (9.49 MB)

**Trainable Parameters:** 2,485,565 (9.48 MB)

**Non-trainable Parameters:** 872 (3.41 KB)

## Training Epoch Details

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|---------------|-------------------|-----------------|---------------------|
| 1     | 0.1545        | 94.0%             | 0.1368          | 95.1%               |
| 2     | 0.0758        | 97.4%             | 0.0912          | 96.5%               |
| 3     | 0.0626        | 97.9%             | 0.1048          | 96.5%               |
| 4     | 0.0544        | 98.1%             | 0.0404          | 98.5%               |
| 5     | 0.0447        | 98.5%             | 0.1544          | 93.4%               |
| 6     | 0.0418        | 98.6%             | 0.0812          | 97.4%               |
| 7     | 0.0353        | 98.7%             | 0.0610          | 97.9%               |
| 8     | 0.0328        | 98.9%             | 0.0454          | 98.3%               |
| 9     | 0.0299        | 98.9%             | 0.0457          | 98.6%               |
| 10    | 0.0296        | 99.0%             | 0.0803          | 97.1%               |
| 11    | 0.0234        | 99.2%             | 0.0430          | 98.7%               |
| 12    | 0.0224        | 99.3%             | 0.2071          | 91.5%               |
| 13    | 0.0231        | 99.2%             | 0.2144          | 93.4%               |
| 14    | 0.0205        | 99.3%             | 0.0495          | 98.3%               |
| 15    | 0.0194        | 99.3%             | 0.0551          | 98.5%               |
| 16    | 0.0159        | 99.4%             | 0.0569          | 98.2%               |
| 17    | 0.0188        | 99.3%             | 0.0453          | 98.7%               |
| 18    | 0.0159        | 99.4%             | 0.0379          | 98.9%               |
| 19    | 0.0140        | 99.5%             | 0.0489          | 98.7%               |
| 20    | 0.0133        | 99.5%             | 0.0501          | 98.6%               |
| 21    | 0.0133        | 99.5%             | 0.0586          | 98.5%               |
| 22    | 0.0144        | 99.5%             | 0.0655          | 98.2%               |
| 23    | 0.0120        | 99.6%             | 0.0582          | 98.7%               |
| 24    | 0.0142        | 99.5%             | 0.0533          | 98.8%               |
| 25    | 0.0106        | 99.6%             | 0.0485          | 98.9%               |
| 26    | 0.0109        | 99.6%             | 0.0544          | 98.6%               |
| 27    | 0.0158        | 99.5%             | 0.0505          | 98.8%               |
| 28    | 0.0113        | 99.6%             | 0.0518          | 98.8%               |
| 29    | 0.0111        | 99.6%             | 0.0514          | 98.8%               |
| 30    | 0.0101        | 99.6%             | 0.0612          | 98.4%               |


### Confusion Matrix
![d15a7840-433f-469d-956d-83136e7a5149](https://github.com/user-attachments/assets/cfd12c38-5fa1-4115-85da-9d97aa5077e0)

### Saving the model
I saved the in h5 format then convert it to tflite for the inference phase

### The Application Output
## On laptop
### TensorFlow Model
![LaptopTensor](https://github.com/user-attachments/assets/a53f5726-b9c7-4621-a79b-ba476ca5b104)
### TensorFlow Lite Model
![LaptopTensorlite](https://github.com/user-attachments/assets/fd88fb54-64b0-44de-8daf-2b246727f966)

## On Raspberry Pi 3B+ Using TensorFlow Lite
![RaspberryPiOutput](https://github.com/user-attachments/assets/d41a77c9-45d1-4de9-9ca9-e55762b4d941)





## 🚗 Lane Detection Using Hough Transform

This project implements a robust lane detection system using image processing techniques in OpenCV. The main steps include preprocessing the image, applying a region of interest mask, detecting edges using the Canny edge detector, and using the Hough Transform to detect lane lines.

![Lane Detection](https://github.com/user-attachments/assets/1dfe5cf2-d373-4886-9a88-c82b3b90b869)

#### 📋 Table of Contents
- [🚀 Introduction](#introduction)
- [🌟 Features](#features)
- [⚡ Installation](#installation)
- [📷 Usage](#usage)
- [📊 Results](#results)
- [🤝 Contributing](#contributing)

#### 🚀 Introduction

![Screenshot 2024-08-07 005002](https://github.com/user-attachments/assets/6f74f151-92f1-4793-aa09-0d06730c7fb8)

![Screenshot 2024-08-07 004458](https://github.com/user-attachments/assets/5ce441d7-cad4-4866-8bb1-8ade2b9feb2f)

Lane detection is a critical component in autonomous driving systems. This project aims to detect lane lines in images and videos by leveraging various image processing techniques. The pipeline includes the following steps:

1. **Preprocessing the Image:** Enhance edges for better detection.
2. **Applying a Mask:** Isolate the region of interest to focus on the road.
3. **Detecting Edges:** Use the Canny edge detector to highlight lane lines.
4. **Using Hough Transform:** Detect lines in the image.
5. **Drawing Detected Lane Lines:** Overlay the detected lane lines on the original image.

#### 🌟 Features

- **Edge Detection:** Utilize the powerful Canny edge detector to highlight lane lines.
- **Region of Interest Masking:** Focus on the road by masking irrelevant areas.
- **Line Detection:** Implement the Hough Transform to find lane lines.
- **Real-time Visualization:** Overlay detected lane lines on the original image.

### ⚡ Installation

#### Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

#### Install Required Packages

```bash
pip install numpy opencv-python matplotlib
```

#### 📷 Usage

The Hough Transform is a versatile image processing technique with numerous applications beyond lane detection. Here are some additional use cases:

#### 🚀 Biometric and Man-Machine Interaction

- **Biometric Applications:** The Hough Transform can be employed in biometric systems such as face recognition or fingerprint recognition. It helps in detecting and localizing specific features within an image, which is crucial for accurate biometric identification.
- **Gesture Recognition:** In man-machine interaction applications, the Hough Transform can detect and track the position and movement of specific body parts, enhancing gesture recognition systems.

#### 🌐 3D Applications

- **3D Shape Detection:** The Hough Transform can be extended to three-dimensional spaces to detect geometric shapes such as planes or spheres in 3D images or point clouds. This capability is valuable in applications like 3D modeling, robotics, and advanced computer vision.

#### 🏷️ Object Recognition

- **Geometric Shape Detection:** By detecting specific geometric shapes in images or point clouds, the Hough Transform facilitates object recognition. This application is useful for automated inspection, surveillance systems, and autonomous vehicles.

#### 🔄 Object Tracking

- **Tracking Over Time:** The Hough Transform can be used to track specific geometric shapes across video frames. This functionality is beneficial for surveillance, autonomous driving, and tracking systems.

#### 🌊 Underwater Applications

- **Underwater Object Detection:** In underwater scenarios, such as sonar imaging, the Hough Transform can detect and localize geometric shapes. This capability is important for underwater object detection and mapping.

#### 🏭 Industrial and Commercial Applications

- **Quality Control and Defect Detection:** In manufacturing and quality control, the Hough Transform can be applied to detect and localize defects or irregularities in products, enhancing inspection processes.

Feel free to explore these applications and adapt the Hough Transform to suit various domains!

#### 📊 Results

![annotated_test1](https://github.com/user-attachments/assets/507ad1a1-6994-4de7-8079-e19faa4c4955)

![annotated_straight_lines2](https://github.com/user-attachments/assets/f30cc224-fa48-489a-80fc-0a312f3abdbb)

![annotated_straight_lines1](https://github.com/user-attachments/assets/a9e0a5af-385f-4057-9a4c-3f545a6973c2)

##### 🤝 Contributing

We welcome contributions from the community! If you have suggestions, improvements, or bug fixes, please fork the repository and create a pull request. For major changes, please open an issue to discuss your ideas before making a pull request.






### Traffic Sign Detection

##### Dataset
The dataset used for Traffic Sign Detection is "Signs Detection For Self-Driving Cars (Computer Vision Project)" from Kaggle. You can access it here: https://www.kaggle.com/datasets/pkdarabi/cardetection


##### About Dataset

Name of Classes: Green Light, Red Light, Speed Limit 10, Speed Limit 100, Speed Limit 110, Speed Limit 120, Speed Limit 20, Speed Limit 30, Speed Limit 40, Speed Limit 50, Speed Limit 60, Speed Limit 70, Speed Limit 80, Speed Limit 90, Stop  (15 classes in total).

##### Here are a few use cases for this project:


- Autonomous Vehicle Navigation: The model can be used in self-driving car systems to recognize traffic signs accurately. This would enable autonomous vehicles to follow traffic rules and regulations, analyzing every sign whether it’s about speed limit or stop-and-go indications to navigate the roads safely.

- Traffic Rule Compliance: This model can be used in driver assistance systems to ensure that drivers comply with all traffic rules. Alerts can be generated when drivers exceed the speed limit or don't stop at red lights, fostering safer roads.

  You can use the following link to download this dataset in other formats and also to access its original file:
  
https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou

  ![inbox_14850461_69bf06b5bd76b597644880af6c884260_ssss](https://github.com/user-attachments/assets/4e132626-7e7c-436d-9fe8-498b204896a4)


i used YOLOv10-N in this dataset to make boundry box and detect the signs 

my model: Model	Test Size	#Params	FLOPs
YOLOv10-N	640	2.3M	6.7G	

![latency](https://github.com/user-attachments/assets/b650f300-f572-446a-9449-50989bebc9f2)


##### There is some visualization to showcase my model's metrics :

###### 35 epochs completed in 0.906 hours via T-4 GPU (Co-lab).
Epochs 1-35: Training Metrics Summary


-Box Loss (Training):
Decreased from 1.75 at Epoch 1 to 1.14 at Epoch 35, indicating a steady improvement in the model’s ability to predict bounding boxes more accurately over time.


-Classification Loss (Training):
Decreased from 10.93 at Epoch 1 to 0.8678 at Epoch 35, showing a significant enhancement in class prediction accuracy as the model trained.


-DFL Loss (Training):
Decreased from 2.297 at Epoch 1 to 1.807 at Epoch 35, reflecting refinement in the model’s distributional focal loss over the epochs.


-Precision (Validation, B):
Ranged from 0.396 at Epoch 1 to 0.912 at Epoch 35, demonstrating increased accuracy in predicting true positives throughout the training.


-Recall (Validation, B):
Varied between 0.0806 at Epoch 1 and 0.867 at Epoch 35, indicating improvement in the model’s ability to detect most positive instances.


-mAP50 (Validation, B):
Improved from 0.117 at Epoch 1 to 0.930 at Epoch 35, showing better overall performance across classes.


-mAP50-95 (Validation, B):
Increased from 0.0976 at Epoch 1 to 0.814 at Epoch 35, reflecting enhanced performance considering different IoU thresholds.


-Box Loss (Validation):
Dropped from 1.75 at Epoch 1 to 1.14 at Epoch 35, indicating better validation results over time.


-Classification Loss (Validation):
Decreased from 10.93 at Epoch 1 to 0.8678 at Epoch 35, showing improved validation accuracy.


-DFL Loss (Validation):
Slightly reduced from 2.297 at Epoch 1 to 1.807 at Epoch 35.
Learning Rate:


-This summary illustrates the model's performance metrics evolving positively over the epochs, showing consistent improvements in precision, recall, and overall accuracy while the losses decreased, reflecting enhanced model performance.


![download](https://github.com/user-attachments/assets/0f8e6622-193a-432f-a163-65b9d237a12b)


##### Here is a sample of predicted images from the inference phase :

![image](https://github.com/user-attachments/assets/a85ff8e0-d668-4b27-b7e9-240ce509b766)



#### Finally, I saved the model in ONNX format using Valid_model.export(format="onnx"). It's now ready for production and deployment.

### This notebook was created in Colab.


## To test the model , open "try_out_model_on_vid&images.ipynb" file , place model.onnx and your image or video in the directory. The results will be saved in the runs directory.
