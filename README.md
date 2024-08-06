#ADAS (Advanced Driver Assistance System) Project

### Overview
This project aims to develop an Advanced Driver Assistance System (ADAS) using various computer vision models. The system includes five models:

1-Drowsiness Detection

2-Face Verification

3-Lane Detection

4-Person Detection

5-Traffic Sign Detection


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
Box Loss (Training):

Decreased from 1.75 at Epoch 1 to 1.14 at Epoch 35, indicating a steady improvement in the model’s ability to predict bounding boxes more accurately over time.
Classification Loss (Training):

Decreased from 10.93 at Epoch 1 to 0.8678 at Epoch 35, showing a significant enhancement in class prediction accuracy as the model trained.
DFL Loss (Training):

Decreased from 2.297 at Epoch 1 to 1.807 at Epoch 35, reflecting refinement in the model’s distributional focal loss over the epochs.
Precision (Validation, B):

Ranged from 0.396 at Epoch 1 to 0.912 at Epoch 35, demonstrating increased accuracy in predicting true positives throughout the training.
Recall (Validation, B):

Varied between 0.0806 at Epoch 1 and 0.867 at Epoch 35, indicating improvement in the model’s ability to detect most positive instances.
mAP50 (Validation, B):

Improved from 0.117 at Epoch 1 to 0.930 at Epoch 35, showing better overall performance across classes.
mAP50-95 (Validation, B):

Increased from 0.0976 at Epoch 1 to 0.814 at Epoch 35, reflecting enhanced performance considering different IoU thresholds.
Box Loss (Validation):

Dropped from 1.75 at Epoch 1 to 1.14 at Epoch 35, indicating better validation results over time.
Classification Loss (Validation):

Decreased from 10.93 at Epoch 1 to 0.8678 at Epoch 35, showing improved validation accuracy.
DFL Loss (Validation):

Slightly reduced from 2.297 at Epoch 1 to 1.807 at Epoch 35.
Learning Rate:

Gradually decreased from 0.000154 at Epoch 1 to 0.000020 at Epoch 35, following the learning rate schedule.
This summary illustrates the model's performance metrics evolving positively over the epochs, showing consistent improvements in precision, recall, and overall accuracy while the losses decreased, reflecting enhanced model performance.

![download](https://github.com/user-attachments/assets/0f8e6622-193a-432f-a163-65b9d237a12b)


##### Here is a sample of predicted images from the inference phase :

![image](https://github.com/user-attachments/assets/b9dedeed-965e-487a-84f0-20c4eadf58cd)


##### Finally, I saved the model in ONNX format using Valid_model.export(format="onnx"). It's now ready for production and deployment.


