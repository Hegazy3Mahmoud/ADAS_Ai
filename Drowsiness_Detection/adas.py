import cv2
import numpy as np
import tensorflow as tf
from pygame import mixer
import os

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="bestModel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

lbl=['Close','Open']


path = os.getcwd()
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Speed.mp4')

# Set resolution and frame rate
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 150)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(150,150))
    left_eye = leye.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(50,50))
    right_eye =  reye.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(50,50))


    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(64,64))
        r_eye=  r_eye.reshape((-1, 64, 64, 1))
        r_eye = r_eye/255.0


        input_data = np.array(r_eye, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        rpred = output_data[0]

        print(rpred)
        if(rpred[0]>0.5):
            lbl='Open' 
        if(rpred[0]<=0.5):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(64,64))
        l_eye=l_eye.reshape((-1, 64, 64, 1))
        l_eye = l_eye/255.0

        input_data = np.array(l_eye, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        lpred = output_data[0]
        
        print(lpred)
        if(lpred[0]>0.5):
            lbl='Open'   
        if(lpred[0]<=0.5):
            lbl='Closed'
        break
    

    if(rpred[0]<=0.5 and lpred[0]<=0.5):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>12):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        #try:
        #    sound.play()
            
        #except:  # isplaying = False
        #    pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
