import keras
from keras.models import load_model
import cv2
import pyautogui as p  # to control the buttons in the keyboard
import numpy as np
import operator

#load the trained model
model=load_model("C:/Users/ELCOT/Downloads/OPENCV/Adjust Volume/volume_up_new_model.h5")

#initialise the camera 
cap=cv2.VideoCapture(0) 

#declaring the categorical output
categories={0:"volume up",1:"volumn down",2:'nothing'} 
#initialise infinite loop
while True:
       _,frame=cap.read() #reading video from a camera
       frame=cv2.flip(frame,1) #fliping the camera output
       x1=int(0.5*frame.shape[1]) #to create a ROI on the screen
       y1=10
       x2=frame.shape[1]-10
       y2=int(0.5*frame.shape[1])

       cv2.rectangle(frame,(x1-1,y1-1),(x2-1,y2-1),(255,0,0),1) # drawing the rectangle in the camera window
       roi=frame[y1:y2,x1:x2]
       roi=cv2.resize(roi,(50,50)) #resize the image
        #converting BGR to gray scale image
      
       result=model.predict(roi.reshape(1,50,50,3)) #making prediction from the loaded model
      
       prediction={
                    "volume up":result[0][0],   
                    "volumn down":result[0][1],
                    'nothing':result[0][2]
           }
       prediction=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True) #sorting of prediction on the basis of max accuracy
       cv2.putText(frame,prediction[0][0],(x1+100,y2+30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3) #showning text on the screen
       cv2.imshow("Frame",frame)
       print(prediction[0][0])
       if prediction[0][0]=="volume up":
                    p.press('up')  # it'll press the up arrow key to increase volume

       if prediction[0][0]=='volumn down':
                    p.press('down') # it'll press the down arrow key to decrease volume

       if prediction[0][0]=='nothing':
                    pass

       key=cv2.waitKey(10)
       if key & 0xFF == 27: #press esc for break
           break

cap.release() #switch off the camera
cv2.destroyAllWindows() #destroy camera windows
