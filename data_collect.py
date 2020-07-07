import cv2
import numpy as np
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

sampleNum=0
while(True):
    ret, img = cam.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("Images/"+ str(sampleNum) + ".jpg", img[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
        cv2.waitKey(100) 
    cv2.imshow('face',img)
    cv2.waitKey(1)
    
    # break if the sample number is morethan 20
    if sampleNum==1001:
        break
cam.release()
cv2.destroyAllWindows()