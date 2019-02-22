import operator
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import math
import time
from datetime import datetime, timedelta
import pyautogui
import cv2, time, pandas
#capturing video through webcam
cap=cv2.VideoCapture(0)
end_time = datetime.now() + timedelta(seconds=3)
X = []
Y = []
A = []
B = []
blfilterX = []
blfilterY = []
static_back = None
tempx = 0
tempy = 0
# List when any moving object appear
motion_list = [ None, None ]

# Time of movement
time = []

# Initializing Dataimg, one column is start
# time and other column is end time
df = pandas.DataFrame(columns = ["Start", "End"])
type = 1
while(1):
    _, img = cap.read()
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    motion = 0
    # Converting color image to gray_scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Converting gray scale image to GaussianBlur
    # so that change can be find easily
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # In first iteration we assign the value
    # of static_back to our first img
    if static_back is None:
        static_back = gray
        continue

    # Difference between static background
    # and current img(which is GaussianBlur)
    diff_img = cv2.absdiff(static_back, gray)

    # If change in between static background and
    # current img is greater than 30 it will show white color(255)
    thresh_img = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_img = cv2.dilate(thresh_img, None, iterations = 2)

    # Finding contour of moving object
    (_, cnts, _) = cv2.findContours(thresh_img.copy(),
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle arround the moving object
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        tempx = (x+(x+w))/2
        tempy = (y+(y+h))/2

    # Appending status of motion
    # Displaying image in gray_scale
    #cv2.imshow("Gray img", gray)

    # Displaying the difference in currentimg to
    # the staticimg(very first_img)
    #cv2.imshow("Difference img", diff_img)

    # Displaying the black and white image in which if
    # intencity difference greater than 30 it will appear white
    #cv2.imshow("Threshold img", thresh_img)

    # Displaying color img with contour of motion of object
    #cv2.imshow("Color img", img)
	#definig the range of red color
	red_lower=np.array([136,87,111],np.uint8)
	red_upper=np.array([180,255,255],np.uint8)

	#defining the Range of Blue color
	blue_lower=np.array([99,115,150],np.uint8)
	blue_upper=np.array([110,255,255],np.uint8)

	#defining the Range of yellow color
	yellow_lower=np.array([22,60,200],np.uint8)
	yellow_upper=np.array([60,255,255],np.uint8)

	#finding the range of red,blue and yellow color in the image
	red=cv2.inRange(hsv, red_lower, red_upper)
	blue=cv2.inRange(hsv,blue_lower,blue_upper)
	yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)

	#Morphological transformation, Dilation
	kernal = np.ones((5 ,5), "uint8")

        red=cv2.dilate(red, kernal)
	res=cv2.bitwise_and(img, img, mask = red)

	blue=cv2.dilate(blue,kernal)
	res1=cv2.bitwise_and(img, img, mask = blue)

	yellow=cv2.dilate(yellow,kernal)
	res2=cv2.bitwise_and(img, img, mask = yellow)


	#Tracking the Red Color
	(_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>300):

			x,y,w,h = cv2.boundingRect(contour)
			#img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			#cv2.putText(img,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

	#Tracking the Blue Color
	(_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
        if(area>100):
            x,y,w,h = cv2.boundingRect(contour)
            if x <= tempx + 300 and x >= tempx-300 and y <= tempy+300 and y >= tempy-300:
                X.append([x])
                Y.append([y])
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
        pu = 0
        #filterq()
        #blfilterX = Removeq(blfilterX)
        #Remove()
        #Remove(blfilterY)
        if len(X)>12 :
            polynomial_features= PolynomialFeatures(degree=1)
            x_poly = polynomial_features.fit_transform(X)
            model = LinearRegression()
            model.fit(x_poly, Y)
            y_poly_pred = model.predict(x_poly)
            rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
            r2 = r2_score(Y,y_poly_pred)
            slope_of_blue = model.coef_[0][1]
            while(pu < len(X)):
				#img = cv2.rectangle(img,(int(X[pu][0]),int(y_poly_pred[pu][0])),(int(3+X[pu][0]),int(3+y_poly_pred[pu][0])),(255,0,0),3)
				img = cv2.line(img,(int(X[X.index(min(X))][0]),int(y_poly_pred[X.index(min(X))][0])),(int(X[X.index(max(X))][0]),int(y_poly_pred[X.index(max(X))][0])),(255,0,0),5)
				pu=pu+1
    	    #Tracking the yellow Color
	(_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
        if(area>50):
            x,y,w,h = cv2.boundingRect(contour)
            if x <= tempx + 300 and x >= tempx-300 and y <= tempy+300 and y >= tempy-300:
                A.append([x])
                B.append([y])
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
                cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))
            pu1 = 0
        if len(A)>10 :
			polynomial_featuresa= PolynomialFeatures(degree=1)
			xa_poly = polynomial_featuresa.fit_transform(A)
			modela = LinearRegression()
			modela.fit(xa_poly, B)
			ya_poly_pred = modela.predict(xa_poly)
			rmsea = np.sqrt(mean_squared_error(B,ya_poly_pred))
			r2a = r2_score(B,ya_poly_pred)
			slope_of_yellow = modela.coef_[0][1]
			while(pu1 < len(A)):
				img = cv2.line(img,(int(A[A.index(min(A))][0]),int(ya_poly_pred[A.index(min(A))][0])),(int(A[A.index(max(A))][0]),int(ya_poly_pred[A.index(max(A))][0])),(0,255,255),5)
				pu1=pu1+1
        if len(X)>10 and len(A)>8  :
            if slope_of_blue !=0  and slope_of_yellow!=0:
                if slope_of_blue + 0.3 > slope_of_yellow or slope_of_blue - 0.3 < slope_of_yellow :
                    if slope_of_blue <0 and slope_of_yellow<0 and type == 1 :
                        print('gesture_left')
                        pyautogui.keyDown('left')
                        pyautogui.keyUp('left')
                        type = 0
                    if slope_of_blue >0 and slope_of_yellow>0 and type == 1:
                        print('gesture_right')
                        pyautogui.keyDown('right')
                        pyautogui.keyUp('right')
                        type = 0
            #print(slope_of_blue)
            #print(slope_of_yellow)
        if end_time <= datetime.now() :
            end_time = datetime.now() + timedelta(seconds=3)
            #print("delete")
            while len(X) > 0 : X.pop()
            while len(Y) > 0 : Y.pop()
            while len(A) > 0 : A.pop()
            while len(B) > 0 : B.pop()
            #sprint(X)
            while len(blfilterX) > 0 : blfilterX.pop()
            while len(blfilterY) > 0 : blfilterY.pop()
            type = 1

        cv2.imshow("Color Tracking",img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
    		cap.release()
    		cv2.destroyAllWindows()

		#cv2.imshow("Color Tracking",img)
