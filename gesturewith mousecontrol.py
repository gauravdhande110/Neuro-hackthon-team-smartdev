import operator
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import math
import time
from datetime import datetime,timedelta 
import pyautogui
from pynput.mouse import Button, Controller
mouse=Controller()
#capturing video through webcam
#time.sleep(5)
end_time = datetime.now() + timedelta(seconds=3)
cap=cv2.VideoCapture(0)
X = []
Y = []
A = []
B = []
green_lowerBound=np.array([33,80,40])
green_upperBound=np.array([102,255,255])
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
pinchFlag=0
sx=1000
sy=700
(camx,camy)=(320,240)
slope_of_yellow=0
green_len=0

pyautogui.moveTo(250,250)
pyautogui.click()

check =1
checkg=1 
while(1):
	_, img = cap.read()
		
	#converting frame(img i.e BGR) to HSV (hue-saturation-value)
	img=cv2.resize(img,(340,220))
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

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
	
	mask=cv2.inRange(hsv,green_lowerBound,green_upperBound)
	maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
	maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
	maskFinal=maskClose
	conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	if(len(conts)==2):
		if(pinchFlag==1):
			pinchFlag=0
			#mouse.release(Button.left)
		x1,y1,w1,h1=cv2.boundingRect(conts[0])
		x2,y2,w2,h2=cv2.boundingRect(conts[1])
		cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
		cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
		cx1=x1+w1/2
		cy1=y1+h1/2
		cx2=x2+w2/2
		cy2=y2+h2/2
		cx=(cx1+cx2)/2
		cy=(cy1+cy2)/2
		cv2.line(img, (cx1,cy1),(cx2,cy2),(255,0,0),2)
		cv2.circle(img, (cx,cy),2,(0,0,255),2)
		ta=(pow(cx2,2)-(2*(cx2*cx1))+pow(cx1,2))
		tb=(pow(cy2,2)-(2*(cy1*cy2))+pow(cy1,2))
		green_len = math.sqrt(ta+tb)
		print("green",green_len)
		mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
		if green_len < 150 :	
			pyautogui.moveTo(sx-(cx*sx/camx),cy*sy/camy) 
			while mouse.position!=mouseLoc:
				pass
	elif(len(conts)==1):
		x,y,w,h=cv2.boundingRect(conts[0])
		if(pinchFlag==0):
			pinchFlag=1
            #mouse.press(Button.left)
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cx=x+w/2
		cy=y+h/2
		cv2.circle(img,(cx,cy),(w+h)/4,(0,0,255),2)
		mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
		pyautogui.moveTo(sx-(cx*sx/camx),cy*sy/camy)        
		#mouse.position=mouseLoc 
		while pyautogui.position()!=mouseLoc:
			pass
	#cv2.imshow("cam",img)
	cv2.waitKey(5)

	#Morphological transformation, Dilation
	kernal = np.ones((5 ,5), "uint8")

        red=cv2.dilate(red, kernal)
	res=cv2.bitwise_and(img, img, mask = red)

	blue=cv2.dilate(blue,kernal)
	res1=cv2.bitwise_and(img, img, mask = blue)

	yellow=cv2.dilate(yellow,kernal)
	res2=cv2.bitwise_and(img, img, mask = yellow)


	#Tracking the Red Color
	(contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>300):

			x,y,w,h = cv2.boundingRect(contour)
			#img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			#cv2.putText(img,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

	#Tracking the Blue Color
	(contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>100):

			x,y,w,h = cv2.boundingRect(contour)
			X.append([x])
			Y.append([y])
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
		pu = 0

		if len(X)>15 :
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
	(contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>100):
			x,y,w,h = cv2.boundingRect(contour)
			A.append([x])
			B.append([y])
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
			cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))
			
		pu1 = 0

		if len(A)>15 :
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
		#print(rmse)
		#print(r2)

		#plt.scatter(X, Y, s=10)
		# sort the values of x before line plot
		#sort_axis = operator.itemgetter(0)
		#sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
		#X, y_poly_pred = zip(*sorted_zip)
		#plt.plot(X, y_poly_pred, color='m')
		#cv2.imshow("Redcolour",red)
		#for mouse click
		if len(conts)==2 and slope_of_yellow>-1 and slope_of_yellow<1 and slope_of_yellow!=0 and green_len<150 and checkg==1 :
			print('click')
			#mouse.press(Button.left)
			pyautogui.keyDown('ctrl')
			pyautogui.keyUp('ctrl')
			#time.sleep(0.5)
			checkg=0

		if len(X)>15 and len(A)>15 :
			if slope_of_blue !=0  and slope_of_yellow!=0:

			    if slope_of_blue <0 and slope_of_yellow<0 and check==1:
					print('gesture_left')
					cv2.putText(img,"Left",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))
					pyautogui.keyDown('left')
					pyautogui.keyUp('left')
					while len(X) > 0 : X.pop() 					
					while len(Y) > 0 : Y.pop()
					while len(A) > 0 : A.pop()
					while len(B) > 0 : B.pop()
					check =0 
					#time.sleep(2)  
			    if slope_of_blue >0 and slope_of_yellow>0 and check ==1:
					print('gesture_right')
					cv2.putText(img,"right",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))
					pyautogui.keyDown('right')
					pyautogui.keyUp('right')
					while len(X) > 0 : X.pop() 					
					while len(Y) > 0 : Y.pop()
					while len(A) > 0 : A.pop()
					while len(B) > 0 : B.pop()
					check =0 
					#time.sleep(2)  
										
			#print(slope_of_blue)
			print(slope_of_yellow)
		if end_time <= datetime.now():
			end_time=datetime.now()+timedelta(seconds=3)        
			while len(X) > 0 : X.pop() 					
			while len(Y) > 0 : Y.pop()
			while len(A) > 0 : A.pop()
			while len(B) > 0 : B.pop()
			check=1
			checkg=1
		cv2.imshow("Color Tracking",img)
    	#cv2.imshow("red",res)
    	if cv2.waitKey(10) & 0xFF == ord('q'):
    		cap.release()
    		cv2.destroyAllWindows()
    		break

		#cv2.imshow("Color Tracking",img)


#print(X)
#print(Y)
'''
C = 15
R = 30
i=0
length =len(X)
print(length)

while(i < length):
	counter=0
	j=0
	while(j<length) :
		r =math.sqrt(((X[i][0]-X[j][0])**2) +((Y[i][0]-Y[j][0])**2))
		if r > R :
			counter=counter+1
		j=j+1
		k=0
	if counter >= C :
		while(k<length) :
			if X[k][0]<(X[i][0]+R) and Y[k][0]<(Y[i][0]+R) and X[k][0]>(X[i][0]-R) and Y[k][0]>(Y[i][0]-R) :
				X.pop(k)
				Y.pop(k)
				#print("in am in")
				length = len(X)
				#print(len(X))
			k=k+1
			#print(k)
		X.pop(i)
		Y.pop(i)
		length = len(X)
	length = len(X)




	i=i+1
print(len(X))
'''
'''
if len(X)!= 0 :
    polynomial_features= PolynomialFeatures(degree=1)
    x_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(x_poly, Y)
    y_poly_pred = model.predict(x_poly)
    rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
    r2 = r2_score(Y,y_poly_pred)
    slope_of_blue = model.coef_[0][1]
#print(rmse)
#print(r2)
    #plt.scatter(X, Y, s=10)
# sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
#sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
#X, y_poly_pred = zip(*sorted_zip)
    #plt.plot(X, y_poly_pred, color='b')

if len(A)!= 0 :
    polynomial_featuresa= PolynomialFeatures(degree=1)

#print(rmsea)
#print(r2a)
    slope_of_yellow = modela.coef_[0][1]
    plt.scatter(A, B, s=10)
# sort the values of x before line plot
    sort_axisa = operator.itemgetter(0)
#sorted_zipa = sorted(zip(Xx,ya_poly_pred), key=sort_axisa)
#xa, ya_poly_pred = zip(*sorted_zipa)
    plt.plot(A, ya_poly_pred, color='y')


# giving a title to my graph
#plt.title('My first graph!')
'''
## function to show the plot
#plt.show()

# create linear regression object
###Y.reshape()
#model2.fit(X, Y)

#plt.scatter(X, Y,color='g')
#lt.plot(X, model2.predict(X),color='k')
#print(slope_of_blue)
#print(slope_of_yellow)
'''
if slope_of_blue !=0  and slope_of_yellow!=0:

    if slope_of_blue <0 and slope_of_yellow<0:
        print('gesture_right')
    if slope_of_blue >0 and slope_of_yellow>0:
        print('gesture_left')
print(slope_of_blue)
print(slope_of_yellow)
plt.show()
'''
