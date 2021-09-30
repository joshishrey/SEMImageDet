#AUTHOR Shreya Joshi
import matplotlib.pyplot as plt
from tkinter import *
import cv2
import numpy as np
import tkinter as tk
# Read image.

image='IFSGTestsAugust_m09.tif'

img = cv2.imread(image, cv2.IMREAD_COLOR)
fullImage=img.copy()
img = img[-65:, 700:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh=cv2.threshold(img,165,255,cv2.THRESH_BINARY_INV)
contours,hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

dCLarge=cv2.drawContours(img,contours,-1,(255,0,0),1)
scale=[]
for i in range(len(contours)):
    rectangle=cv2.boundingRect(contours[i])
    area = cv2.contourArea(contours[i])  
    if area>500:
        if area<8200:
            sizeRect=rectangle
            scale.append(contours[i])
            dCRect=cv2.rectangle(thresh,(rectangle[0],rectangle[1]),(rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]),(0,255,0),2)
        
#PixToNM=300/(sizeRect[2]-sizeRect[0])
PixToNM=5000/(sizeRect[2])
cv2.imshow('thresh',thresh)



img = cv2.imread(image, cv2.IMREAD_COLOR)
sampleDes='SEM sample,  directly from the generator'




img = img[0:900, :]
img2=img
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur=cv2.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT)

canny=cv2.Canny(blur, 85,175)

dilated= cv2.dilate(canny,(7,7), iterations=1)

k=0
tvar=124
bvar=1
while(k!=13):
    print(k)
    if(k==112):
        tvar+=2
    elif(k==111):
        tvar-=2
    elif(k==108):
        bvar+=2
    elif(k==107):
        if(bvar>1):
            bvar-=2
    print('tvar'+str(tvar))
    print('bvar'+str(bvar))
    blur=cv2.medianBlur(gray,(bvar))    
    ret, thresh=cv2.threshold(blur,tvar,255,cv2.THRESH_BINARY_INV)
    #adthresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,301,1)
    cv2.imshow('normalThresh', thresh)
    

#ret, thresh=cv2.threshold(gray,185,255,cv2.THRESH_BINARY_INV)
#thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,301,1)

#thresh=cv2.GaussianBlur(thresh,(1,1),cv2.BORDER_DEFAULT)



#cv2.imshow('Original', gray)
#cv2.imshow('thresh', thresh)
#cv2.waitKey(0)
#cv2.imshow('blur', blur)

    contours,hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largeCont=[]
    rectangle=[]
    Solidity=[]
    for cnt in contours:              
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        

        if area > 50:
            if area < 11000:
                largeCont.append(cnt)
                rectangle.append(cv2.boundingRect(cnt))
                solidity = float(area)/hull_area
                Solidity.append(solidity)
                
    #dC=cv2.drawContours(img,contours,-1,(0,255,0),1)
    #cv2.imshow('dc', dC)
    fm=fullImage.copy()
    img=cv2.drawContours(fm,largeCont,-1,(255,0,0),1)
    cv2.imshow('img',img)
    k=cv2.waitKey(0)
dCRect=cv2.rectangle(dCLarge,(rectangle[0][0],rectangle[0][1]),(rectangle[0][0]+rectangle[0][2],rectangle[0][1]+rectangle[0][3]),(0,0,255),1)
AR=[]
CompactCount=0
LaceyCount=0
##for i in range(len(rectangle)):
##    if(rectangle[i][3]>rectangle[i][2]):
##        AR.append(rectangle[i][2]/rectangle[i][3])
##        ar=rectangle[i][2]/rectangle[i][3]
##    else:
##        AR.append(rectangle[i][3]/rectangle[i][2])
##        ar=rectangle[i][3]/rectangle[i][2]
##    if ar <0.8:
##        dCRect=cv2.rectangle(dCLarge,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,0,255),1)
##        LaceyCount+=1
##    else:
##        dCRect=cv2.rectangle(dCLarge,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(255,0,0),1)
##        CompactCount+=1
#root = tk.Tk()
  

import tkinter.messagebox
manual = tk.messagebox.askyesno(title='confirmation',
                    message='Do you want to manually check classification?\n In manual Mode : \n\tFor correct classification press Enter.\r\t For incorrect classification press Esc \r\t For non soot selections press X. "')


isLacey=[]
for i in range(len(rectangle)):
    if(rectangle[i][3]>rectangle[i][2]):
        AR.append(rectangle[i][2]/rectangle[i][3])
        ar=rectangle[i][2]/rectangle[i][3]
    else:
        AR.append(rectangle[i][3]/rectangle[i][2])
        ar=rectangle[i][3]/rectangle[i][2]

    img2=fullImage.copy()
    cv2.putText(img2,sampleDes, (10,680), 1, 1,(255,255,255),2)
    cv2.putText(img2,'Total number of particles:'+str(len(largeCont)), (10,700), 1, 1,(255,255,255),2)
    cv2.putText(img2,'Total number of compact particles:'+str(CompactCount), (10,720), 1, 1,(0,0,0),2)
    cv2.putText(img2,'Total number of Lacey particles:'+str(LaceyCount), (10,740), 1, 1,(255,255,255),2)
    if Solidity[i] <0.8:
        dCRect=cv2.rectangle(img2,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,255,0),1)
        
        if manual:
            dCRect=cv2.putText(dCRect,'Lacey?', (rectangle[i][0],rectangle[i][1]-8), 1, 1,(255,255,255),2)
            cv2.imshow('Added rectangle',dCRect)
        
            k=cv2.waitKey(0)
            print(k)
        
            if(k==13):
                LaceyCount+=1
                isLacey.append(1)
            elif(k==120):
                isLacey.append(2)
            else:
                CompactCount+=1
                isLacey.append(0)
        else:
            LaceyCount+=1
            isLacey.append(1)
    else:
        dCRect=cv2.rectangle(img2,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,0,255),1)
        if manual:
            
            dCRect=cv2.putText(dCRect,'Compact?', (rectangle[i][0],rectangle[i][1]-8), 1, 1,(255,255,255),2)
            cv2.imshow('Added rectangle',dCRect)
            
            k=cv2.waitKey(0)
            print(k)
            
            if(k==13):
                CompactCount+=1
                isLacey.append(0)
            elif(k==120):
                isLacey.append(2)
            else:
                LaceyCount+=1
                isLacey.append(1)
        else:
            CompactCount+=1
            isLacey.append(0)




img2= cv2.imread(image, cv2.IMREAD_COLOR)
#img2 = img2[0:800, 0:1200]
gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
selectedRect=[]
area=[]
selectedSolidity=[]
for i in range(len(rectangle)):
    if(isLacey[i]==1):
        dCRect=cv2.rectangle(img2,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,255,0),1)

        selectedRect.append(rectangle[i])
        area.append(cv2.contourArea(largeCont[i]))
        equDiameter=2*((cv2.contourArea(largeCont[i])*PixToNM*PixToNM)/np.pi)**(1/2)
        dCRect=cv2.putText(dCRect,"{:.3f}".format(Solidity[i]), (rectangle[i][0],rectangle[i][1]-8), 1, 1,(255,255,255),1)
        selectedSolidity.append(Solidity[i])
       
   
    elif isLacey[i]==0:
        dCRect=cv2.rectangle(img2,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,0,255),1)
        selectedRect.append(rectangle[i])
        area.append(cv2.contourArea(largeCont[i]))
        equDiameter=2*((cv2.contourArea(largeCont[i])*PixToNM*PixToNM)/np.pi)**(1/2)
        dCRect=cv2.putText(dCRect,"{:.3f}".format(Solidity[i]), (rectangle[i][0],rectangle[i][1]-8), 1, 1,(255,255,255),1)
        selectedSolidity.append(Solidity[i])

particleSize=[]
for i in selectedRect:
    if(i[3]>i[2]):
        particleSize.append(i[3])
    else:
        particleSize.append(i[2])

print('Total number of particles:'+str(len(largeCont)))
print('Total number of compact particles:'+str(CompactCount))
print('Total number of Lacey particles:'+str(LaceyCount))
cv2.putText(img2,sampleDes, (10,680), 1, 1,(255,255,255),2)
cv2.putText(img2,'Total number of particles:'+str(len(largeCont)), (10,700), 1, 1,(255,255,255),2)
cv2.putText(img2,'Total number of compact particles:'+str(CompactCount), (10,720), 1, 1,(0,0,0),2)
cv2.putText(img2,'Total number of Lacey particles:'+str(LaceyCount), (10,740), 1, 1,(255,255,255),2)

ret, thresh2=cv2.threshold(gray2,165,255,cv2.THRESH_BINARY_INV)

img2=cv2.drawContours(img2,largeCont,-1,(255,0,0),1)
#circles = cv2.HoughCircles(gray2,cv2.HOUGH_GRADIENT,1,20,
#                            param1=50,param2=30,minRadius=0,maxRadius=30)


#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
#    # draw the outer circle
#    cv2.circle(dCLarge,(i[0],i[1]),i[2],(0,255,0),2)
#    # draw the center of the circle
#    cv2.circle(dCLarge,(i[0],i[1]),2,(0,0,255),2)
    
#cv2.imshow('detected circles',gray)
fig, ax = plt.subplots(1,1,figsize=(10,10))
area=np.asarray(area)
equDiameter=2*((area*PixToNM*PixToNM)/np.pi)**(1/2)
hist=ax.hist(equDiameter[1:], bins='auto')
ax.grid( which='minor', axis='both')
ax.grid( which='major', axis='both')
ax.set_xlabel("Area equivalent Diameter (nm)")
ax.set_ylabel("Count")
#ax.set_xscale('log')
fig.suptitle(sampleDes+'Area equivalent diameter(nm) histogram', fontsize=16)

fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
solidityHist=ax2.hist(selectedSolidity[1:], bins='auto')
ax2.grid( which='minor', axis='both')
ax2.grid( which='major', axis='both')
ax2.set_xlabel("Solidity")
ax2.set_ylabel("Count")
fig2.suptitle(sampleDes+'Solidity histogram', fontsize=16)
#cv2.imshow('dCLarge', dCRect)
#cv2.imshow('gray2', gray2)
cv2.imshow('img2', img2)
#cv2.waitKey(0)
plt.show()
cv2.destroyAllWindows()
