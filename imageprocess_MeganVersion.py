#AUTHOR Shreya Joshi
import matplotlib.pyplot as plt
from tkinter import *
import cv2
import numpy as np
import tkinter as tk
# Read image.
import pandas as pd

image='07022016_2015A5_T75_6.tif'

img = cv2.imread(image, cv2.IMREAD_COLOR)
fullImage=img.copy()
img = img[-65:, 700:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh=cv2.threshold(img,165,255,cv2.THRESH_BINARY_INV)
contours,hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

dCLarge=cv2.drawContours(thresh,contours,-1,(255,0,0),1)
scale=[]
for i in range(len(contours)):
    rectangle=cv2.boundingRect(contours[i])
    area = cv2.contourArea(contours[i])
    dCLarge=cv2.rectangle(dCLarge,(rectangle[0],rectangle[1]),(rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]),(0,255,0),2)
    print(area)
    if area>500:
        if area<5200:
            sizeRect=rectangle
            scale.append(contours[i])

PixToNM=2000/(sizeRect[2])

thresh=cv2.rectangle(thresh,(sizeRect[0],sizeRect[1]),(sizeRect[0]+sizeRect[2],sizeRect[1]+sizeRect[3]),(0,0,255),1)
cv2.imshow('dCLarge',dCLarge)
#cv2.waitKey()


img = cv2.imread(image, cv2.IMREAD_COLOR)
sampleDes='SEM sample,  from AAC 155nm with NO external neutralizer'




img = img[0:800, 0:1200]
img2=img
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img',img)
c1=45
c2=155
k=0
import tkinter.messagebox
manual = tk.messagebox.askyesno(title='confirmation',
                    message='You can manually tune the threshold for the particle detection. use o and p keys to control the threshol. Use the k and l keys to control the blur intensity. \r To control the thresh for the black region use n and m.'
                                '\r Press enter when you are satisfied with the selection. \r Use <> and [] to adjust the edge detection')


k=0
tvar=80
tvar2=30
bvar=7
c1=45
c2=155
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
    elif(k==110):
        if(tvar2>0):
            tvar2-=2
    elif(k==109):
        tvar2+=2
    if(k==46):
        c1+=10
    elif(k==44):
        c1-=10
    elif(k==93):
        c2+=10
    elif(k==91):
        if(c2>0):
            c2-=10
    blur=cv2.medianBlur(gray,(bvar))    
    ret, thresh=cv2.threshold(blur,tvar,255,cv2.THRESH_BINARY_INV)
    ret, thresh2=cv2.threshold(blur,tvar2,255,cv2.THRESH_BINARY)
    thresh=cv2.addWeighted(thresh,0.5,thresh2,0.5,0)
    ret, thresh=cv2.threshold(thresh,200,255,cv2.THRESH_BINARY)
    canny=cv2.Canny(blur, c1,c2)
    ret, canny=cv2.threshold(canny,tvar,255,cv2.THRESH_BINARY_INV)
    
    added=cv2.addWeighted(thresh,0.5,canny,0.5,0)
    ret, thresh=cv2.threshold(added,200,255,cv2.THRESH_BINARY)
    cv2.imshow('Threshold', thresh)
    k=cv2.waitKey(0)


    contours,hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largeCont=[]
    rectangle=[]
    Solidity=[]
    for i in range(len(contours)):              
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        

        if area > 200:
            if area < 9000:
    #            if(hie[0][i][3]>=0):
                largeCont.append(cnt)
                rectangle.append(cv2.boundingRect(cnt))
                solidity = float(area)/hull_area
                Solidity.append(solidity)
    img2=img.copy()            
    dC=cv2.drawContours(img2,largeCont,-1,(0,255,0),1)
    cv2.imshow('dc', dC)
#    img=cv2.drawContours(fullImage,largeCont,-1,(255,0,0),1)
dCRect=cv2.rectangle(dCLarge,(rectangle[0][0],rectangle[0][1]),(rectangle[0][0]+rectangle[0][2],rectangle[0][1]+rectangle[0][3]),(0,0,255),1)
AR=[]
ParticleCount=0




manual = tk.messagebox.askyesno(title='confirmation',
                    message='Do you want to manually check classification?\n In manual Mode : \n\tFor correct classification press Enter.\r\t  For wrong selections press X. "')


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

    dCRect=cv2.rectangle(img2,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,255,0),1)
        
    if manual:
        dCRect=cv2.putText(dCRect,'Correct?', (rectangle[i][0],rectangle[i][1]-8), 1, 1,(255,255,255),2)
        cv2.imshow('Added rectangle',dCRect)
    
        k=cv2.waitKey(0)
        print(k)
    
        if(k==13):
            ParticleCount+=1
            isLacey.append(1)
        elif(k==120):
            isLacey.append(2)

    else:
        ParticleCount+=1
        isLacey.append(1)


img2= cv2.imread(image, cv2.IMREAD_COLOR)

gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
selectedRect=[]
area=[]
selectedSolidity=[]
width=[]
height=[]
AOD=[]
count=0
for i in range(len(rectangle)):
    if(isLacey[i]<2):    
        count+=1
        dCRect=cv2.rectangle(img2,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,0,255),1)
        selectedRect.append(rectangle[i])
        area.append(cv2.contourArea(largeCont[i]))
        equDiameter=2*((cv2.contourArea(largeCont[i])*PixToNM*PixToNM)/np.pi)**(1/2)
        dCRect=cv2.putText(dCRect,'  w='+str(int(rectangle[i][2]*PixToNM))+'nm', (rectangle[i][0],rectangle[i][1]-8), 1, 1,(255,255,255),1)
        dCRect=cv2.putText(dCRect,'Particle #='+str(count)+ ' h='+str(int(rectangle[i][3]*PixToNM))+'nm', (rectangle[i][0],rectangle[i][1]-20), 1, 1,(255,255,255),1)
        AOD.append([count,rectangle[i][2]*PixToNM,rectangle[i][3]*PixToNM,rectangle[i][0]*PixToNM,rectangle[i][1]*PixToNM])
        selectedSolidity.append(Solidity[i])
        width.append(int(rectangle[i][2]*PixToNM))
        height.append(int(rectangle[i][3]*PixToNM))


df = pd.DataFrame(np.array(AOD),
                   columns=['Sn', 'w', 'h','x','y'])
df.to_csv(image[:-4]+'Analysis.csv')
particleSize=[]
for i in selectedRect:
    if(i[3]>i[2]):
        particleSize.append(i[3])
    else:
        particleSize.append(i[2])

print('Total number of particles:'+str(ParticleCount))

cv2.putText(img2,sampleDes, (10,680), 1, 1,(255,255,255),2)
cv2.putText(img2,'Total number of particles:'+str(ParticleCount), (10,700), 1, 1,(255,255,255),2)


ret, thresh2=cv2.threshold(gray2,135,255,cv2.THRESH_BINARY_INV)

img2=cv2.drawContours(img2,largeCont,-1,(255,0,0),1)
fig, ax = plt.subplots(1,1,figsize=(10,10))
area=np.asarray(area)
equDiameter=2*((area*PixToNM*PixToNM)/np.pi)**(1/2)
hist=ax.hist(width[:], bins='auto')
ax.grid( which='minor', axis='both')
ax.grid( which='major', axis='both')
ax.set_xlabel("Width (nm)")
ax.set_ylabel("Count")
#ax.set_xscale('log')
fig.suptitle('Width(nm) histogram', fontsize=16)

fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
solidityHist=ax2.hist(height[:], bins='auto')
ax2.grid( which='minor', axis='both')
ax2.grid( which='major', axis='both')
ax2.set_xlabel("Height (nm)")
ax2.set_ylabel("Count")
fig2.suptitle('Height(nm) histogram', fontsize=16)
#cv2.imshow('dCLarge', dCRect)
#cv2.imshow('gray2', gray2)
cv2.imshow('img2', img2)
#cv2.waitKey(0)
plt.show()
cv2.destroyAllWindows()
