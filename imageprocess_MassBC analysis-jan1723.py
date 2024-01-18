#AUTHOR Shreya Joshi
import matplotlib.pyplot as plt
from tkinter import *
import cv2
import numpy as np
import tkinter as tk
# Read image.
import pandas as pd
import os
import json

results_directory = 'Results'

# Check if the directory exists
if not os.path.exists(results_directory):
    # Create the directory
    os.makedirs(results_directory)
    print(f"Directory '{results_directory}' created.")
else:
    print(f"Directory '{results_directory}' alreadydCLarge exists.")


directory=os.getcwd()
for filename in os.listdir(directory):
    if filename.lower().endswith(('.tif', '.tiff')):
        # Construct full file path
        file_path = os.path.join(directory, filename)

        # Process the TIFF file
        print(f"Processing {file_path}")
    else:
        continue  
        
    image=filename

    img = cv2.imread(image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Unable to read the image at the specified path.")
    
    fullImage=img.copy()

    #Chamber these ranges if the scale selection is wrong

    img2=img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh=cv2.threshold(img,165,255,cv2.THRESH_BINARY_INV)
    contours,hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Finds the scale in the image
    cropped_img = img2[900:1000, 900:1400]  # Example crop coordinates

    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find the leftmost and rightmost non-zero pixels
    coordinates = np.column_stack(np.where(thresh > 0))
    leftmost = tuple(coordinates[coordinates[:, 1].argmin()][::-1])
    rightmost = tuple(coordinates[coordinates[:, 1].argmax()][::-1])

    # Calculate the distance between these points
    distance = np.linalg.norm(np.array(rightmost) - np.array(leftmost))

    # Draw a line between the leftmost and rightmost points
    cv2.line(cropped_img, leftmost, rightmost, (0, 255, 0), 2)
    
    # Display the image with the measured line
    cv2.imshow("Measured Distance", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Gives us the pixel to nm conversion based on the scale 
    PixToNM=5000/distance


    img = cv2.imread(image, cv2.IMREAD_COLOR)
    sampleDes='SEM sample,  from AAC 155nm with NO external neutralizer'



    #crop the image and ignore the lower information bar
    img = img[0:800, 0:1200]
    img2=img
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img',img)
    c1=45
    c2=155
    k=0

    #gives us the instruction 
    import tkinter.messagebox
    manual = tk.messagebox.askyesno(title='confirmation',
                        message='You can manually tune the threshold for the particle detection. use o and p keys to control the threshol. Use the k and l keys to control the blur intensity. \r To control the thresh for the black region use n and m.'
                                    '\r Press enter when you are satisfied with the selection. \r Use <> and [] to adjust the edge detection. \r Press no to skip image')

        # If the user presses 'No', skip to the next file
    if not manual:
        continue
    
    # parameters are changed using keys to detect teh particles correctly

    k=0
    tvar=80
    tvar2=30
    bvar=7
    c1=45
    c2=155
    smallest=200

    #read the variables from last run if they exist
    if os.path.exists('variables.json'):
        with open('variables.json', 'r') as file:
            data = json.load(file)
            c1 = data.get('c1', c1)
            c2 = data.get('c2', c2)
            tvar2 = data.get('tvar2', tvar2)
            tvar = data.get('tvar', tvar)
            smallest = data.get('smallest', smallest)
            bvar = data.get('bvar',bvar)
            
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
        elif(k==115):
            smallest+=5
        elif(k==97):
            smallest-=5


        if smallest <=0:
            smallest=1
            
        blur=cv2.medianBlur(gray,(bvar))    
        ret, thresh=cv2.threshold(blur,tvar,255,cv2.THRESH_BINARY_INV)
        ret, thresh2=cv2.threshold(blur,tvar2,255,cv2.THRESH_BINARY)
        thresh=cv2.addWeighted(thresh,0.5,thresh2,0.5,0)
        ret, thresh=cv2.threshold(thresh,200,255,cv2.THRESH_BINARY)
        canny=cv2.Canny(blur, c1,c2)
        ret, canny=cv2.threshold(canny,tvar,255,cv2.THRESH_BINARY_INV)
        
        added=cv2.addWeighted(thresh,0.5,canny,0.5,0)
        ret, thresh=cv2.threshold(added,200,255,cv2.THRESH_BINARY)
        
        k=cv2.waitKey(0)


        contours,hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largeCont=[]
        rectangle=[]
        Solidity=[]
        #goes through the particles for the current settings
        for i in range(len(contours)):              
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
        
            if area > smallest:
                if area < 9000:
                    largeCont.append(cnt)
                    rectangle.append(cv2.boundingRect(cnt))
                    solidity = float(area)/hull_area
                    Solidity.append(solidity)
        img2=img.copy()
        #draw around the particles detected
        dC=cv2.drawContours(img2,largeCont,-1,(0,255,0),1)
        for i in range(len(rectangle)):
            dC=cv2.rectangle(dC,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,0,255),1)
            thresh=cv2.rectangle(thresh,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,0,255),1)

        cv2.imshow('Threshold', thresh)
        cv2.imshow('dc', dC)

    import json

    # Variables to be saved
    data_to_save = {
        'c1': c1,
        'c2': c2,
        'tvar2': tvar2,
        'tvar': tvar,
        'smallest': smallest,
        'bvar': bvar
    }

    # Writing variables to a file
    with open('variables.json', 'w') as file:
        json.dump(data_to_save, file)

#   dCRect=cv2.rectangle(dCLarge,(rectangle[0][0],rectangle[0][1]),(rectangle[0][0]+rectangle[0][2],rectangle[0][1]+rectangle[0][3]),(0,0,255),1)
    AR=[]
    ParticleCount=0



    #Check if particles are correctly what we see individal or accept all particles
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
    equivalentDiameter=[]
    perimeter = []
    count=0
    for i in range(len(rectangle)-1):
        if(isLacey[i]<2):    
            count+=1
            dCRect=cv2.rectangle(img2,(rectangle[i][0],rectangle[i][1]),(rectangle[i][0]+rectangle[i][2],rectangle[i][1]+rectangle[i][3]),(0,0,255),1)
            selectedRect.append(rectangle[i])
            area.append(cv2.contourArea(largeCont[i]))
            perimeter_contour = cv2.arcLength(largeCont[i], True)*PixToNM
            perimeter.append(perimeter_contour)
            equDiameter=2*((cv2.contourArea(largeCont[i])*PixToNM*PixToNM)/np.pi)**(1/2)
            dCRect=cv2.putText(dCRect,'  w='+str(int(rectangle[i][2]*PixToNM))+'nm', (rectangle[i][0],rectangle[i][1]-8), 1, 1,(255,255,255),1)
            dCRect=cv2.putText(dCRect,'Particle #='+str(count)+ ' h='+str(int(rectangle[i][3]*PixToNM))+'nm', (rectangle[i][0],rectangle[i][1]-20), 1, 1,(255,255,255),1)
            Sphericity=np.pi*equDiameter/perimeter_contour
            AOD.append([count,rectangle[i][2]*PixToNM,rectangle[i][3]*PixToNM,rectangle[i][0]*PixToNM,rectangle[i][1]*PixToNM,Solidity[i],equDiameter,perimeter_contour,Sphericity])
            selectedSolidity.append(Solidity[i])
            equivalentDiameter.append(equDiameter)
            width.append(int(rectangle[i][2]*PixToNM))
            height.append(int(rectangle[i][3]*PixToNM))


    df = pd.DataFrame(np.array(AOD),
                       columns=['Sn', 'w', 'h','x','y','Solidity','equivalentDiameter9nm','perimeter','Sphericity'])
    df.to_csv('Results/'+image[:-4]+'Analysis.csv')
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

    cv2.imwrite('Results/'+image[:-4]+'Analyzed.png', dCRect) 
#    cv2.imshow('dCLarge', dCRect)

    #cv2.imshow('gray2', gray2)
    cv2.imshow('img2', img2)
    #cv2.waitKey(0)
    plt.show()
    cv2.destroyAllWindows()
