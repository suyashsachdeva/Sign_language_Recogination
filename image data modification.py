import cv2
import os
import numpy as np

FOLDER = r"C:\Users\suyash\Desktop\Hand"
NEW = r'C:\Users\suyash\Desktop\Hands Processed'
files = os.listdir(FOLDER)
# os.chdir(NEW)

for c, file in enumerate(files):
    img = cv2.imread(FOLDER+"/"+file)
    img = cv2.resize(img, (500,500), interpolation=cv2.INTER_AREA)
    # cv2.imshow("image", img)
    cv2.imwrite(NEW+"/"+str(c)+".jpg", img)
    
    