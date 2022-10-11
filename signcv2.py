# A better method to get good accuracy is to consider the differnce of the highest and the second highest accuracy 

from gtts import gTTS
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time

global c 
global string

def preprocess(img):
    # loc = r'C:/Users/suyash/Desktop\asl_alphabet_test\asl_alphabet_test'
    # img = image.load_img(loc+'\\'+file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_dims)

loc = r'C:\Users\suyash\Downloads\Trying2.h5'
model = load_model(loc)

cap = cv2.VideoCapture(0)

num2alp = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 
            15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z', 26:'del', 27:'', 28:' '}

ptime = time.time()
c =100
string = ''

while True:
    _, img = cap.read()
    resize = cv2.resize(img, (224, 224),interpolation=cv2.INTER_AREA)
    imag = preprocess(resize)
    ans = model.predict(imag)
    li = ans.tolist()
    
    # ctime = time.time()
    # fps = 1/(ctime - ptime)
    # ptime = ctime

    m = li[0].index(max(max(li)))

    if c ==100:
        c = m
    ntime = time.time()
    if ntime-ptime>30:
        break

    if (max(max(ans))>0.7 and m!=c):
        print(num2alp[m], max(max(li)))
        cv2.putText(img, str(m), (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5 ,(255,0,0), 1)
        c=m
        string = string + num2alp[m]
        ptime = ntime

    cv2.imshow("Video", img)
    key = cv2.waitKey(1)

    if key == 27:
        break

print(string)
cap.release()
cv2.destroyAllWindows()

try:
    myobj = gTTS(text=string, lang='en', slow=False)
    myobj.save("welcome.mp3")
    os.system("welcome.mp3")
except:
    print("PLease say something")