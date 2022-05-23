import numpy as np
import tensorflow as tf

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
import tkinter as tk
from tkinter import *
import easygui
'''tf.keras.applications.imagenet_utils.decode_predictions(
    *args, **kwargs
)'''



NUM_CLASSES = 3
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# Using model without transfer learning
outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )
model.load_weights('model.h5')
model.summary()

def run(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    my_image=imread(path)
    imshow(my_image)
    print('Input image shape:', x.shape)
    preds=model.predict(x)
    print(preds)
    menin=preds.item(0)
    glioma=preds.item(1)
    pitutary=preds.item(2)
    r1="Result: MENINGIOMA"
    r2="Result: GLIOMA"
    r3="Result: PITUITARY "
    if(menin>glioma and menin>pitutary):
        return r1
    if(glioma>menin and glioma>pitutary):
        return r2
    if(pitutary>menin and pitutary>glioma):
        return r3
    
    #return preds.item(1)




#ntmg = image.load_img(img_path, target_size=(224, 224))
#x = img.img_to_array(img)



def upload():
    path=easygui.fileopenbox()
    t=run(path)
    frame= Frame(top, width=100, height=100)
    frame.pack()
    frame.place(x=175, y=300)
    print(t)
    lab= Label(frame,text=t)
    lab.configure(background='#364156',foreground='white', font=('calibri',15,'bold'))
    lab.pack(side=BOTTOM,pady=20)    



top=tk.Tk()
top.geometry('600x500')
top.title('Prediction of your tumor type')
backg=tk.PhotoImage(file='bg.png')
label1= tk.Label(top,image=backg)
upload=Button(top,text="Select an Image",command=upload,padx=10,pady=10)
upload.configure(background='#364156', foreground='white',font=('calibri',15,'bold'))
upload.pack(side=TOP,pady=50)
top.mainloop()





