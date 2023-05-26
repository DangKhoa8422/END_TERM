#%%
from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
import tensorflow as tf
from keras.utils.image_utils import img_to_array
from keras.utils import load_img
import numpy as np

face_cascade = cv.CascadeClassifier()
face_cascade.load('haarcascade_frontalface_default.xml')

model_gender = tf.keras.models.load_model('gioitinh.h5', compile=False)
model_age = tf.keras.models.load_model('AGE.h5', compile=False)
def show(event):
    invite.configure(text = '', font = ('Time',16), width = 10)
    file_path = address.get()
    img = Image.open(file_path)
    # Resize ảnh
    new_width, new_height = 480, 500
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    photo_img = ImageTk.PhotoImage(img)
    label_img.config(image=photo_img)
    label_img.image = photo_img
    
def predict():
    file_path = address.get()
    age = {0: '1-5',1:'6-10', 2:'11-15', 3:'16-20', 4:'21-25', 5:'26-30' }
    gender = {0: 'Nữ',1:'Nam'} 
    frame = cv.imread(file_path, cv.IMREAD_COLOR)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x+10:x+w-10]
        roi = cv.resize(roi, (40,30)) 
        cv.imshow('H', roi)
        roi = img_to_array(roi)
        #roi=roi.astype('float32')
        #roi = roi/255
        roi=roi.reshape(1,40,30,3)
        #cv.normalize(roi,roi,0,1.0,cv.NORM_MINMAX)
        roi = roi.astype('float32')
        roi =roi/255
    result_gender=np.argmax(model_gender.predict(roi),axis=1)
    result_age=np.argmax(model_age.predict(roi),axis=1)
    label_result.configure(text=(age[result_age[0]], gender[result_gender[0]]), font = ('Time',16))
rt = Tk()


rt.title('Age & Gender Prediction')
rt.geometry('800x800')



label_img = Label()
invite = Label(text = 'Please type name of image you wanna predict',font = ('Time',16), width = 40)

empty = Label(font = ('Time',16), width = 10)

address = Entry(font = ('Time',16), width = 15)
address.insert(0,'.jpg')
address.bind('<Return>', show)

label_result = Label(font = ('Time',16), width = 15)
predict_button = Button(text = 'Predict', font = ('Time',16), width = 15, command = predict)


label_img.grid(column=2,row=0)
invite.grid(column=2,row=1)
address.grid(column=2,row=2)
label_result.grid(column=2,row=3)
predict_button.grid(column=2,row=4)

empty.grid(column=1,row=1)

mainloop()
# %%
