
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO

def face_detect(image, var_scaleFactor, var_minNeighbors, var_minSize):
  col1.image(image, caption="Selected Image", use_column_width=True)

  # perform face detection
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=var_scaleFactor, minNeighbors=var_minNeighbors, minSize=var_minSize)
  # print bounding box for each detected face
  for (x, y, w, h) in faces:  
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 3)

  col2.image(image, caption="Face detection", use_column_width=True)    


st.title("Face Detection using OpenCV")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Add a selectbox to the sidebar:
# Add a slider to the sidebar:

st.sidebar.title('Face Detection')
url_cover = 'https://im.indiatimes.in/content/2020/Oct/1_5f8beb58300e6.jpg'
response = requests.get(url_cover)
cover_img = Image.open(BytesIO(response.content))
st.sidebar.image(cover_img)

var_imgs = st.sidebar.selectbox(
    'Select sample images',
    ('Rose Blackpink', 'Jisoo Blackpink', 'People')
)

if(var_imgs == 'Rose Blackpink'):
  url_img = 'https://i.pinimg.com/474x/b2/08/f8/b208f8e64522caf81bf17c5ecb6ec526.jpg'
elif(var_imgs == 'Jisoo Blackpink'):
  url_img = 'https://media.philstar.com/photos/2021/04/02/jisoo_2021-04-02_17-28-20.jpg'
elif(var_imgs == 'People'):
  url_img = 'https://peoplewhogive.org/wp-content/uploads/2019/11/header100people.jpg'

#response = requests.get(url_img)
#image_from_url = np.array(Image.open(BytesIO(response.content)))


st.sidebar.title('Parameter Tuning')
var_scaleFactor = st.sidebar.slider(
    'scaleFactor',
    1.1, 3.0, 1.1
)

var_minNeighbors = st.sidebar.slider(
    'minNeighbors', 
    1, 4, 3
)

var_minSize = st.sidebar.slider(
    'minSize',
    10, 40, (20,20)
)

img_file_buffer = st.file_uploader("Upload image or select sample images from box", type=["png", "jpg", "jpeg"])

col1, col2 = st.beta_columns(2)
if img_file_buffer is not None:
  st.write('Click the cross sign before select sample images from box') 
  image = np.array(Image.open(img_file_buffer))
  face_detect(image, var_scaleFactor, var_minNeighbors, var_minSize)

else:
  st.write("Click Browse files to choose image from your computer")
  response = requests.get(url_img)
  image_from_url = np.array(Image.open(BytesIO(response.content)))
  face_detect(image_from_url, var_scaleFactor, var_minNeighbors, var_minSize)