import streamlit as st
import numpy as np
import pandas as pd
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras import backend as K
import os
import time
import io
from PIL import Image, ImageOps
import plotly.express as px


st.title("SKPRED:Skin cancer classification using our Skin cancer model")
st.header("Skin cancer prediction: upload only jpg file")
st.text("Upload your image to check")

def teachable_machine_classification(img, weights_file):
    #load the model
    model=keras.models.load_model(weights_file)

    #create array of the right shape to feed into the keras model
    data=np.ndarray(shape=(1,256,256,3), dtype=np.float32)
    image=img
    #image sizing
    size=(256,256)
    image=ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into numpy array
    image_array=np.asarray(image)
    #normalize the image
    normalized_image_array=image_array/255
    normalized_image_array = normalized_image_array.reshape(-1,256,256,3)
    #normalized_image_array=(image_array.astype(np.float32)/127.0)-1

    #Load the image into the array

    #data[0] =normalized_image_array
    #print(data)

    #run the inference

    prediction=model.predict(normalized_image_array)
    #return prediction
    return np.argmax(prediction)

uploaded_file=st.file_uploader("SCPRED:choose a skin image", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Successfully uploaded", use_column_width=True)
    st.write("")
    st.write("Classifying the upload image...")
    labels=teachable_machine_classification(image,"weights.h5")
    if labels == 0:
        st.write("The probability is basal_cell_carcinoma")
    elif labels == 1:
        st.write("The probability is melanova")
    elif labels == 2:
        st.write("The probability is nevus")
    elif labels == 3:
        st.write("The probability is pigmented_benign_keratosis")
    elif labels == 4:
        st.write("The probability is squamous_cell_carcinoma")
    else:
        st.write("The probability is vascular leison")
   

