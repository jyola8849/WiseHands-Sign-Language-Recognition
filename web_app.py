from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from PIL import Image
from googletrans import Translator

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title='ASL Recognition')
st.title('Odaibo voice Sign Language Recognition')
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


#@st.cache(allow_output_mutation=True)
def get_best_model():
    best_model = keras.models.load_model('models/experiment-dropout-0')
    return best_model


#@st.cache
def get_label_binarizer():
    train_df = pd.read_csv('data/alphabet/sign_mnist_train.csv')
    y = train_df['label']
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    return label_binarizer


def preprocess_image(image, image_file, best_model, label_binarizer):
    # image: numpy array

    # To display the uploaded image
    # image_width = image.shape[0]
    # st.image(image_file, caption='Uploaded Image', width=max(image_width, 100))

    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    image = image/255
    image = tf.image.resize(image, [28, 28], preserve_aspect_ratio=True)

    preprocessed_image = np.ones((1, 28, 28, 1))
    preprocessed_image[0, :image.shape[0], :image.shape[1], :] = image

    prediction = best_model.predict(preprocessed_image)

    index_to_letter_map = {i: chr(ord('a') + i) for i in range(26)}
    letter = index_to_letter_map[label_binarizer.inverse_transform(prediction)[
                                                                   0]]

    return letter


best_model = get_best_model()
label_binarizer = get_label_binarizer()

st.markdown(
    'Use 28x28 images (size of the training images) to obtain the accurate results')

st.subheader('Convert Image to English letter')
image_file = st.file_uploader('Choose the ASL Image', ['jpg', 'png'])
translator = Translator()
if image_file is not None:
    image = Image.open(image_file).convert('L')
    image = np.array(image, dtype='float32')
    letter = preprocess_image(image, image_file, best_model, label_binarizer)
    translation_yoruba = translator.translate(f'{letter}', dest='yo')
    yoruba_text = translation_yoruba.text
    translation_hausa = translator.translate(f'{letter}', dest='ha')
    translation_igbo = translator.translate(f'{letter}', dest='ig')
    st.write(f'{letter}')
    if image_file:
        col1, col2 = st.beta_columns(2)
            # Apply Fxn Here
        with col1:
            st.success("letter")
            st.write(letter)
            st.success("yoruba translation")
            st.write("{}".format(yoruba_text))

        with col2:
            st.success("igbo translation")
            st.write("{}".format(translation_igbo))
            st.success("hausa translation")
            st.write("{}".format(translation_hausa))


st.subheader('Convert images to English sentence')
sentence_image_files=st.file_uploader(
    'Select the ASL Images', ['jpg', 'png'], accept_multiple_files=True)

if len(sentence_image_files) > 0:
    sentence=''
    for image_file in sentence_image_files:
        image=Image.open(image_file).convert('L')
        image=np.array(image, dtype='float32')
        letter=preprocess_image(image, image_file, best_model, label_binarizer)
        sentence += letter
        translation_yoruba=translator.translate(f'{sentence}', dest='yo')
        translation_hausa=translator.translate(f'{sentence}', dest='ha')
        translation_igbo=translator.translate(f'{sentence}', dest='ig')
        if sentence_image_files:
            col1, col2=st.beta_columns(2)

            # Apply Fxn Here

            with col1:
                st.success("sentence")
                st.write(sentence)

                st.success("yoruba translation")
                st.write("{}".format(translation_yoruba))


            with col2:
                st.success("igbo translation")
                st.write("{}".format(translation_igbo))
                st.success("hausa translation")
                st.write("{}".format(translation_hausa))
