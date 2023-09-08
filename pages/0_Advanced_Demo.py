import cv2
import easyocr
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
import pytesseract as tess
import streamlit as st
from PIL import Image

st.title("Testing OCR libraries on code snippets")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.sidebar:
        st.header("Configuration")
        select_tesseract = st.checkbox("Compute tesseract")
        select_keras = st.checkbox("Compute keras-ocr")
        select_easyocr = st.checkbox("Compute easyocr")
    
    image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(image)

    if select_tesseract:
        compute_tesseract(np_image)
    if select_keras:
        compute_keras(np_image)
    if select_easyocr:
        compute_easyocr(np_image)


def compute_tesseract(np_image):
    """
    https://www.opcito.com/blogs/extracting-text-from-images-with-tesseract-ocr-opencv-and-python
    https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
    """
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    inverted_img = cv2.bitwise_not(threshold_img)

    def _compute_ocr(label, img):
        custom_oem_psm_config = r"--oem 3 --psm 6"
        text = tess.image_to_string(img, config=custom_oem_psm_config)
        st.subheader(label)
        c1, c2 = st.columns((1, 2))
        c1.image(img)
        c2.code(text)

    _compute_ocr("BGR image", np_image)
    _compute_ocr("RGB image", rgb_image)
    _compute_ocr("Binary image", threshold_img)
    _compute_ocr("Inverted image", inverted_img)


def compute_keras(np_image):
    """
    Keras-ocr
    """
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [np_image]
    prediction_groups = pipeline.recognize(images)

    fig, ax = plt.subplots(figsize=(20, 20))
    keras_ocr.tools.drawAnnotations(
        image=images[0], predictions=prediction_groups[0], ax=ax
    )

    st.pyplot(fig)


def compute_easyocr(np_image):
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    inverted_img = cv2.bitwise_not(threshold_img)

    reader = easyocr.Reader(["en"], gpu=False)
    resp = reader.readtext(inverted_img, detail=0, paragraph=False)

    st.subheader("EasyOCR")
    c1, c2 = st.columns((1, 2))
    c1.image(inverted_img)
    c2.write(resp)
