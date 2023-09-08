import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np

# Function to read an image from an uploaded file
def read_image(uploaded_file):
    return Image.open(uploaded_file)

# Function to write text to a Python file
def write_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

if uploaded_file is not None:
    # Read the uploaded image
    img = read_image(uploaded_file)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image using Otsu's thresholding
    _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the original and preprocessed images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(binarized_image, cmap='gray')
    axes[1].set_title("Preprocessed Image")
    axes[1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    # OCR to get text from the preprocessed image
    text = pytesseract.image_to_string(binarized_image)
    
    # Display the OCR text in the Streamlit app
    st.text_area("Extracted Text", text)

    # Write the OCR text to a Python file
    if st.button('Save Text to File'):
        write_text_to_file(text, 'Preprocessing.py')
        st.success('Text has been written to Preprocessing.py')
