import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract

def read_image(uploaded_file):
    """Read an image from an uploaded file and return it."""
    return Image.open(uploaded_file)

def write_image(image, path):
    """Save the image plot to a file."""
    plt.imshow(image)
    plt.axis('off')  # No axes for this plot
    plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)

def show_image(image):
    """Save the image plot to a file."""
    plt.show()

st.title('ID Card Verification - Arabic Demo')
st.header('Overview')

st.markdown("""
### Libraries Used

- **streamlit**: Used to create the web app.
- **PIL (Python Imaging Library)**: Utilized to open, manipulate, and save various formats of image files.
- **cv2 (OpenCV)**: Facilitates image processing tasks including converting the image to grayscale and binarization.
- **matplotlib**: Employed to display images within the Streamlit app.
- **pytesseract**: A Python binding for Google's Tesseract-OCR Engine, leveraged to extract text from the image.
- **numpy**: Assists in working with arrays.

### App Overview

Users are provided with an option to upload an image file (JPG or PNG).

#### If an image is uploaded:

1. The image is read and converted to a format suitable for processing with OpenCV.
2. The image undergoes grayscale conversion (optional)
3. The image is binarized utilizing Otsu's thresholding.
4. Original image displayed
5. OCR / ML is executed on the preprocessed image to extract text, utilizing Tesseract.
6. The extracted text is displayed in a text area in the app.
""")

st.subheader("Run Application")
             
# Upload the image
uploaded_file = st.file_uploader("Please choose an image...", type=['jpg', 'png'])

if uploaded_file is not None:
    img = read_image(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Process button
    if st.button('Process'):
        output_path = 'output_image.png'
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(img, lang='en')
        st.text_area("Extracted Text", text)
        
        # Save and display the image using matplotlib
        #write_image(img, output_path)
        show_image(img)
        st.success(f"Image saved to: {output_path}")
