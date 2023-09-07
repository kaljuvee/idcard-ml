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

st.title('Image Upload and Process')

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

if uploaded_file is not None:
    img = read_image(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Process button
    if st.button('Process'):
        output_path = 'output_image.png'
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(img, lang='eng')
        st.text_area("Extracted Text", text)
        
        # Save and display the image using matplotlib
        #write_image(img, output_path)
        show_image(img)
        st.success(f"Image saved to: {output_path}")
