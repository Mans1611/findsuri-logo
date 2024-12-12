import streamlit as st
from PIL import Image
import io
import os 
import base64
import numpy as np
from deepface import DeepFace
import requests
logo_path = os.path.join(os.getcwd(),"logo.png")

def process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Use DeepFace to detect faces
    detected_faces = DeepFace.extract_faces(np.array(image), detector_backend='mtcnn', enforce_detection=False)

    facial_area = detected_faces[0]['facial_area']
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    
    # Calculate expansion
    expand_w = w // 4
    expand_h = h // 4  

    # New coordinates
    new_x = max(0, x - expand_w)  
    new_y = max(0, y - expand_h)  
    new_w = w + (2 * expand_w)
    new_h = h + (2 * expand_h)

    # Crop the expanded region
    expanded_face = np.array(image)[new_y:new_y + new_h, new_x:new_x + new_w]
    
    # Convert to PIL Image for processing
    expanded_face_image = Image.fromarray(expanded_face)

    # Load the logo image
    logo = Image.open(logo_path)
    logo = logo.convert("RGBA")  # Ensure logo has an alpha channel

    # Resize the logo to be smaller
    logo_width = int(expanded_face_image.width)
    logo_height = int(expanded_face_image.height)
    logo = logo.resize((logo_width, logo_height), int(Image.Resampling.LANCZOS))

    # Paste the logo as a watermark (bottom-right corner)
    x_offset = int(expanded_face_image.width  / 2) - logo_width - 10  # 10px padding from the edge
    y_offset = int(expanded_face_image.height / 2) - logo_height - 10
    expanded_face_image.paste(logo, (0, 0), logo)

   
   
    return expanded_face_image

# Streamlit App
st.title("Image Processing Interface")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    processed_image = process_image(image)

    # Display the processed image
    st.subheader("Processed Image")
    st.image(processed_image, caption="Processed Image", use_column_width=True)

    # Provide a download link for the processed image
    buf = io.BytesIO()
    processed_image.save(buf, format="PNG")
    buf.seek(0)

    st.download_button(
        label="Download Processed Image",
        data=buf,
        file_name="processed_image.png",
        mime="image/png"
    )
else:
    st.info("Please upload an image to process.")
