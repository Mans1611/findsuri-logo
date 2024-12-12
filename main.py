import streamlit as st
from PIL import Image
import io
import os
import numpy as np
from deepface import DeepFace

logo_path = os.path.join(os.getcwd(), "logo.png")

def process_image(image):
    try:
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Attempt face extraction
        detected_faces = DeepFace.extract_faces(np.array(image), detector_backend='mtcnn', enforce_detection=False)
        
        if not detected_faces:
            st.warning("No faces detected in the image. Showing original image.")
            return image

        # Take the first detected face (assuming a single main face)
        facial_area = detected_faces[0]['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

        # Expand the face region slightly
        expand_w = w // 4
        expand_h = h // 4

        new_x = max(0, x - expand_w)
        new_y = max(0, y - expand_h)
        new_w = w + (2 * expand_w)
        new_h = h + (2 * expand_h)

        # Extract the expanded face area
        expanded_face = np.array(image)[new_y:new_y + new_h, new_x:new_x + new_w]
        expanded_face_image = Image.fromarray(expanded_face)

        # Overlay the logo
        logo = Image.open(logo_path).convert("RGBA")
        logo = logo.resize((expanded_face_image.width, expanded_face_image.height), Image.Resampling.LANCZOS)
        expanded_face_image.paste(logo, (0, 0), logo)

        return expanded_face_image

    except Exception as e:
        st.error(f"An error occurred during face processing: {e}")
        return image

st.title("Image Processing Interface")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Resize the image if it's too large (e.g., max width or height of 2000px)
    max_dimension = 2000
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

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