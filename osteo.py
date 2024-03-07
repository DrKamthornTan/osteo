# Import the necessary libraries
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.set_page_config(page_title='DHV AI Startup', layout='wide')

# Set the title and subtitle
st.title("DHV AI Startup Falls-Related Bone Fracture Prediction ")
st.subheader("คาดการณ์จากภาพ X-Ray เข่า")

# Define a function to get the prediction from an uploaded image
def predict_image(image_file):
    # Load the image
    image = Image.open(image_file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, method=Image.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the image and prediction
    caption=f"<p style='font-size: 24px'>Class: {class_name[2:]}\nConfidence Score: {confidence_score:.2f}</p>"
    st.image(image, caption=None)
    st.markdown(caption, unsafe_allow_html=True)

# Define the main function
def main():
    # Add a file uploader to allow the user to choose an image file
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # If the user has uploaded an image file, call the 'predict_image' function
    if uploaded_file is not None:
        predict_image(uploaded_file)

    # If the user has not uploaded an image file, display a message
    else:
        st.error("Please upload an image file")

# Call the main function
if __name__ == "__main__":
    main()