import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np

st.set_page_config(page_title="Home", layout='wide')

st.title("YOLO V5 OBJECT DETECTION APPLICATION")
st.caption("This web application demonstrates object detection")

st.write("YOLO (You Only Look Once) V5 is a state-of-the-art object detection model known for its speed and accuracy. It is used to detect and classify multiple objects within an image or a video stream in real-time. The YOLO V5 model, developed by the Ultralytics team, builds upon previous versions to provide improved performance, ease of use, and deployment flexibility.")

st.write("Key Features")
st.write("Real-time Object Detection: YOLO V5 can process images and video streams in real-time, making it suitable for applications that require quick response times.")
st.write("High Accuracy: The model is trained on the COCO dataset, which includes a wide variety of objects, ensuring high accuracy in detecting and classifying objects")
st.write("Lightweight and Efficient: YOLO V5 is designed to be lightweight and efficient, making it possible to run on devices with limited computational resources, such as smartphones and edge devices.")
st.write("Scalability: The model comes in different sizes (e.g., YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x) to balance the trade-off between speed and accuracy based on the application requirements.")

url = "https://github.com/ultralytics/yolov5.git"

if st.button('YOLO GitHub'):
    st.write(f'<meta http-equiv="refresh" content="0; url={url}">', unsafe_allow_html=True)

st.info("Made by: Shreyansh Singh")

# options = ["YOLO for Images", "YOLO real-time detection"]
# choice = st.sidebar.selectbox("Choose an option", options)

# Load YOLO model once
with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='best.onnx', data_yaml='data.yaml')

def upload_image():
    image_file = st.file_uploader(label="Upload image")
    if image_file is not None:
        size_mb = image_file.size / (1024 ** 2)
        file_details = {"filename": image_file.name,
                        "filetype": image_file.type,
                        "filesize": "{:,.2f}MB".format(size_mb)}

        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('Valid')
            return {"file": image_file, "details": file_details}
        else:
            st.error("Invalid file type")
            st.caption("Only jpeg , png files are allowed ")
            return None

def image_detection():
    st.title("Welcome to YOLO for Images")
    st.write("Upload an Image")

    obj = upload_image()
    if obj:
        prediction = False
        image_obj = Image.open(obj['file'])

        col1, col2 = st.columns(2)
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)

        with col2:
            st.subheader("Check Below for file details ")
            st.json(obj['details'])

            if st.button('Prediction from YOLO'):
                image_array = np.array(image_obj)
                pred_image = yolo.predictions(image_array)
                pred_image_obj = Image.fromarray(pred_image)
                prediction = True

        if prediction:
            st.image(pred_image_obj, caption="Predicted Image")

def real_time_detection():
    st.title("YOLO Real-Time Detection")

    import av
    from streamlit_webrtc import webrtc_streamer

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        pred_img = yolo.predictions(img)
        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

    webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
    
    
choice = ["YOLO for Images" , "YOLO real-time detection"]
    
if choice == "YOLO for Images":
    image_detection()
elif choice == "YOLO real-time detection":
    real_time_detection()
