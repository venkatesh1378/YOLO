import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
import tempfile
import os


# Load the model
model = YOLO("yolov8s.pt")
names = model.model.names
# Upload a video file
video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
print(video)

# Detect objects in the video
if video is not None:
    # Load the video
    st.write("hello")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video.read())
    cap = cv2.VideoCapture(temp_file.name)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    st.write("hello 2")
    # Get the video's width and height
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    st.write("hello 3")

    center_point = (-10, h)

    # Process each frame of the video
    while True:
        ret, im0 = cap.read()
        if not ret:
            st.write("Video frame is empty or video processing has been successfully completed.")
            break

        results = model.predict(im0)
        
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        

        annotator = Annotator(im0, line_width=2)
    

        for box, cls in zip(boxes, clss):
            annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
            annotator.visioneye(box, center_point)
        out.write(im0)  
        st.image(im0)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    os.remove(temp_file.name)