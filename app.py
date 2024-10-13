import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile

model = YOLO('models/finetuned_yolo11s.pt')
st.set_page_config(page_title='Дрон-детектор', page_icon=':collision:')
st.title("Обнаружение дронов :collision:")
st.write("""Бета-версия приложения для определения БПЛА самолетного и вертолетного типа в видеоряде.""")

def annotate_image(image, results):
    for box in results[0].boxes:  # Extract bounding boxes from the results
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        conf = box.conf[0]  # Confidence score of the detection
        cls_id = int(box.cls[0])  # Class ID of the detected object
        label = f"{results[0].names[cls_id]} {conf:.2f}"  # Label with class name and confidence
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate the text size and position
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)  # Background rectangle for label
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Text label in black color

    return image

# Video upload and processing
uploaded_video = st.file_uploader("Выберите видеоряд", type=["mp4", "avi", "mov"])
if uploaded_video:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Open the video file
    video_cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()  # Placeholder for displaying video frames

    # Initialize VideoWriter for saving the output video
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while video_cap.isOpened():
        ret, frame = video_cap.read()  # Read a frame from the video
        if not ret:
            break

        # Run inference on the frame
        results = model(frame)

        # Annotate the frame with detections
        annotated_frame = annotate_image(frame, results)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Convert annotated frame back to RGB for displaying in Streamlit
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    video_cap.release()  # Release the video capture object
    out.release()  # Release the VideoWriter object



