import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np

model = YOLO('models/finetuned_yolo11s.pt')  # Load your YOLO model

st.set_page_config(page_title='Дрон-детектор', page_icon=':collision:')
st.title("Обнаружение дронов :collision:")
st.write("""Приложение для определения БПЛА самолетного и вертолетного типа в видеоряде и на фотографиях.""")

def annotate_image(image, results):
    """Adds bounding boxes and labels to the image."""
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls_id = int(box.cls[0])
        label = f"{model.names[cls_id]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return image

# File uploader for both video and image
uploaded_file = st.file_uploader("Выберите файл (видео или фото)", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.type.startswith('video/'):  # Video processing
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1])
        tfile.write(uploaded_file.read())
        video_cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = annotate_image(frame, results)
            out.write(annotated_frame)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        video_cap.release()
        out.release()
        # Optionally, display the output video here (requires more Streamlit magic)

    elif uploaded_file.type.startswith('image/'):  # Image processing
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model(image)
        annotated_image = annotate_image(image, results)
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    else:
        st.error("Неподдерживаемый тип файла.  Загрузите видео или изображение.")