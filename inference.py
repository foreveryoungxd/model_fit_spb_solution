from ultralytics import YOLO
import cv2
import os
import gc
import time
import numpy as np
import json

class GetRecogniseFramesFromVideo:
    def __init__(self):
        self.model_detect = YOLO('models/finetuned_yolo11s.pt')

    def recognize_video(self, video_name, skip_frames : int):
        vidcap = cv2.VideoCapture(video_name)
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        prev_time = 0
        frame_count = 0
        while True:
            ret, frame = vidcap.read()
            frame = cv2.resize(frame, (576, 576))
            if ret:
                frame_count += 1
                if frame_count % skip_frames == 0:
                    result_detect = self.model_detect(frame)

                    tensor_bb = result_detect[0].boxes.xyxy
                    tensor_conf = result_detect[0].boxes.conf
                    tensor_cls = result_detect[0].boxes.cls

                    for iterator_index_bb in range(len(tensor_bb)):

                        x_left_top_coord = int(tensor_bb[iterator_index_bb][0])
                        y_left_top_coord = int(tensor_bb[iterator_index_bb][1])
                        x_right_down_coord = int(tensor_bb[iterator_index_bb][2])
                        y_right_down_coord = int(tensor_bb[iterator_index_bb][3])
                        cv2.rectangle(frame, (x_left_top_coord, y_left_top_coord),
                                      (x_right_down_coord, y_right_down_coord),
                                      (255, 0, 0), 1)

                        # Получите класс и вероятность
                        cls = int(tensor_cls[iterator_index_bb])
                        conf = tensor_conf[iterator_index_bb].item()
                        class_name = self.model_detect.names[cls]  # Получение  имени  класса

                        #  Отобразите  класс  и  вероятность
                        label = f'{class_name} {conf:.2f}'
                        #  Позиционирование  метки  над  bounding  box
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                        text_x = x_left_top_coord
                        text_y = y_left_top_coord - text_size[1] - 5
                        if text_y < 0:
                            text_y = y_left_top_coord + 5
                        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                                      (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
                        cv2.putText(frame, label, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                gc.collect()

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Real-Time Object Detection', frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break


def main():
    class_recog = GetRecogniseFramesFromVideo()
    class_recog.recognize_video(r"C:\Users\Evgenii\Desktop\Датасет видео\27042021_вертолетниип 12_30.mp4",
                                skip_frames=5)


if __name__ == "__main__":
    main()