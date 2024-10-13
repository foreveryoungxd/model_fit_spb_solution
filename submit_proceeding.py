from ultralytics import YOLO
import cv2
import json
import os



def detect_objects_and_save_json(image_path, model='finetuned_yolo11s.pt'):
    """
    Определяет объекты на изображении, предсказывает класс и координаты bounding box
    и сохраняет результаты в JSON-файл.

    Args:
        image_path (str): Путь к изображению.
        model_path (str): Путь к файлу модели YOLO (по умолчанию 'finetuned_yolov9s.pt').
        output_json_path (str): Путь к выходному JSON-файлу (по умолчанию 'detections.json').
    """

    # Загрузите изображение
    image = cv2.imread(image_path)
    model = YOLO(model)

    # Сделайте предсказание
    results = model(image)

    detection_results = {
        "filename": os.path.basename(image_path),
        "objects": []
    }
    try:
        for result in results:
            objects = []
            for detection in result[0].boxes.data:
                x1, y1 = (int(detection[0]), int(detection[1]))
                x2, y2 = (int(detection[2]), int(detection[3]))
                score = round(float(detection[4]), 2)
                cls = int(detection[5])
                object_name = model.names[cls]
                image_width, image_height = image.shape[1], image.shape[0]

                #  Переводим  координаты  в  относительные
                x1_rel = x1 / image_width
                y1_rel = y1 / image_height
                x2_rel = x2 / image_width
                y2_rel = y2 / image_height
                width_rel = (x2 - x1) / image_width
                height_rel = (y2 - y1) / image_height

                #  Формируем  данные  для  JSON
                object_data = {
                    "obj_class": str(cls),
                    "x": str(round(x1_rel, 2)),
                    "y": str(round(y1_rel, 2)),
                    "width": str(round(width_rel, 2)),
                    "height": str(round(height_rel, 2))}
                detection_results["objects"].append(object_data)

    except Exception as ex:
        return detection_results

    return detection_results

def main():
    # Создайте список для хранения всех результатов
    all_results = []

    #  Загрузите  модель
    #model = YOLO('finetuned_yolo11s.pt')

    #  Итерируйте  по  изображениям
    for filename in os.listdir(r"C:\Users\Evgenii\Desktop\data"):
        if filename.endswith('.jpg'):
            image_path = os.path.join(r"C:\Users\Evgenii\Desktop\data", filename)
            #  Обработайте  изображение
            results = detect_objects_and_save_json(image_path, model='models/finetuned_yolo11s.pt')
            print(results)
            #  Добавьте  результаты  в  список
            all_results.extend(results)

    #  Сохраните  все  результаты  в  JSON-файл
    with open('submit.json', 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()