import cv2
import numpy as np
import os
import random

image_dir_planes = 'photos/type_plane/images/'
label_path_planes = 'photos/type_plane/labels/'
output_path_planes = 'photos/type_plane/resized_images'

image_dir_heli = 'photos/type_helicopter/images/'
label_path_heli = 'photos/type_helicopter/labels/'
output_path_heli = 'photos/type_helicopter/resized_images'


def preprocess_heli_image_path(dir_path, extensions):
    for filename in os.listdir(dir_path):
        if any(filename.endswith(ext) for ext in extensions):
            old_path = os.path.join(dir_path, filename)
            if 'вертолетниип' in filename:
                filename = filename.replace('вертолетниип', 'type_heli')
            if 'ниип вертолет' in filename:
                filename = filename.replace('ниип вертолет', 'heli_type')
            if 'ниип_вертолет продолжение' in filename:
                filename = filename.replace('ниип_вертолет продолжение', 'heli_type_cont')
            if 'ниип_вертолет' in filename:
                filename = filename.replace('ниип_вертолет', 'heli_type_new')
            os.rename(old_path, os.path.join(dir_path, filename))


def replace_spaces_in_filenames(dir_path, extensions):
    for filename in os.listdir(dir_path):
        if any(filename.endswith(ext) for ext in extensions):
            old_path = os.path.join(dir_path, filename)
            new_path = os.path.join(dir_path, filename.replace("_", ""))
            os.rename(old_path, new_path)


def zoom_out_bbox_fixed_padding(image, bbox, padding, target_size=(640, 512)):
    """
    Уменьшает (zoom out) область вокруг объекта, используя bounding box,
    с фиксированным отступом и сохранением разрешения.

    Args:
        image: Исходное изображение (numpy array).
        bbox: Список с координатами bounding box: [class_id, center_x, center_y, width, height] (относительные координаты).
        padding: Желаемый отступ (в пикселях) вокруг bounding box.
        target_size: Желаемый размер результирующего изображения (ширина, высота).

    Returns:
        Уменьшенное изображение (numpy array) с целевым разрешением.
    """

    img_height, img_width = image.shape[:2]
    target_width, target_height = target_size

    # Переводим относительные координаты bounding box в абсолютные
    bbox_x = int(bbox[1] * img_width)
    bbox_y = int(bbox[2] * img_height)
    bbox_w = int(bbox[3] * img_width)
    bbox_h = int(bbox[4] * img_height)

    # Вычисляем координаты области с учетом отступа
    x_min = max(0, bbox_x - bbox_w // 2 - padding)
    y_min = max(0, bbox_y - bbox_h // 2 - padding)
    x_max = min(img_width, bbox_x + bbox_w // 2 + padding)
    y_max = min(img_height, bbox_y + bbox_h // 2 + padding)

    # Вырезаем фрагмент изображения
    cropped_img = image[y_min:y_max, x_min:x_max]

    # Вычисляем коэффициент масштабирования для "приближения"
    zoom_x = target_width / (x_max - x_min)
    zoom_y = target_height / (y_max - y_min)

    # Применяем аффинное преобразование для "приближения"
    M = np.array([[zoom_x, 0, 0], [0, zoom_y, 0]], dtype=np.float32)
    zoomed_img = cv2.warpAffine(cropped_img, M, (target_width, target_height), flags=cv2.INTER_AREA)

    return zoomed_img

def process_images(images_dir, annotations_dir, output_dir, step):
    """
    Обрабатывает каждое N-ное изображение из предоставленного датасета для разрежения данных.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_filenames)

    # Счетчик для отслеживания обработанных изображений

    counter = 0
    for i, filename in enumerate(image_filenames):
        if counter % step == 0:
            filename = filename.encode("ascii", "ignore").decode("ascii")
            # Извлекаем имя файла без расширения
            base_filename = filename[:-4]
            image_path = os.path.join(images_dir, filename)
            annotation_path = os.path.join(annotations_dir, f'{base_filename}.txt')
            if os.path.exists(annotation_path) and os.path.exists(image_path):
                image = cv2.imread(image_path)
                with open(annotation_path, "r") as f:
                    for line in f:
                        bbox = [float(x) for x in line.strip().split(" ")]
                        print(annotation_path, image_path, images_dir, filename)
                        try:
                            zoomed_out_image = zoom_out_bbox_fixed_padding(
                                image.copy(), bbox, padding=200, target_size=(640, 512)
                            )
                        except Exception as ex:
                            print(ex)

                        # Сохраняем обработанное изображение
                        output_filename = f'{os.path.splitext(filename)[0]}_zoomed.jpg'
                        output_path = os.path.join(output_dir, output_filename)
                        cv2.imwrite(output_path, zoomed_out_image)
            else:
                print(f'Файл разметки не найден для изображения: {filename}')

        counter += 1


def main():
    # replace_spaces_in_filenames(image_dir, ['.jpg', '.jpeg', '.png'])
    # replace_spaces_in_filenames(label_path, ['.txt'])

    preprocess_heli_image_path(image_dir_heli, ['.jpg', '.jpeg', '.png'])
    preprocess_heli_image_path(label_path_heli, ['.txt'])

    process_images(image_dir_planes, label_path_planes, output_path_planes, step=64)
    process_images(image_dir_heli, label_path_heli, output_path_heli, step=8)

if __name__ == '__main__':
    main()
