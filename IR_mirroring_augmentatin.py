import cv2
import os
from skimage.util import random_noise
import random
import numpy as np

def apply_ir_effect(image):
    """Имитирует эффект ИК-снимка (преобразование в оттенки серого с добавлением шума)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = random_noise(gray, mode='gaussian', var=0.01**2)
    return (255 * noise).astype(np.uint8)

def invert_image(image):
    """Инвертирует цвета изображения (эффект негатива)."""
    return 255 - image

def mirror_image(image):
    """Зеркально отображает изображение по горизонтали."""
    return cv2.flip(image, 1)


def augment_image(image_path, output_dir):
    """Увеличивает одно изображение и обрабатывает соответствующий текстовый файл (если он существует)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            return

        base, ext = os.path.splitext(os.path.basename(image_path)) #Извлекаем имя файла и расширение


        ir_img = apply_ir_effect(img)
        cv2.imwrite(os.path.join(output_dir, f"{base}-ir{ext}"), ir_img)

        inverted_img = invert_image(ir_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}-inv{ext}"), inverted_img)

        mirrored_img = mirror_image(ir_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}-aughor{ext}"), mirrored_img)

        mirrored_inverted_img = mirror_image(inverted_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}-aughor-inv{ext}"), mirrored_inverted_img)


        # Обработка текстового файла (если он есть).  Просто копируем файл для каждого варианта.
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(txt_path):
            copy_txt_file(txt_path, output_dir, base) # Используем функцию для копирования

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")

def copy_txt_file(txt_path, output_dir, base_name):
    """Копирует текстовый файл для каждого варианта увеличенного изображения."""
    try:
        with open(txt_path, 'r') as f:
            content = f.read()

        for suffix in ["-ir", "-inv", "-aughor", "-aughor-inv"]:
            with open(os.path.join(output_dir, f"{base_name}{suffix}.txt"), 'w') as f:
                f.write(content)

    except Exception as e:
        print(f"Ошибка при копировании текстового файла {txt_path}: {e}")


def process_directory(input_dir, output_dir):
    """Обрабатывает все изображения в указанном каталоге."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            augment_image(os.path.join(input_dir, filename), output_dir)

if __name__ == "__main__":
    input_directory = r""  # Замените на ваш путь к папке с изображениями
    output_directory = "augmented_data"
    process_directory(input_directory, output_directory)
    print("Аугментация завершена!")

