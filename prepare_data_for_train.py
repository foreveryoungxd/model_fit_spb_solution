import os
import shutil
import random

image_dir_planes = 'photos/type_plane/images/'
label_path_planes = 'photos/type_plane/labels/'

image_dir_heli = 'photos/type_helicopter/images/'
label_path_heli = 'photos/type_helicopter/labels/'

yolo_train_images = 'photos/train/images/'
yolo_test_images = 'photos/test/images/'
yolo_val_images = 'photos/valid/images/'

yolo_train_labels = 'photos/train/labels/'
yolo_test_labels = 'photos/test/labels/'
yolo_val_labels = 'photos/valid/labels/'

def create_valid_set(source_dir,
                     target_dir,
                     labels_dir,
                     valid_labels_dir,
                     percentage=0.3):
    filenames = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(filenames)
    num_files_to_move = int(len(filenames) * percentage)
    for i in range(num_files_to_move):
        filename = filenames[i]
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        label_filename = filename[:-4] + ".txt"
        label_source_path = os.path.join(labels_dir, label_filename)
        label_target_path = os.path.join(valid_labels_dir, label_filename)
        shutil.move(source_path, target_path)
        shutil.move(label_source_path, label_target_path)

def split_data(images_dir,
               labels_dir,
               yolo_train_images,
               yolo_test_images,
               yolo_train_labels,
               yolo_test_labels):
    """
    Разделяет данные на train и test папки, копируя каждое N-ое изображение и его аннотацию.

    Args:
        images_dir: Путь к папке с изображениями.
        labels_dir: Путь к папке с аннотациями.
        train_dir: Путь к папке для train данных.
        test_dir: Путь к папке для test данных.
        step: Каждое N-ое изображение будет копироваться.
    """

    image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_filenames)
    counter = 0
    for i, filename in enumerate(image_filenames):
        if counter % 64 == 0:
            shutil.copy(os.path.join(images_dir, filename), os.path.join(yolo_test_images, filename))
            annotation_filename = filename[:-4] + ".txt"
            shutil.copy(os.path.join(labels_dir, annotation_filename), os.path.join(yolo_test_labels, annotation_filename))
        elif (counter % 16 == 0) or (counter % 32 == 0) or (counter % 48 == 0):
            shutil.copy(os.path.join(images_dir, filename), os.path.join(yolo_train_images, filename))
            annotation_filename = filename[:-4] + ".txt"
            shutil.copy(os.path.join(labels_dir, annotation_filename), os.path.join(yolo_train_labels, annotation_filename))
        counter += 1


def convert_into_single_label(label_path):
    for filename in os.listdir(label_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(label_path, filename)
            with open(full_path, "r") as f_in:
                line = f_in.readline()
            parts = line.strip().split()
            parts[0] = '0'
            new_line = ' '.join(parts) + '\n'
            with open(full_path, "w") as f_out:
                f_out.write(new_line)


def main():
    split_data(image_dir_planes, label_path_planes, yolo_train_images, yolo_test_images, yolo_train_labels, yolo_test_labels)
    split_data(image_dir_heli, label_path_heli, yolo_train_images, yolo_test_images, yolo_train_labels,
               yolo_test_labels)
    convert_into_single_label(yolo_train_labels)
    convert_into_single_label(yolo_test_labels)

    create_valid_set(yolo_test_images, yolo_val_images, yolo_test_labels, yolo_val_labels)

if __name__ == '__main__':
    main()
