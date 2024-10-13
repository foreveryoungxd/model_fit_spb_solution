import cv2
from r_channel import R_chanel
import os
# Функция для отражения изображения слева направо
def mirroring(image):
    mirrored_image = cv2.flip(image, 1)
    return mirrored_image

# Функция для применения эффекта негатива к изображению
def negative(image):
    negative_image = 255 - image
    return negative_image

def quadro_image(image_path):
    print(image_path)
    image = cv2.imread(image_path)
    cv2.imwrite(image_path[:-4]+"-aughor.jpg", mirroring(image))
    cv2.imwrite(image_path[:-4] + "-inv.jpg", negative(image))
    cv2.imwrite(image_path[:-4] + "-aughor-inv.jpg", negative(mirroring(image)))
    # quadro_txt(image_path[:-4] + ".txt")

def quadro_txt(txt_path):
    print(txt_path)
    bnd = []
    file = open(txt_path, 'r')
    while True:
        content = file.readline()
        if not content:
            break
        bnd.append(content.split(" "))
    file.close()
    if bnd != []:
        bnd_new = []
        for line in bnd:
            new_line = []
            new_line.append(line[0])
            # new_line.append(str(0))
            new_line.append(str(1 - float(line[1])))
            new_line.append(line[2])
            new_line.append(str(1 - float(line[3])))
            new_line.append(line[4])
            bnd_new.append(" ".join(new_line))
        with open (txt_path[:-4] + "-aughor.txt", 'w') as file:
            file.write("".join(bnd_new))
        file.close()
        with open (txt_path[:-4] + ".txt", 'w') as file:
            for line in bnd:
                # line[0] = str(0)
                file.writelines(" ".join(line))
        file.close()
        with open (txt_path[:-4] + "-inv.txt", 'w') as file:
            for line in bnd:
                file.writelines(" ".join(line))
        file.close()
        with open (txt_path[:-4] + "-aughor-inv.txt", 'w') as file:
            file.write("".join(bnd_new))
        file.close()


