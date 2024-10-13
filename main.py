from r_channel import R_CHB_chanel
from invariant import quadro_image
import os


folder_path = r"C:\Users\West Timor\PycharmProjects\UAV\planesmetrics"
os.chdir(folder_path)



jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

for file in jpg_files:
    file_path = os.path.join(folder_path, file)
    print(file_path)
    # R_CHB_chanel(file_path)
    quadro_image(file_path)

