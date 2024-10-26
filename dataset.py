import os

def list_files(directory):
    return os.listdir(directory)

# Example usage:
directory = 'D:\JamilyaYOLO\OpenLabeling\main\output\YOLO_darknet'
txt = list_files(directory)
str = "8 0.49 0.49 0.99999999 0.99999999"
for name in txt:
    file = open(os.path.join(directory, name), "w")
    file.write(str)
    file.close()
