#加载类别
import numpy as np

with open('data/stl10_binary/class_names.txt', 'r') as file:
    class_names = [line.strip() for line in file.readlines()]
class_names = np.array(class_names)
print(class_names[0])