
#tiny-imagenet类别
with open('data/archive/tiny-imagenet-200/words.txt', 'r') as file:
    class_dict = {line.split()[0]: line.split()[1] for line in file.readlines()}
class_names= [value for value in class_dict.values()]
class_names = np.array(class_names)
# Cifar10类别
# class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]