import matplotlib.pyplot as plt
import numpy as np
# STL模型Recall、Precision
no_epoch100_recall = [0.6, 0.0, 0.71, 0.13, 0.59, 0.0, 0.58, 0.16, 0.7, 0.0]
no_epoch100_precision = [0.46, 0, 0.36, 0.24, 0.27, 0, 0.3, 0.3, 0.47, 0]
no_epoch200_recall = [0.62, 0.0, 0.75, 0.29, 0.56, 0.0, 0.56, 0.34, 0.69, 0.0]
no_epoch200_precision = [0.5, 0, 0.43, 0.25, 0.33, 0, 0.34, 0.29, 0.51, 0]
no_epoch300_recall = [0.67, 0.0, 0.75, 0.28, 0.58, 0.03, 0.56, 0.41, 0.71, 0.0]
no_epoch300_precision = [0.53, 0, 0.48, 0.27, 0.32, 0.24, 0.37, 0.3, 0.51, 0]
no_epoch400_recall = [0.7, 0.0, 0.73, 0.34, 0.56, 0.1, 0.55, 0.42, 0.67, 0.28]
no_epoch400_precision = [0.56, 0, 0.54, 0.29, 0.36, 0.28, 0.43, 0.35, 0.58, 0.43]

conv1_epoch100_recall = [0.53, 0.0, 0.52, 0.04, 0.54, 0.22, 0.0, 0.01, 0.65, 0.18]
conv1_epoch100_precision = [0.39, 0, 0.25, 0.15, 0.21, 0.19, 0, 0.12, 0.36, 0.28]
conv1_epoch200_recall = [0.54, 0.0, 0.54, 0.2, 0.56, 0.05, 0.39, 0.28, 0.58, 0.26]
conv1_epoch200_precision = [0.48, 0, 0.37, 0.22, 0.26, 0.23, 0.31, 0.29, 0.47, 0.37]
conv1_epoch300_recall = [0.51, 0.0, 0.59, 0.24, 0.54, 0.13, 0.46, 0.32, 0.59, 0.35]
conv1_epoch300_precision = [0.58, 0, 0.43, 0.24, 0.3, 0.23, 0.33, 0.3, 0.51, 0.44]
conv1_epoch400_recall = [0.56, 0.0, 0.65, 0.26, 0.52, 0.14, 0.46, 0.35, 0.6, 0.4]
conv1_epoch400_precision = [0.55, 0, 0.46, 0.26, 0.32, 0.24, 0.38, 0.3, 0.53, 0.47]

pool1_epoch100_recall = [0.56, 0.0, 0.6, 0.06, 0.49, 0.0, 0.5, 0.17, 0.6, 0.0]
pool1_epoch100_precision = [0.38, 0, 0.31, 0.25, 0.22, 0, 0.26, 0.25, 0.39, 0]
pool1_epoch200_recall = [0.55, 0.0, 0.57, 0.22, 0.47, 0.01, 0.45, 0.34, 0.59, 0.24]
pool1_epoch200_precision = [0.47, 0, 0.4, 0.22, 0.27, 0.17, 0.31, 0.28, 0.47, 0.38]
pool1_epoch300_recall = [0.56, 0.0, 0.63, 0.25, 0.44, 0.06, 0.46, 0.33, 0.6, 0.32]
pool1_epoch300_precision = [0.48, 0, 0.4, 0.24, 0.31, 0.24, 0.33, 0.29, 0.49, 0.41]
pool1_epoch400_recall = [0.56, 0.0, 0.66, 0.29, 0.46, 0.1, 0.44, 0.38, 0.62, 0.35]
pool1_epoch400_precision = [0.52, 0, 0.46, 0.24, 0.32, 0.24, 0.37, 0.29, 0.52, 0.47]

conv5_epoch100_recall = [0.69, 0.0, 0.48, 0.0, 0.55, 0.0, 0.48, 0.28, 0.0, 0.49]
conv5_epoch100_precision = [0.36, 0, 0.31, 0.14, 0.23, 0, 0.3, 0.29, 0, 0.32]
conv5_epoch200_recall = [0.7, 0.0, 0.51, 0.23, 0.45, 0.0, 0.51, 0.38, 0.0, 0.53]
conv5_epoch200_precision = [0.41, 0, 0.39, 0.22, 0.28, 0, 0.34, 0.27, 0, 0.37]
conv5_epoch300_recall = [0.72, 0.0, 0.59, 0.3, 0.46, 0.0, 0.51, 0.41, 0.0, 0.51]
conv5_epoch300_precision = [0.42, 0, 0.45, 0.23, 0.31, 0, 0.35, 0.29, 0, 0.38]
conv5_epoch400_recall = [0.75, 0.02, 0.57, 0.25, 0.58, 0.03, 0.53, 0.4, 0.0, 0.54]
conv5_epoch400_precision = [0.46, 0.3, 0.52, 0.24, 0.29, 0.2, 0.38, 0.3, 0, 0.41]

pool4_epoch100_recall = [0.73, 0.0, 0.5, 0.04, 0.59, 0.0, 0.52, 0.15, 0.0, 0.52]
pool4_epoch100_precision = [0.37, 0, 0.35, 0.27, 0.23, 0, 0.3, 0.29, 0, 0.33]
pool4_epoch200_recall = [0.73, 0.0, 0.59, 0.25, 0.51, 0.0, 0.48, 0.34, 0.0, 0.5]
pool4_epoch200_precision = [0.42, 0, 0.44, 0.22, 0.29, 0.0, 0.34, 0.27, 0, 0.36]
pool4_epoch300_recall = [0.75, 0.0, 0.62, 0.29, 0.49, 0.07, 0.51, 0.35, 0.0, 0.52]
pool4_epoch300_precision = [0.43, 0, 0.48, 0.26, 0.34, 0.2, 0.37, 0.28, 0, 0.38]
pool4_epoch400_recall = [0.79, 0.0, 0.68, 0.25, 0.58, 0.14, 0.51, 0.31, 0.0, 0.5]
pool4_epoch400_precision = [0.45, 0, 0.49, 0.28, 0.32, 0.21, 0.4, 0.31, 0, 0.41]
#AVG Recall & AVG Precision
no_avg_recall = [0.35, 0.38, 0.4, 0.43]
no_avg_precision = [0.24, 0.265, 0.3, 0.38]
conv1_avg_recall = [0.27, 0.34, 0.37, 0.39]
conv1_avg_precision = [0.2, 0.3, 0.34, 0.35]
pool1_avg_recall = [0.3, 0.35, 0.37, 0.39]
pool1_avg_precision = [0.21, 0.3, 0.32, 0.34]
conv5_avg_recall = [0.3, 0.33, 0.35, 0.37]
conv5_avg_precision = [0.2, 0.23, 0.24, 0.31]
pool4_avg_recall = [0.3, 0.34, 0.36, 0.38]
pool4_avg_precision = [0.21, 0.23, 0.27, 0.29]
#Accuracy
no_acc = [0.348, 0.38, 0.398, 0.435]
conv1_acc = [0.27, 0.343, 0.373, 0.39]
pool1_acc = [0.3, 0.344, 0.366, 0.388]
conv5_acc = [0.296, 0.343, 0.35, 0.363]
pool4_acc = [0.306, 0.347, 0.364, 0.376]

#3-set
conv1_all_acc = [0.202, 0.293, 0.305, 0.343]
conv1_no = [0.145, 0.087, 0.092, 0.092]
conv1_noise = [0.068, 0.05, 0.068, 0.047]

pool1_all_acc = [0.275, 0.298, 0.307, 0.343]
pool1_no = [0.073, 0.082, 0.091, 0.090]
pool1_noise = [0.028, 0.047, 0.060, 0.043]

conv5_all_acc = [0.209, 0.262, 0.264, 0.298]
conv5_no = [0.139, 0.118, 0.133, 0.137]
conv5_noise = [0.088, 0.082, 0.086, 0.065]

pool4_all_acc = [0.219, 0.264, 0.275, 0.310]
pool4_no = [0.128, 0.115, 0.123, 0.125]
pool4_noise = [0.087, 0.083, 0.089, 0.066]

#Cifar10 Accuracy
no = [71.72, 81.31, 90.87, 90.72, 91.33]
grad = [72.53, 81.68, 91.37, 82.7, 87.19]
layer1 = [41.53, 72.56, 85.8, 86.85, 87.41]
layer3 = [57.05, 80.88, 88.89, 88.55, 89.71]
layer7 = [58.11, 77.89, 83.87, 91.46, 91.48]
layer14 = [73.82, 82.98, 91.14, 91.2, 91.7]

#cifar10
class_names = ['airplane','Automobile', 'bird' , 'cat' ,'deer' ,'dog', 'frog', 'horse' ,'ship' ,'truck']
#stl
# class_names = ['airplane' ,'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

x = ['20', '40', '60', '80', '100']

Loc = [1,3,7,14]
for loc in Loc:
    path = f"../model/cifar10/layer{loc}/"
    count = np.load(path + "count.npz")
    percentage = np.load(path + "percentage.npz")
    all_acc = []
    no = []
    noise = []
    for index in x:
        i = 'epoch' + index
        p = percentage[i]
        all_acc.append(p[1])
        no.append(p[3])
        noise.append(p[4])
    print(all_acc)
    print(no)
    print(noise)
    plt.figure(figsize=(8, 6))
    plt.plot(x, all_acc, marker='^', label="both")
    plt.plot(x, no, marker='o', label='noiseless')
    plt.plot(x, noise, marker='s', label='1')
    title = f"layer{loc}"
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("Percentage")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()  # Adding the legend
    plt.savefig("cifar10/3-Set/" + title + ".png")
    plt.close()


# # Recall、Precision
# x = ['100','200', '300', '400']
# # Data
# for i in range(10):
#     no_recall = [no_epoch100_recall[i], no_epoch200_recall[i], no_epoch300_recall[i],no_epoch400_recall[i]]
#     no_precision = [no_epoch100_precision[i], no_epoch200_precision[i], no_epoch300_precision[i],no_epoch400_precision[i]]
#     conv1_recall = [conv1_epoch100_recall[i], conv1_epoch200_recall[i], conv1_epoch300_recall[i],conv1_epoch400_recall[i]]
#     conv1_precision = [conv1_epoch100_precision[i], conv1_epoch200_precision[i], conv1_epoch300_precision[i],conv1_epoch400_precision[i]]
#     pool1_recall = [pool1_epoch100_recall[i], pool1_epoch200_recall[i], pool1_epoch300_recall[i],pool1_epoch400_recall[i]]
#     pool1_precision = [pool1_epoch100_precision[i], pool1_epoch200_precision[i], pool1_epoch300_precision[i],pool1_epoch400_precision[i]]
#     conv5_recall = [conv5_epoch100_recall[i], conv5_epoch200_recall[i], conv5_epoch300_recall[i],conv5_epoch400_recall[i]]
#     conv5_precision = [conv5_epoch100_precision[i], conv5_epoch200_precision[i], conv5_epoch300_precision[i],conv5_epoch400_precision[i]]
#     pool4_recall = [pool4_epoch100_recall[i], pool4_epoch200_recall[i], pool4_epoch300_recall[i],pool4_epoch400_recall[i]]
#     pool4_precision = [pool4_epoch100_precision[i], pool4_epoch200_precision[i], pool4_epoch300_precision[i],pool4_epoch400_precision[i]]
#     title = class_names[i]
#     #Recall
#     plt.figure(figsize=(8, 6))
#     plt.plot(x, no_recall, marker='o', label="noiseless")
#     plt.plot(x, conv1_recall, marker='s', label='1')
#     plt.plot(x, pool1_recall, marker='s', label='3')
#     plt.plot(x, conv5_recall, marker='<', label='7')
#     plt.plot(x, pool4_recall, marker='<', label='14')
#     plt.title(title)
#     plt.xlabel("epoch")
#     plt.ylabel("Recall")
#     plt.ylim(0, 1)
#     plt.grid(True)
#     plt.legend()  # Adding the legend
#     plt.savefig("stl/Recall/" + title +".png")
#     plt.close()


