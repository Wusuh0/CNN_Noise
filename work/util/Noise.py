import numpy as np
import torch
def gasuss_noise(image, mean=0, var=0.004):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    noise_np = np.random.normal(mean, var ** 0.5, image.shape).astype(np.float32)
    noise = torch.tensor(noise_np)
    out = image + noise
    return out

if "__name__" == "__main__":
    image = torch.tensor(np.array([1.33,1.22],dtype=np.float32))
    x = np.random.normal(0, 0.001 ** 0.5, image.shape).astype(np.float32)
    y = torch.tensor(x)
    y = y + image
    print(y)