import numpy as np
import torch
def gasuss_noise(image, mean=0, var=0.004, device = "cuda:0"):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    noise_np = np.random.normal(mean, var ** 0.5, image.shape).astype(np.float32)
    noise = torch.tensor(noise_np).to(device)
    # print(noise.shape)
    # print(noise)
    out = image + noise
    return out

if __name__ == "__main__":
    image = torch.tensor(np.array([1.33,1.22],dtype=np.float32)).to("cuda:0")
    y = gasuss_noise(image)
    print(y)