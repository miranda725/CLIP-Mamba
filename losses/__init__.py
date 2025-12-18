import torch.nn.functional as F
import torch
from math import exp
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义MRI图像的损失函数
def mri_loss (fused_result,input_mri):
    a=fused_result-input_mri
    b=torch.square(fused_result-input_mri)
    c=torch.mean(torch.square(fused_result-input_mri))
    mri_loss=c
    return mri_loss
# 定义SPECT图像的损失函数
def spect_loss (fused_result , input_spect):
    spect_loss=torch.mean(torch.square(fused_result-input_spect))
    return spect_loss

# 定义结构相似性损失函数
def ssim_loss (fused_result,input_mri,input_spect ):
    ssim_loss=ssim(fused_result,torch.maximum(input_mri,input_spect))

    return ssim_loss

# 定义梯度损失函数
def gra_loss(fused_result, input_mri,input_spect):
    gra_loss = Gradient(fused_result, input_mri,input_spect)
    return gra_loss

# 定义高斯函数
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# 创建高斯窗口
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

# 定义结构相似性（SSIM）计算函数
def ssim(img1, img2, window_size=11, window=None, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map.mean()
    return 1-ret

class Sobelxy_hd(nn.Module):
    def __init__(self):
        super(Sobelxy_hd, self).__init__()
        # kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # Sobel算子用于计算水平梯度
        # kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]  # Scharr算子用于计算水平梯度
        kernely = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]  # Scharr算子用于计算垂直梯度
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)
            batch_list.append(torch.stack(tensor_list, dim=1))
        return torch.cat(batch_list, dim=0)

class charbonnier_Loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(charbonnier_Loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

# 丹姐的梯度计算函数
class gradient(nn.Module):
    def __init__(self):
        super(gradient, self).__init__()
        self.sobelconv = Sobelxy_hd()
        self.charbonnier_Loss = charbonnier_Loss()

    def forward(self,img_f, img_gray,img_b):
        grad_gray = self.sobelconv(img_gray)
        grad_b = self.sobelconv(img_b)
        # grad_gray = grad_gray.expand(B, C, K, W, H)
        x_grad_joint = torch.maximum(grad_b, grad_gray)
        generate_img_grad = self.sobelconv(img_f)
        loss_grad = self.charbonnier_Loss(generate_img_grad, x_grad_joint)
        return loss_grad

# 定义梯度计算函数
def Gradient(fused_img, input_mri,input_spect):
    gradient_model = gradient().to(device)
    g = gradient_model(fused_img, input_mri,input_spect)
    return g