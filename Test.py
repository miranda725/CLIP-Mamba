from PIL import Image
import numpy as np
import os
import torch
import time
import imageio
import torchvision.transforms as transforms
from Network import MODEL as net
import statistics
import glob
device = torch.device('cuda:1')
# 使用 GPU 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

model = net(in_channel=2)

model_path = "./models/model_300.pth"
# model_path = "./models/Abl_CLIP/model_300.pth"

model = model.to(device)
# state_dict = torch.load(model_path, weights_only=True, map_location=device)
# model.load_state_dict(state_dict)
model.load_state_dict(torch.load(model_path),strict=False)

'''------3通道图像---------'''
# def RGB2YCrCb(input_im):
#     print(input_im.shape)
#     # if input_im.size(1) != 3:
#     #     input_im = input_im.repeat(1, 3, 1, 1)
#     im_flat = input_im.transpose(1, 3).transpose(
#         1, 2).reshape(-1, 3)  # (nhw,c)
#     R = im_flat[:, 0]
#     G = im_flat[:, 1]
#     B = im_flat[:, 2]
#     Y = 0.299 * R + 0.587 * G + 0.114 * B
#     Cr = (R - Y) * 0.713 + 0.5
#     Cb = (B - Y) * 0.564 + 0.5
#     Y = torch.unsqueeze(Y, 1)
#     Cr = torch.unsqueeze(Cr, 1)
#     Cb = torch.unsqueeze(Cb, 1)
#     temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
#     out = (
#         temp.reshape(
#             list(input_im.size())[0],
#             list(input_im.size())[2],
#             list(input_im.size())[3],
#             3,
#         )
#         .transpose(1, 3)
#         .transpose(2, 3)
#     )
#     return out

'''------其他通道图像---------'''

def RGB2YCrCb(input_im):
    if input_im.size(1) > 3:
        # Take the first 3 channels if there are more than 3
        input_im = input_im[:, :3, :, :]
    elif input_im.size(1) < 3:
        raise ValueError("Input tensor must have at least 3 channels for RGB to YCrCb conversion.")

    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw, c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):

    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    print(im_flat.shape)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def fusion(MRI,SPECT):
    fuse_time = []

    tran = transforms.ToTensor()
    path1 = MRI
    path2 = SPECT
    img2 = tran(Image.open(SPECT))

    if img2.size(0)>1:
        img_spect = img2.unsqueeze(0)

        image_spect_ycrcb = RGB2YCrCb(img_spect)
        img_spect = image_spect_ycrcb[:, 0, :, :]
    else:
        img_spect = Image.open(path2).convert('L')
        img_spect = tran(img_spect)
    img_mri = Image.open(path1).convert('L')

    img1_org = tran(img_mri).to(device)
    img2_org = img_spect.to(device)
    #print(img2_org.shape)
    input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)

    input_img = input_img.to(device)

    model.eval()
    start = time.time()
    fusion_image = model(input_img)
    end = time.time()
    fuse_time.append(end - start)
   
    if img2.size(0)>1:
        fusion_ycrcb = torch.cat(
            (fusion_image, image_spect_ycrcb[:, 1:2, :, :],
             image_spect_ycrcb[:, 2:, :, :]),
            dim=1,
        )
        fusion_image = YCrCb2RGB(fusion_ycrcb)
        ones = torch.ones_like(fusion_image)
        zeros = torch.zeros_like(fusion_image)
        fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
        fusion_image = torch.where(
            fusion_image < zeros, zeros, fusion_image)
        fused_image = fusion_image.detach().cpu().numpy()
        #print(fused_image.shape)
        fused_image = fused_image.transpose((0, 2, 3, 1))

        fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
        )
    else:
        fused_image = fusion_image.detach().cpu().numpy()

    fused_image = np.uint8(255.0 * fused_image)
    fused_image = fused_image.squeeze()

    save_path = "./dataset/output/" + mri.split("/")[-1]  # 设置保存路径
    imageio.imwrite(save_path,fused_image )


if __name__ == '__main__':
    training_dir_MRI = "./dataset/test/MRI/" #50张
    folder_dataset_train_MRI = glob.glob(os.path.join(training_dir_MRI, '*'))
    training_dir_SPECT = "./dataset/test/SPECT/"
    folder_dataset_train_SPECT = glob.glob(os.path.join(training_dir_SPECT, '*'))

    Time =[]

    for i in range(len(folder_dataset_train_SPECT)):
        tic = time.time()
        mri = folder_dataset_train_MRI[i]
        spect = folder_dataset_train_SPECT[i]
        fusion(mri,spect)
        toc = time.time()
        Time.append(toc - tic)
        print(np.mean(Time))
