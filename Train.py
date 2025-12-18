import os
# pip install git+https://github.com/openai/CLIP.git
import argparse
from tqdm import tqdm
import pandas as pd
import joblib
import glob
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Network import MODEL as net
from losses import mri_loss, spect_loss, ssim_loss, gra_loss
import clip
from PIL import Image
import torchvision
import torch.nn.functional as F

device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_path = "./models/ViT-B-32.pt"
c_model, preprocess = clip.load("ViT-B/32", device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MRI_CT_model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight', default=[1, 1, 10, 100], type=float) #v1
    # parser.add_argument('--weight', default=[3, 3, 12, 40], type=float) #v4,效果较差
    # parser.add_argument('--weight', default=[3, 3, 20, 100], type=float)  # v5 不行
    args = parser.parse_args()
    return args


class GetDataset(Dataset):
    def __init__(self, imageFolderDatasetspect, imageFolderDatasetmri, transform=None):
        self.imageFolderDataset_spect = imageFolderDatasetspect
        self.imageFolderDataset_mri = imageFolderDatasetmri
        self.transform = transform

    def __getitem__(self, index):
        mri = self.imageFolderDataset_mri[index]
        spect = self.imageFolderDataset_spect[index]

        mri = Image.open(mri).convert('L')
        spect = Image.open(spect).convert('L')
       
        if self.transform is not None:
            resize = transforms.Resize((256, 256))  # 调整图像尺寸为 256x256
            mri = resize(mri)
            spect = resize(spect)

            tran = transforms.ToTensor()
            mri = tran(mri)
            spect = tran(spect)
            input = torch.cat((mri, spect), -3)
            return input, mri, spect

    def __len__(self):
        return len(self.imageFolderDataset_mri)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion_ir, criterion_vi, criterion_ssim, criterion_gra, optimizer, epoch,
          scheduler=None):
    losses = AverageMeter()
    losses_mri = AverageMeter()
    losses_vi = AverageMeter()
    losses_ssim = AverageMeter()
    losses_gra = AverageMeter()
    loss_clip1_meter = AverageMeter()
    loss_clip2_meter = AverageMeter()
    weight = args.weight

    model.train()

    for i, (input, mri, spect) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        mri = mri.cuda()
        spect = spect.cuda()
        out = model(input)
        mri_3 = torch.cat([mri] * 3, dim=1)
        spect_3 = torch.cat([spect] * 3, dim=1)
        out_3 = torch.cat([out] * 3, dim=1)
        mri_3 = F.interpolate(mri_3, size=(224, 224), mode='bilinear', align_corners=True)
        spect_3 = F.interpolate(spect_3, size=(224, 224), mode='bilinear', align_corners=True)
        out_3 = F.interpolate(out_3, size=(224, 224), mode='bilinear', align_corners=True)

        with torch.no_grad():
            mri_features = c_model.encode_image(mri_3)
            spect_features = c_model.encode_image(spect_3)
            out_features = c_model.encode_image(out_3)
        # 定义文本描述a vivid image with detailed background and obvious objects
        #text_description = "a comprehensive image with clearly defined anatomical structures and distinct lesion features"
        text_description = "a clear image with clearly defined anatomical structures and distinct lesion features"  #该句话用于训练功能-解剖图像
        # 该句话用于训练MRI和CT融合的模型
        # text_description = "A high-contrast fused image with clear bone structures from CT(改成全称) and detailed soft tissue visualization from MRI, highlighting both anatomical precision and pathological features"

        # 文本编码
        text_input = torch.tensor(clip.tokenize(text_description)).to(device)
        text_feature = c_model.encode_text(text_input)

        # 计算相似度
        similarities = (text_feature @ out_features.T).squeeze(0)
        similarities = torch.sigmoid(similarities)
        # 计算相似度均值
        loss_clip2 = 1 - similarities.mean()
        loss_clip1 = torch.mean(torch.square(out_features - mri_features)) + torch.mean(
            torch.square(out_features - spect_features))
        loss_mri = weight[0] * criterion_ir(out, mri)
        loss_spect = weight[1] * criterion_vi(out, spect)
        loss_ssim = weight[2] * criterion_ssim(out, mri, spect)
        loss_gra = weight[3] * criterion_gra(out,mri, spect)
        # loss = loss_ir + loss_vi + loss_ssim+ loss_clip2
        loss = loss_ssim + loss_gra + loss_clip1 + loss_clip2
        losses.update(loss.item(), input.size(0))
        losses_mri.update(loss_mri.item(), input.size(0))
        loss_spect.update(loss_spect.item(), input.size(0))
        losses_ssim.update(loss_ssim.item(), input.size(0))
        losses_gra.update(loss_gra.item(), input.size(0))
        loss_clip1_meter.update(loss_clip1.item(),input.size(0))
        loss_clip2_meter.update(loss_clip2.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ir', losses_mri.avg),
        ('loss_vi', loss_spect.avg),
        ('loss_ssim', losses_ssim.avg),
        ('loss_gra', losses_gra.avg),
        ('loss_clip1', loss_clip1_meter.avg),
        ('loss_clip2', loss_clip2_meter.avg)
    ])

    return log


def main():
    args = parse_args()

    import os

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)
    else:
        print(f"Directory 'models/{args.name}' already exists.")

    # 确保目录存在
    os.makedirs(os.path.join('models', args.name), exist_ok=True)

    with open(os.path.join('models', args.name, 'args.txt'), 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)
    cudnn.benchmark = True

    training_dir_mri = "./dataset/train/MRI/"
    folder_dataset_train_mri = glob.glob(os.path.join(training_dir_mri, '*'))

    # 修改这里的目录路径
    training_dir_spect = "./dataset/train/PET/"
    folder_dataset_train_spect = glob.glob(os.path.join(training_dir_spect, '*'))

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

    dataset_train = GetDataset(imageFolderDatasetvi=folder_dataset_train_spect,
                               imageFolderDatasetir=folder_dataset_train_mri,
                               transform=transform_train)

    train_loader = DataLoader(dataset_train,
                              shuffle=True,
                              batch_size=args.batch_size)

    model = net(in_channel=2)
    if use_gpu:
        model = model.cuda()
        model.cuda()

    else:
        model = model
    criterion_mri = mri_loss
    criterion_spect = spect_loss
    criterion_ssim = ssim_loss
    criterion_gra = gra_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'loss',
                                'loss_mri',
                                'loss_spect',
                                'loss_ssim',
                                'loss_gra',
                                'loss_clip1',
                                'loss_clip2'
                                ])

    for epoch in range(args.epochs):

        train_log = train(args, train_loader, model, criterion_mri, criterion_spect, criterion_ssim, criterion_gra,
                          optimizer, epoch)
        tmp = pd.Series([
            epoch + 1,
            train_log['loss'],
            train_log['loss_mri'],
            train_log['loss_spect'],
            train_log['loss_ssim'],
            train_log['loss_gra'],
            train_log['loss_clip1'],
            train_log['loss_clip2']
        ], index=['epoch', 'loss', 'loss_mri', 'loss_spect', 'loss_ssim', 'loss_gra','loss_clip1', 'loss_clip2'])

        log = pd.concat([log, tmp.to_frame().T], ignore_index=True)
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch + 1) % args.name)


if __name__ == '__main__':
    main()


