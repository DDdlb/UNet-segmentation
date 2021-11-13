import numpy as np
import torch
import wandb
from tensorboardX import SummaryWriter
from model.UNet import UNet
from dataset import SegDataset
from torch.utils.data import DataLoader
import SimpleITK as sitk
import cv2
from PIL import Image

DEVICE = 'cuda:3'
BATCH_SIZE = 8
MODEL_PATH = '/home/user2/selftest/segmentation/UNet/result/unet'

base_dir = '/home/user2/selftest/segmentation/UNet/data/Water_Bodies_Dataset'
img_dir = 'Images'
lab_dir = 'Masks'


# wandb.init(project='UNet', entity='dlb')

def get_dataloader(name):
    dataset = SegDataset(
        base_dir=base_dir,
        img_dir=img_dir,
        label_dir=lab_dir,
        name=name,
        transform=True
    )
    dataLoader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataLoader

def get_dice(pre, lab):
    # pre : [256, 256]    lab: [256, 256]

    gt_n_pred = ((pre + lab) == 2).sum()
    # print('n:' + str(gt_n_pred))
    gt = lab.sum()
    # print('gt:'+str(gt))
    pred = pre.sum()
    # print('pred:'+str(pred))

    dice = 2.0*gt_n_pred/(gt + pred)
    return dice

def main():
    net = UNet()
    net.load_state_dict(torch.load(MODEL_PATH + '/49.pth'))
    dataLoader = get_dataloader('test')

    total_dice = 0.0
    num = 0
    for index, (image, label) in enumerate(dataLoader):

        out = net(image)
        # print(out.shape, image.shape, label.shape)


        for i in range(len(label)):
            pre = out[i] < 0
            # print(pre, label[i][0])
            dice = get_dice(pre[0], label[i][0])
            total_dice += dice
            num += 1
            print(dice)

        pre = out[0] < 0
        if index % 5 == 4:
            print("save:")
            print(image[0].numpy().shape)
            print(pre[0].shape)
            print(label[0][0].shape)
            out_img = Image.fromarray(pre[0].squeeze().numpy())
            out_img.save('/home/user2/selftest/segmentation/UNet/result/unet/out'+str(index)+'.bmp', 'bmp')
            src_img = np.uint8(image[0]).transpose((1, 2, 0))
            src_img = Image.fromarray(src_img).convert('RGB')
            src_img.save('/home/user2/selftest/segmentation/UNet/result/unet/src'+str(index)+'.png', 'png')
            lab_img = np.uint8(label[0][0])
            lab_img = Image.fromarray(lab_img)
            lab_img.save('/home/user2/selftest/segmentation/UNet/result/unet/lab'+str(index)+'.bmp', 'bmp')




    avg_dice = total_dice / num
    print("avg_dice:")
    print(avg_dice)








if __name__ == '__main__':
    main()