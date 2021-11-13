import os

from cv2 import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import numpy as np
from torchvision.transforms import transforms
from skimage import transform

class SegDataset(Dataset):
    def __init__(self, base_dir, img_dir, label_dir, name,transform=None):
        super(SegDataset, self).__init__()
        self.base_dir = base_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.name = name
        self.img_list = os.listdir(base_dir + '/' + img_dir)
        self.img_list.sort()

        self.label_list = os.listdir(base_dir + '/' + label_dir)
        self.label_list.sort()

        train_len = int(len(self.img_list) * 0.85)
        if name == 'train':

            self.img_list = self.img_list[:train_len]
            self.label_list = self.label_list[:train_len]
        else:
            self.img_list = self.img_list[train_len:]
            self.label_list = self.label_list[train_len:]
        print(len(self.img_list))



    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # print('item')
        img_path = self.base_dir + '/' + self.img_dir + '/' + self.img_list[index]
        label_path = self.base_dir + '/' + self.label_dir + '/' + self.label_list[index]

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        img = img.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        label = label[0]
        # print(label.shape)



        if self.transform:
            img_trans = transforms.Compose([
                Resize((3, 256, 256), 1),
                ToTensor()
            ])
            label_trans = transforms.Compose([
                Resize((256, 256), 0),
                ToTensor()
            ])
            img = img_trans(img)
            label = label_trans(label)
            label = label.reshape(1, 256, 256)
            # print(label[0][1][1].type())
        label = label>100


        # print(img.shape, label.shape)
        return img.float(), label.long()

class Resize(object):
    def __init__(self, output_size: tuple, flag):
        self.output_size = output_size
        self.flag = flag

    def __call__(self, img):
        if self.flag:
            img = transform.resize(img, self.output_size)
        else:
            img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_NEAREST)

        return img

class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img)

if __name__ == '__main__':

    dataset = SegDataset(base_dir='./data/Water_Bodies_Dataset',
                         img_dir='Images',
                         label_dir='Masks',
                         name='train',
                         transform=True
                         )
    dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    writer = SummaryWriter()
    img_list = []
    lab_list = []
    srcimg_list = []
    srclab_list = []
    for index, (image, label) in enumerate(dataLoader):

        print(label)
        if index == 3:
            break
        srcimg_list.append(np.array(image[0]))
        srclab_list.append((np.array(label[0])))
        image = image.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        print(image.shape)
        print(label.shape)
        img_list.append(np.array(image[0]))
        lab_list.append(np.array(label[0]))

    writer.add_images('img', img_list, 0, dataformats='CHW')
    writer.add_images('label', lab_list, 0, dataformats='CHW')
    writer.add_images('srcimg', srcimg_list, 0, dataformats='CHW')
    writer.add_images('srclab', srclab_list, 0, dataformats='CHW')
    writer.close()





