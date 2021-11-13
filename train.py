import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import SegDataset, Resize, ToTensor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model.UNet import UNet
import torch.optim as optim
import wandb

DEVICE = 'cuda:3'
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
SAVE_PATH = './result/unet'

base_dir = './data/Water_Bodies_Dataset'
img_dir = 'Images'
lab_dir = 'Masks'

wandb.init(project='UNet', entity="dlb")

print(torch.cuda.is_available())


def train(net, loss_fn, optimizer, dataLoader):
    net.to(DEVICE)
    for epoch in range(EPOCHS):
        print("EPOCH: " + str(epoch))

        # train net
        net.train()


        for index, (image, label) in enumerate(tqdm(dataLoader)):
            # Set the Optimizer gradient to 0 first
            optimizer.zero_grad()

            # change image type to FloatTensor
            image = image.to(DEVICE)
            # image = image.type(torch.cuda.FloatTensor)


            # print(label.type())
            # calculate out
            out = net(image)
            # calculate loss
            loss = loss_fn(out, label.squeeze().to(DEVICE))

            # backward
            loss.backward()

            # update parameters
            optimizer.step()

            wandb.log({'loss': loss, 'mask': out})

    pth_path = SAVE_PATH + '/' + str(epoch) + '.pth'
    torch.save(net.state_dict(), pth_path)



def main():
    # loader data
    dataset = SegDataset(
        base_dir=base_dir,
        img_dir=img_dir,
        label_dir=lab_dir,
        name='train',
        transform=True
    )

    dataLoader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


    # net
    net = UNet()


    # loss
    loss_fn = nn.CrossEntropyLoss()

    # optim
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    train(net, loss_fn, optimizer, dataLoader)

if __name__ == '__main__':
    main()