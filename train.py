import os
import argparse
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dataset.dataset import ModelNet10Dataset
from dataset.transforms import mode_transform, default_transforms
from models.pointnet import PointNet
from losses.loss import pointnetloss

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)


def train(model, train_loader, val_loader,  epochs, lr, save_model_path):
    optimizer= torch.optim.AdamW(model.parameters(), lr= lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    max_acc = 0
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                running_loss = 0.0
        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            if val_acc > max_acc:
                # save the model with best accuracy on validation set
                torch.save(model.state_dict(), f"{save_model_path}/model_epoch_{epoch}.pth")
                max_acc = val_acc
            print('Valid accuracy: %d %%' % val_acc)

        
        


def args_parser():
    parser = argparse.ArgumentParser(description='Process some arguments...!')
    parser.add_argument('--root_path',type=str,required = True,help='Path to train workspace file!')
    parser.add_argument('--train_path',type=str,required = True,help='Path to train csv file!')
    parser.add_argument('--val_path',type=str,required = True,help='Path to validation csv file!')
    parser.add_argument('--model_name',type=str,default='pointnet',help='Model name to train!')
    parser.add_argument('--train_path',type=str,required = True,help='Path to train csv file!')
    parser.add_argument('--save_model_path',type=str,required = True,help='Path to save model from training!')
    parser.add_argument('--num_classes',type=int,default=10,help='Number of class to classify!')
    parser.add_argument('--epochs',type=int,default=100,help='Number of epochs to train !')
    parser.add_argument('--batch_size',type=int,default=16,help='Number of batch size !')
    parser.add_argument('--lr',type=float,default=0.01,help='Learning rate!')
    return parser

if __name__ == '__main__':
    args = args_parser()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_tranforms = mode_transform(mode='train')
    trainset = ModelNet10Dataset(root=args.root_path, csv= args.train_path, mode ='train', transforms= train_tranforms)
    testset = ModelNet10Dataset(root=args.root_path, csv= args.val_path, mode='test', transforms=default_transforms())
    train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle =True, num_workers=4)
    valid_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = PointNet(classes=args.num_classes).to(device)
    train(model= model, train_loader=train_loader, val_loader=valid_loader, epochs=args.epochs, lr= args.lr, save_model_path=args.save_model_path)