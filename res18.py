from torchvision import models
model = models.resnet18(pretrained=True)

from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os.path
import torch
import torchvision
import matplotlib.pyplot as plt
import time

class Config():
    training_dir = "/home/shared/CS341/Dataprocessing/train"
    testing_dir = "/home/shared/CS341/Dataprocessing/finaltest/train"
    train_batch_size = 64
    train_number_epochs = 100
    
class NetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img0 = Image.open(img0_tuple[0])
        img0 = img0.convert("RGB")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)

        if self.transform is not None:
            img0 = self.transform(img0)
        
        return img0, img0_tuple[1]
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
    
transform_dataset = transforms.Compose([transforms.Scale((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

folder_dataset = dset.ImageFolder(root=Config.training_dir)
dataset = NetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform_dataset
                                       ,should_invert=False)
data_image = {x:dset.ImageFolder(root = Config.training_dir,
                                     transform = transform_dataset)
              for x in ["train", "val"]}

data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],
                                                batch_size = 64,
                                                shuffle = True)
                     for x in ["train", "val"]}

classes = data_image["train"].classes
classes_index = data_image["train"].class_to_idx
print(classes)
print(classes_index)


use_gpu = torch.cuda.is_available()
print(torch.cuda.device_count())
torch.cuda.set_device(2)
print(torch.cuda.current_device())


num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, 256),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(p=0.5),
                               torch.nn.Linear(256, 15))

if use_gpu:
    model = model.cuda()

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters())


print(model)

import os
if os.path.exists('model_res18_finetune.pkl'):
    checkpoint = torch.load('model_res18_finetune.pkl')
    model.load_state_dict(checkpoint)

train_batch_size = 64
fh = open('report_res.txt','w')
fc = open('comp_res.txt','w')
n_epochs = 10
for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-"*10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0 
        batch = 0
        for data in data_loader_image[param]:
            batch += 1
            X, y = data
            if use_gpu:
                X, y  = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)
        
            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            loss = cost(y_pred, y)
            if param =="train":
                loss.backward()
                optimizer.step()
            running_loss += loss.data[0]
            running_correct += torch.sum(pred == y.data)
            if batch%50 == 0 and param =="train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                      batch, running_loss/(train_batch_size*batch), 100*running_correct/(train_batch_size*batch)))
                fh.write("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                      batch, running_loss/(train_batch_size*batch), 100*running_correct/(train_batch_size*batch)))
            torch.save(model.state_dict(), "model_res18_finetune.pkl")
        epoch_loss = running_loss/len(data_image[param])
        epoch_correct = 100*running_correct/len(data_image[param])

        print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
        fh.write("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since   
    print("Training time is:{:.0f}m {:.0f}s".format(now_time//60, now_time%60))