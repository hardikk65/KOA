import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import timm
import random
import shutil
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pytorch_lightning as pl
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from torchvision import transforms
def import_dataset(clip_limit=None,tilegridsize=None,Clahe = False):
    
    if Clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tilegridsize)
    
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    train_path = "/kaggle/input/new-augmented-images/final_1/train/"
    val_path = "/kaggle/input/knee-osteoarthritis-dataset-with-severity/val/"
    test_path = "/kaggle/input/knee-osteoarthritis-dataset-with-severity/test/"
    
    for label in os.listdir(train_path):
        train_dict[int(label)] = os.listdir(train_path + label)
    for label in os.listdir(test_path):
        test_dict[int(label)] = os.listdir(test_path + label)
    for label in os.listdir(val_path):
        val_dict[int(label)] = os.listdir(val_path + label)
        
    train_images = {}
    val_images = {}
    test_images = {}
    
    for keys in train_dict.keys():  
        normal_train_images = [np.array(Image.open(train_path + str(keys) + "/" + images)) for images in train_dict[keys]]
        normal_test_images = [np.array(Image.open(test_path + str(keys) + "/" + images)) for images in test_dict[keys]]
        normal_val_images = [np.array(Image.open(val_path + str(keys) + "/" + images)) for images in val_dict[keys]]
        
        if Clahe:
            clahe_train_images = [clahe.apply(np.array(Image.open(train_path + str(keys) + "/" + images))) for images in train_dict[keys]]
            clahe_test_images = [clahe.apply(np.array(Image.open(test_path + str(keys) + "/" + images))) for images in test_dict[keys]]
            clahe_val_images = [clahe.apply(np.array(Image.open(val_path + str(keys) + "/" + images))) for images in val_dict[keys]]
            
            train_images[keys] =  clahe_train_images
            test_images[keys] =  clahe_test_images
            val_images[keys] =  clahe_val_images
            
        else:
            train_images[keys] = normal_train_images
            test_images[keys] = normal_test_images
            val_images[keys] = normal_val_images 
            
            
    return train_images,test_images,val_images
def create_tensors(train_images,test_images,val_images):
    
    trlabel0 = [0 for i in range(len(train_images[0]))]
    trlabel1 = [1 for i in range(len(train_images[1]))]
    trlabel2 = [2 for i in range(len(train_images[2]))]
    trlabel3 = [3 for i in range(len(train_images[3]))]
    trlabel4 = [4 for i in range(len(train_images[4]))]
    
    tslabel0 = [0 for i in range(len(test_images[0]))]
    tslabel1 = [1 for i in range(len(test_images[1]))]
    tslabel2 = [2 for i in range(len(test_images[2]))]
    tslabel3 = [3 for i in range(len(test_images[3]))]
    tslabel4 = [4 for i in range(len(test_images[4]))]
    
    vlabel0 = [0 for i in range(len(val_images[0]))]
    vlabel1 = [1 for i in range(len(val_images[1]))]
    vlabel2 = [2 for i in range(len(val_images[2]))]
    vlabel3 = [3 for i in range(len(val_images[3]))]
    vlabel4 = [4 for i in range(len(val_images[4]))]
    
    transforms1 = v2.Compose([
        v2.Lambda(lambda x: cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)),
        v2.ToTensor(),
        v2.Resize((224,224)),
        v2.Lambda(lambda x: x.view(224,224)),
        v2.Lambda(lambda x: x.numpy())

    ])
    
    transforms2 = v2.Compose([
        v2.ToTensor(),
        v2.Resize((224,224)),
        v2.Lambda(lambda x: x.view(224,224)),
        v2.Lambda(lambda x: x.numpy())
    ])
    
    for keys in range(5):
        for i in range(len(train_images[keys])):
            if train_images[keys][i].shape[-1] == 3:
                train_images[keys][i] = transforms1(train_images[keys][i])
            elif train_images[keys][i].shape[-1] > 224 or train_images[keys][i].shape[-1] < 224:
                train_images[keys][i] = transforms2(train_images[keys][i])
        
            
    
    
    training_image  = torch.tensor(torch.cat((torch.tensor(train_images[0]),torch.tensor(train_images[1]),torch.tensor(train_images[2]),torch.tensor(train_images[3]),torch.tensor(train_images[4])),0),dtype = torch.float32)
    training_labels = torch.tensor(torch.cat((torch.tensor(trlabel0),torch.tensor(trlabel1),torch.tensor(trlabel2),torch.tensor(trlabel3),torch.tensor(trlabel4)),0))
    training_image = training_image.view(training_image.shape[0],1,224,224)
    
    testing_image  = torch.tensor(torch.cat((torch.tensor(test_images[0]),torch.tensor(test_images[1]),torch.tensor(test_images[2]),torch.tensor(test_images[3]),torch.tensor(test_images[4])),0),dtype = torch.float32)
    testing_labels = torch.tensor(torch.cat((torch.tensor(tslabel0),torch.tensor(tslabel1),torch.tensor(tslabel2),torch.tensor(tslabel3),torch.tensor(tslabel4)),0))
    testing_image = testing_image.view(testing_image.shape[0],1,224,224)
    
    val_image  = torch.tensor(torch.cat((torch.tensor(val_images[0]),torch.tensor(val_images[1]),torch.tensor(val_images[2]),torch.tensor(val_images[3]),torch.tensor(val_images[4])),0),dtype = torch.float32)
    val_labels = torch.tensor(torch.cat((torch.tensor(vlabel0),torch.tensor(vlabel1),torch.tensor(vlabel2),torch.tensor(vlabel3),torch.tensor(vlabel4)),0))
    val_image = val_image.view(val_image.shape[0],1,224,224)
    
    return training_image,training_labels,testing_image,testing_labels,val_image,val_labels
def create_dataset(training_image,training_labels,testing_image,testing_labels,val_image,val_labels,batch_size = 4):
    train_dataset = TensorDataset(training_image,training_labels)
    val_dataset = TensorDataset(val_image,val_labels)
    test_dataset = TensorDataset(testing_image,testing_labels)
    
    train_loader = DataLoader(train_dataset,shuffle = True,batch_size = batch_size)
    val_loader = DataLoader(val_dataset,shuffle = True,batch_size = batch_size)
    test_loader = DataLoader(test_dataset,shuffle = True,batch_size = batch_size)
    
    return train_loader,test_loader,val_loader
def testing(model,test_loader):
    device = 'cuda'
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        test_loss = 0.0
        test_accuracy = 0
        for i,(images, labels) in enumerate(test_loader):

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            
            loss = corn_loss(outputs, labels, 5)
            
            
            predicted_labels = corn_label_from_logits(outputs).float()
            
            n_samples += labels.size(0)
            n_correct += (predicted_labels == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

        
        confusion_mat = confusion_matrix(all_labels, all_predictions)


        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the {n_samples} test images:{acc:.4f}%')
        
        return confusion_mat
    
train_images,test_images,val_images = import_dataset(Clahe = False)
training_image,training_labels,testing_image,testing_labels,val_image,val_labels = create_tensors(train_images,test_images,val_images)
tr_image = training_image[:,:,65:180,:]
ts_image = testing_image[:,:,65:180,:]
v_image = val_image[:,:,65:180,:]
train_loader,test_loader,val_loader = create_dataset(tr_image,training_labels,ts_image,testing_labels,v_image,val_labels,batch_size = 28)
original_model= torchvision.models.vgg19_bn(pretrained = True)
original_model.features[0] = nn.Conv2d(1,64,kernel_size = (3,3),stride = 1,padding = 1)
original_model.avgpool = nn.AvgPool2d(2)
original_model.classifier = nn.Sequential(
                            nn.Linear(512*3,512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(inplace = True),
                            nn.Linear(512,4)
)

no_of_layers = 16
for i, params in enumerate(original_model.parameters()):
    if i < no_of_layers:
        params.requires_grad = False
original_model.to('cuda')
optimizer = torch.optim.Adam(original_model.parameters(),lr = 0.0001)
epochs = 8
training_loss = 0.0
training_accuracy = 0
validation_accuracy = 0
device = 'cuda'
for epoch in range(epochs):
    n_samples_train = 0
    n_samples_val = 0
    original_model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = original_model(images)
#         print(output.shape)
        loss = corn_loss(output, labels, 5)
        n_samples_train += labels.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predicted_labels = corn_label_from_logits(output).float()
        training_accuracy += (predicted_labels == labels).sum().item()

        if (i + 1) % 512 == 0:
            print(f'Training - Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item():.4f}')

    print(f'Training Accuracy - Epoch: {epoch + 1}: {training_accuracy * 100 / n_samples_train:.2f}%')
    training_accuracy = 0
    
    original_model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

            images = images.to(device)
            labels = labels.to(device)
            val_output = original_model(images)
            
            loss = corn_loss(val_output,labels, 5)
            
            predicted_labels = corn_label_from_logits(val_output).float()
            
            validation_accuracy += (predicted_labels == labels).sum().item()
            n_samples_val += labels.size(0)

    print(f'Validation Accuracy - Epoch: {epoch + 1}: {validation_accuracy * 100 / n_samples_val:.2f}% Loss: {loss.item():.4f}')
    validation_accuracy = 0
conf_mat = testing(original_model,test_loader)
confusion_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusion_mat_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix with Accuracy")
plt.show()
