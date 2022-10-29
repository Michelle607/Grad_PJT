from gc import callbacks
import pyaudio
import librosa
import numpy as np
from tqdm.auto import tqdm
import os
import pandas as pd
import time

import torchvision.datasets as datasets # 데이터셋 집합체
import torchvision.transforms as transforms # 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴
from torch.utils.data import DataLoader, Dataset

import torch.nn.init as init
import torch.nn as nn # 신경망들이 포함됨
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당
batch_size = 15

def preprocess_dataset(data):
    mfccs = []
    for i in data:
        extracted_features = librosa.feature.mfcc(y=i.flatten(),
                                            sr = 44100,
                                            n_mfcc=40)
        mfccs.append(extracted_features)
        
    return mfccs

def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    correct_ = 0
    with torch.no_grad():
        for wav in tqdm(iter(test_loader)):
            wav = wav.to(device)

            pred_logit = model(wav)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
          
    return model_pred

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = torch.nn.Sequential(
            nn.Conv2d(40, 16, kernel_size=2, stride=1, padding=1), #cnn layer
            #nn.ELU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2), #pooling layer
            #nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1), #cnn layer
            #nn.ELU(), #activation function
            #nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2), #pooling layer
            nn.Dropout2d(p=0.2),

            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1), #cnn layer
            #nn.ELU(), #activation function
            #nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2), #pooling layer
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1), #cnn layer
            #nn.ELU(), #activation function
            #nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2), #pooling layer
            nn.Dropout2d(0.2))
        
        self.fc_layer = nn.Sequential(
            nn.Linear(128, 50),
            nn.ELU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm1d(50),
            nn.Linear(50, 12), #fully connected layer(ouput layer)
        )    

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data) 
            m.bias.data.fill_(0)
          if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight.data)
            m.bias.data.fill_(0)
        
    def forward(self, x):
        out = self.layer(x)
        out = torch.flatten(out, start_dim=1) # N차원 배열 -> 1차원 배열
        out = self.fc_layer(out)

        return out

class CustomDataset(Dataset):
    def __init__(self, X, y, train_mode=True, transforms=None): #필요한 변수들을 선언
        self.X = X
        self.y = y
        self.train_mode = train_mode
        self.transforms = transforms

    def __getitem__(self, index): #index번째 data를 return
        X = self.X[index]
        
        if self.transforms is not None:
            X = self.transforms(X)

        if self.train_mode:
            y = self.y[index]
            return X, y
        else:
            return X
    
    def __len__(self): #길이 return
        return len(self.X)
    
    
class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 2**10
        self.p = None
        self.stream = None
        self.n_mfcc = 40

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        mfccs = preprocess_dataset(numpy_array)
        data_mfccs = np.array(mfccs)
        data_mfccs = data_mfccs.reshape(-1, data_mfccs.shape[1], data_mfccs.shape[2], 1)
        dataset = CustomDataset(X=data_mfccs, y= None, train_mode=False)
        loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)
        
        # model 불러오기
        model = CNN().to(device)
        checkpoint = torch.load('/Users/anjaeu/Code/졸업프로젝트/weights.best.cnn.pth', map_location=device)
        model.load_state_dict(checkpoint)
    
        # Inference
        preds = predict(model, loader, device)
        value = preds[0]
        print('예측 Label은 {}'.format(value))
                
        return None, pyaudio.paContinue
    

    def mainloop(self):
        while (self.stream.is_active()): 
            time.sleep(3.0)
            
            
while(True):
    
    audio = AudioHandler()
    audio.start()
    audio.mainloop()