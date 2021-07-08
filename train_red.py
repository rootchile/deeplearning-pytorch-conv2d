import torch
import torch.nn as nn
import torch.nn.functional as F # capas sin parametros

class Net(nn.Module):
    
    def __init__(self, num_channels):
        # num_channels: cantidad de canales por los cuales va a expandir la imagen
        super(Net, self).__init__()
        self.num_channels = num_channels
        
        # Capas de la red
        # extractores de features
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride = 1, padding=1) #3: canales de la imagen, rgb
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2,3, stride = 1, padding=1) # expando los canales de salida
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride = 1, padding=1) 
        
        #fc: linear o fully connected
        # max_pool: divide los canales por 2
        self.fc1 = nn.Linear(self.num_channels * 4 * 8 * 8, self.num_channels*4) # la salida es arbitraria
        self.fc2 = nn.Linear(self.num_channels*4, 6) # por que 6? el dataset tiene 6 signos
        
    def forward(self, x):
        
        #Empieza con 3 canales x 64 px x 64 px
        x = self.conv1(x)  # num_channel x 64 x 64
        x = F.relu(F.max_pool2d(x,2)) # 2d igual q conv2d, dividirá el tamaño de la imagen por dos => num_channel x 32 x 32
        
        x = self.conv2(x)  # num_channel x 2 x 32 x 32
        x = F.relu(F.max_pool2d(x,2)) # num_channel * 2 x 16 x 16
      
        x = self.conv3(x)  # num_channel x 4 x 16 x 16
        x = F.relu(F.max_pool2d(x,2)) #  # num_channel * 4 x 8 x 8 = TAMAÑO TENSOR DE ENTRADA, PARA FC1

        # flatten para capas lineales
        x = x.view(-1, self.num_channels * 4 * 8 * 8 ) # -1 = flatten, tamaño tensor
        
        #fc 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # log_softmax = para tener probabilidades en la salida
        
        x = F.log_softmax(x, dim=1)
        
        return x
        