
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# propias
from train_data import SIGNSData
from train_red_v2 import Net
from train_logging import RunningMetric

import random
import time

def train_and_evaluate(model, optimizer, loss_fn, data_loaders, device, num_epochs = 10, lr = 0.001):

  # modificamos lr del optimizer
  for g in optimizer.param_groups:
        g['lr'] = lr
      
        
  for epoch in range(num_epochs):
      time_init = time.time()
      print('\tEpoch {}/{}'.format(epoch+1,num_epochs))
      print('\t----------------------')
      for phase in ['train', 'val']:
        if phase == 'train': 
          model.train()
        else:
            model.eval()
      
        running_loss = RunningMetric() # pérdida 
        running_acc = RunningMetric() # accuracy 

        for inputs, targets in data_loaders[phase]:
          
            inputs, targets = inputs.to(device), targets.to(device)

            # gradiente a cero
            optimizer.zero_grad()
            
            # solo calculamos perdida en phase == train
            with torch.set_grad_enabled(phase == 'train'):
              # core
              outputs = net(inputs)
              # lo que predice la red sobre el mini batch size que entrega dataloader
              _, preds = torch.max(outputs,1) 
              # funcion de perdida
              loss = loss_fn(outputs, targets)
              
              if phase == 'train':
                # magias: gradientes calculados automáticamente
                loss.backward()
                # actualiza los parametros
                optimizer.step() 
            
            batch_size = inputs.size()[0]
            # *batch_size porque la perdida es un promedio
            running_loss.update(loss.item()*batch_size, 
                                batch_size) 
            
            running_acc.update(torch.sum((preds == targets)).float(),
                              batch_size)
            
        print('\tPhase {} Loss {:.4f}, Acc: {:.4f}, Time: {} sec.'.format(phase, running_loss(), running_acc(), round(time.time()-time_init,1)))
            
  return model

# Data Augmentation: No aumenta la cantidad de imagenes, solo en cada epoch serán datos distintos
transform = transforms.Compose(
    [ transforms.RandomHorizontalFlip(), #rotar 
      transforms.ToTensor(), # a tensor
      transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ) #defaults buenos
    ]
)

train_data =  SIGNSData(base_dir='./datasets/64x64_SIGNS', 
                        split="train",
                        transform=transform
                        )

val_data =  SIGNSData(base_dir='./datasets/64x64_SIGNS', 
                        split="val",
                        transform=transform
                        )


test_data =  SIGNSData(base_dir='./datasets/64x64_SIGNS', 
                        split="test",
                        transform=transform
                        )
# DataLoader: entrega por batch size el dataset a la red neuronal
train_loader = DataLoader(train_data, batch_size=32) # batch de 32 imagenes
val_loader = DataLoader(val_data, batch_size=32) # batch de 32 imagenes
test_loader = DataLoader(test_data, batch_size=32) # batch de 32 imagenes

data_loaders = {'train': train_loader,
                'val': val_loader,
                'test': test_loader
                }


# 32 numero de canales, no tiene que ver con el nro de batchs
device = torch.device('cpu')
net = Net(num_channels=32)
loss_fn = nn.NLLLoss()

# Optimizador: Stochat Gradent Descent
# lr: learning rate
optimizer = optim.SGD(net.parameters(), lr= 1e-3, momentum =0.9)

lrs = [ 10 ** (-random.randint(3,7)) for _ in range(3)]

for lr in lrs:
  print('LR: {}'.format(lr))
  print('-'*50)
  train_and_evaluate(net, optimizer, loss_fn, data_loaders, device, 500, lr)