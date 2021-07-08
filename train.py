
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# propias
from train_data import SIGNSData
from train_red import Net
from train_logging import RunningMetric

train_data =  SIGNSData(base_dir='./datasets/64x64_SIGNS', 
                        split="train",
                        transform=transforms.ToTensor()
                        )
# DataLoader: entrega por batch size el dataset a la red neuronal
dataloader = DataLoader(train_data, batch_size=32) # batch de 32 imagenes

# dataloader es iterable
# for inputs, targets in dataloader:
#     print(inputs)
#     print(targets)
#     break

# 32 numero de canales, no tiene que ver con el nro de batchs
device = torch.device('cpu')
net = Net(num_channels=32)
loss_fn = nn.NLLLoss()

# Optimizador: Stochat Gradent Descent
# lr: learning rate
optimizer = optim.SGD(net.parameters(), lr= 1e-3, momentum =0.9)

# Loop de train

num_epochs = 100 # pasadas por el dataset completo

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1,num_epochs))
    print('-'*10)
    
    running_loss = RunningMetric() # pérdida 
    running_acc = RunningMetric() # accuracy 

    for inputs, targets in dataloader:
       
        inputs, targets = inputs.to(device), targets.to(device)

        # gradiente a cero
        optimizer.zero_grad()
        
        # core
        outputs = net(inputs)
        # lo que predice la red sobre el mini batch size que entrega dataloader
        _, preds = torch.max(outputs,1) 
        
        # funcion de perdida
        loss = loss_fn(outputs, targets)
        
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
        
        print('Loss {:.4f}, Acc: {:.4f} '.format(running_loss(), running_acc()))
        