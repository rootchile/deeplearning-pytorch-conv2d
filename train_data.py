import os

from PIL import Image
# from plot_helpers import imshow
from torch.utils.data import Dataset, DataLoader

class SIGNSData (Dataset):
    
    def __init__ (self, base_dir, split='train', transform=None):
        # base_dir: ubicacion imagenes
        # split: train, test, valid
        # transform: pipeline de preprocesamiento
        
        path = os.path.join(base_dir,"{}_signs".format(split))
        files = os.listdir(path)
        
        self.filenames = [ os.path.join(path, f) for f in files if f.endswith('.jpg')]
        
        # primer caracter del archivo es el label
        self.targets = [ int(f[0]) for f in files ] #labels: 0,1,2,3,4,5
        self.transform = transform
        
    def __len__ (self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]) # path de la imagen en idx
        
        # si hay un pipeline de preprocesamiento, lo aplicamos
        if self.transform:
            image = self.transform(image)
            
        # imagen, label
        return image, self.targets[idx]