#  auxiliar - para loggin del entrenamiento
class RunningMetric():
    """
    Class to calculate means
    """
    
    def __init__(self):
        self.sum = 0
        self.n = 0
        
    def update(self, val, size):
        self.sum += val
        self.n += size
        
    def __call__(self):
        return self.sum/float(self.n)