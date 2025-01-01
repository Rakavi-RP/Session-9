# 3. Updated Parameters class
class Params:
    def __init__(self):
        self.batch_size = 512 #incresed from 128
        self.name = "resnet50_enhanced"
        self.workers = 12   #int(multiprocessing.cpu_count() * 0.8)
        self.base_lr = 0.175  # This will be the max_lr
        self.num_epochs = 250
        self.weight_decay = 1e-4  # Slightly reduced weight decay
        
    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

params = Params()
print(params) 