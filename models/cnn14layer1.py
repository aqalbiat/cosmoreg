import torch
import torch.nn as nn

class CNN14Steps_Model1(nn.Module):
    def __init__(self):
        super(CNN14Steps_Model1, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=5, padding=2),     
            nn.LeakyReLU(0.2),                              
            nn.MaxPool2d(4, 4),                           

            nn.Conv2d(32, 64, kernel_size=5, padding=2),   
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                             
            nn.MaxPool2d(4, 4),                       

            nn.Conv2d(128, 128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                          
            nn.AdaptiveAvgPool2d((2, 2)),                   

            nn.Flatten(),                                   
            nn.Linear(512, 32),                            
            nn.Linear(32, 2)                                    
                                           
        )

    def forward(self, x):
        return self.net(x)
    


class CNN14Steps_Model2(nn.Module):
    def __init__(self):
        super(CNN14Steps_Model2, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     
            nn.BatchNorm2d(16),  
            nn.LeakyReLU(0.2),                             
            nn.MaxPool2d(4, 4),                           
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),                          
            nn.AdaptiveAvgPool2d((2, 2)),                 

            #Decision 
            nn.Flatten(),                                   
            nn.Linear(64*2*2, 32),              
            nn.LeakyReLU(0.2),              
            nn.Linear(32, 2)                                
        )

    def forward(self, x):
        return self.net(x)
    

class CNN14Steps_Model3(nn.Module):
    def __init__(self):
        super(CNN14Steps_Model3, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=5, padding=2),     
            nn.LeakyReLU(0.2),                              
            nn.MaxPool2d(4, 4),                             

            nn.Conv2d(32, 64, kernel_size=5, padding=2),    
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    
            nn.Conv2d(128, 128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                                    
            nn.AdaptiveAvgPool2d((2, 2)),                  

            nn.Flatten(),                                  
            nn.Linear(512, 128), 
            nn.BatchNorm1d(128),                      
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 2)                                    
                                           
        )

    def forward(self, x):
        return self.net(x)
    


class CNN14Steps_Model4(nn.Module):
    def __init__(self):
        super(CNN14Steps_Model4, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=5, padding=2),     
            nn.LeakyReLU(0.2),                              
            nn.MaxPool2d(4, 4),                       

            nn.Conv2d(32, 64, kernel_size=5, padding=2),    
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                            
            nn.MaxPool2d(4, 4),                             

            nn.Conv2d(128, 128, kernel_size=3, padding=1),   
            nn.LeakyReLU(0.2),                             
            nn.AdaptiveAvgPool2d((2, 2)),                

            nn.Flatten(),                                  
            nn.Linear(512, 128),                   
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)                                    
                                           
        )

    def forward(self, x):
        return self.net(x)
    



class CNN14Steps_Model5(nn.Module):
    def __init__(self):
        super(CNN14Steps_Model5, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=7, padding=3),   
            nn.LeakyReLU(0.2),                             
            nn.MaxPool2d(4, 4),                    

            nn.Conv2d(32, 64, kernel_size=5, padding=2),   
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                          
            nn.MaxPool2d(4, 4),                         

            nn.Conv2d(128, 128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                           
            nn.AdaptiveAvgPool2d((2, 2)),               
            
            #Decision
            nn.Flatten(),                                  
            nn.Linear(512, 128),                     
            nn.Linear(128, 2)                                    
                                           
        )

    def forward(self, x):
        return self.net(x)
    


class CNN14Steps_Model6(nn.Module):
    def __init__(self):
        super(CNN14Steps_Model6, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=7, padding=3),   
            nn.LeakyReLU(0.2),                             
            nn.MaxPool2d(4, 4),                    

            nn.Conv2d(32, 64, kernel_size=5, padding=2),   
            nn.Conv2d(64, 128, kernel_size=5, padding=2),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                          
            nn.MaxPool2d(4, 4),                         

            nn.Conv2d(128, 128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),                           
            nn.AdaptiveAvgPool2d((2, 2)),               
            
            #Decision
            nn.Flatten(),                                  
            nn.Linear(512, 128),                     
            nn.Linear(128, 2)                                    
                                           
        )

    def forward(self, x):
        return self.net(x)