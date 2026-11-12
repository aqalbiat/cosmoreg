import torch
import torch.nn as nn

class CNN12Steps_Model1(nn.Module):
    def __init__(self):
        super(CNN12Steps_Model1, self).__init__()
        
        self.net = nn.Sequential(
            # --- FEATURE EXTRACTION (Steps 1 to 9) ---
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(16),                            # Step 2: Activation
            nn.MaxPool2d(4, 4),                             # Step 3: Pool (shrinks to 64x64)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # Step 4: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 5: Activation
            nn.MaxPool2d(4, 4),                             # Step 6: Pool (shrinks to 16x16)

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # Step 7: Conv (16x16)
            nn.LeakyReLU(0.2),                              # Step 8: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 9: Pool (shrinks to 2x2)

            # --- DECISION / REGRESSION (Steps 10 to 12) ---
            nn.Flatten(),                                   # Step 10: Unroll (64 * 2 * 2 = 256 values)
            nn.Linear(256, 32),                             # Step 11: Linear summary layer
            nn.Linear(32, 2)                                # Step 12: Output (Omega_m, sigma_8)
        )

    def forward(self, x):
        return self.net(x)

# Quick check
if __name__ == "__main__":
    model = CNN12Steps_Model1()
    dummy_x = torch.randn(4, 1, 256, 256)
    out = model(dummy_x)
    print("Output shape:", out.shape) # Returns torch.Size([4, 2])p


class CNN12Steps_Model2(nn.Module):
    def __init__(self):
        super(CNN12Steps_Model2, self).__init__()
        
        self.net = nn.Sequential(
            
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     
            nn.Conv2d(16, 16, kernel_size=3, padding=1),    
            nn.LeakyReLU(0.2),                             
            nn.MaxPool2d(4, 4),      #64X64                     
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),                            
            nn.AdaptiveAvgPool2d((2, 2)),                 

            #Decision 
            nn.Flatten(),                                   
            nn.Linear(32*2*2, 32),              
            nn.LeakyReLU(0.2),              
            nn.Linear(32, 2)                                
        )

    def forward(self, x):
        return self.net(x)
    


class CNN12Steps_Model3(nn.Module):
    def __init__(self):
        super(CNN12Steps_Model3, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     
            nn.BatchNorm2d(16),  
            nn.LeakyReLU(0.2),                             
            nn.MaxPool2d(4, 4),      #64X64                     
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    
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
    

class CNN12Steps_Model4(nn.Module):
    def __init__(self):
        super(CNN12Steps_Model4, self).__init__()
        
        self.net = nn.Sequential(
            
            #Feature Extraction    
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4, 4),      #64X64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((2, 2)),

            #Decision 
            nn.Flatten(),      
            nn.Linear(64*2*2, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Added dropout for regularization                             
            nn.Linear(32, 2)                      
        )

    def forward(self, x):
        return self.net(x)
    

class CNN12Steps_Model5(nn.Module):
    def __init__(self):
        super(CNN12Steps_Model5, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 16, kernel_size=5, padding=2),     # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(16),                            # Step 2: Activation
            nn.MaxPool2d(4, 4),                             # Step 3: Pool (shrinks to 64x64)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=5, padding=2),    # Step 4: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 5: Activation
            nn.MaxPool2d(4, 4),                             # Step 6: Pool (shrinks to 16x16)

            # Block 3
            nn.Conv2d(32, 64, kernel_size=5, padding=2),    # Step 7: Conv (16x16)
            nn.LeakyReLU(0.2),                              # Step 8: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 9: Pool (shrinks to 2x2)

            # --- DECISION / REGRESSION (Steps 10 to 12) ---
            nn.Flatten(),                                   # Step 10: Unroll (64 * 2 * 2 = 256 values)
            nn.Linear(256, 32),                             # Step 11: Linear summary layer
            nn.Linear(32, 2)                                    
                                           
        )

    def forward(self, x):
        return self.net(x)
    

class CNN12Steps_Model6(nn.Module):
    def __init__(self):
        super(CNN12Steps_Model6, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=5, padding=2),     # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(32),                            # Step 2: Activation
            nn.MaxPool2d(4, 4),                             # Step 3: Pool (shrinks to 64x64)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),    # Step 4: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 5: Activation
            nn.MaxPool2d(4, 4),                             # Step 6: Pool (shrinks to 16x16)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=5, padding=2),    # Step 7: Conv (16x16)
            nn.LeakyReLU(0.2),                              # Step 8: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 9: Pool (shrinks to 2x2)

            # --- DECISION / REGRESSION (Steps 10 to 12) ---
            nn.Flatten(),                                   # Step 10: Unroll (64 * 2 * 2 = 256 values)
            nn.Linear(512, 32),                             # Step 11: Linear summary layer
            nn.Linear(32, 2)                                    
                                           
        )

    def forward(self, x):
        return self.net(x)
    

#     top_10_models = [
#     "model_16_8",
#     "model_16_7",
#     "model_14_1",
#     "model_16_4",
#     "model_16_2",
#     "model_14_5",
#     "model_14_2",
#     "model_16_5",
#     "model_16_3",
#     "model_16_6"
# ]


# "model_16_4",
#     "model_16_2",
#     "model_14_2",
#     "model_16_7",
#     "model_14_1"