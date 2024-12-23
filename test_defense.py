# test_defense.py

import torch
import torch.nn as nn
from defense import ttpa_improved
from model import Net  # Assuming your model class is defined in model.py

# Define your model architecture in model.py
# Example:
# class Net(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(Net, self).__init__()
#         self.fc = nn.Linear(input_size, num_classes)
#     
#     def forward(self, x):
#         return self.fc(x)

def test_ttpa_improved():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model and move it to the device
    input_size = 77  # As per your data_loader.py
    num_classes = 9  # As per your encoded classes
    model = Net(input_size=input_size, num_classes=num_classes).to(device)
    
    # Load model weights if necessary
    # model.load_state_dict(torch.load('path_to_model_weights.pth'))
    
    # Create a sample adversarial example tensor
    # Ensure it's on the same device as the model
    # Example: Create a random tensor with the correct input size
    x_adv = torch.randn(1, input_size).to(device)  # Batch size of 1 for testing
    
    try:
        # Apply TTOPA
        x_defended = ttpa_improved(
            model, 
            x_adv, 
            num_steps=10, 
            learning_rate=0.001
        )
        
        # Pass the defended adversarial example through the model
        outputs = model(x_defended)
        print("TTOPA successfully applied. Model output:", outputs)
        
    except Exception as e:
        print("Error during TTOPA application:", e)

if __name__ == "__main__":
    test_ttpa_improved()
