# models.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from config import DIFFUSION_PARAMS, DEVICE

# models.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from config import DIFFUSION_PARAMS, DEVICE

class Net(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128], activation='relu'):
        super(Net, self).__init__()
        self.hidden_layers = nn.ModuleList()
        previous_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(previous_size, hidden_size))
            if activation.lower() == 'relu':
                self.hidden_layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                self.hidden_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            self.hidden_layers.append(nn.Dropout(0.5))  # Added dropout
            previous_size = hidden_size
        self.output_layer = nn.Linear(previous_size, num_classes)
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)
        return out

    def predict_proba(self, x):
        """
        Returns the probability estimates for each class.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Probability scores for each class.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, noise_steps=DIFFUSION_PARAMS["noise_steps"], 
                 beta_start=DIFFUSION_PARAMS["beta_start"], beta_end=DIFFUSION_PARAMS["beta_end"]):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.noise_steps = noise_steps
        self.beta = self.prepare_noise_schedule(beta_start, beta_end).to(DEVICE)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(DEVICE)
    
        # Neural network layers for the diffusion model
        self.fc1 = nn.Linear(input_dim + 1, 256).to(DEVICE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, input_dim).to(DEVICE)
    
    def prepare_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps).to(DEVICE)
    
    def noise_data(self, x, t):
        batch_size = x.shape[0]
        t = t.view(-1)
        beta_t = self.beta[t].view(-1, 1).to(DEVICE)
        alpha_t = self.alpha_hat[t].view(-1, 1).to(DEVICE)
        noise = torch.randn_like(x).to(DEVICE)
        x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise
    
    def forward(self, x, t):
        # Append normalized time step t as a scalar to the input
        t_normalized = t.float() / self.noise_steps  # Normalize and convert to float
        x_input = torch.cat([x, t_normalized.unsqueeze(1)], dim=1)  # Shape: (batch_size, input_dim + 1)
        x = self.fc1(x_input)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def generate_adversarial(self, x, t):
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            t_tensor = torch.tensor([t]*batch_size, device=DEVICE).long()
            x_noisy, noise = self.noise_data(x.to(DEVICE), t_tensor)
            predicted_noise = self(x_noisy, t_tensor)
            alpha_t = self.alpha[t_tensor].view(-1, 1).to(DEVICE)
            alpha_hat_t = self.alpha_hat[t_tensor].view(-1, 1).to(DEVICE)
            x_adv = (1 / torch.sqrt(alpha_t)) * (
                x_noisy - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            )
        self.train()
        return x_adv
