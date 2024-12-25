# models.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from config import DIFFUSION_PARAMS, DEVICE

# models.py

import numpy as np
import torch
import torch.nn.functional as F
from config import DIFFUSION_PARAMS, DEVICE

class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.device = DEVICE  # Add this line
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class VarMaxModel:
    """
    A model-like class that applies the classify_with_unknown logic
    for comparison with neural network models.
    """

    def __init__(self, model, top_two_threshold=0.5, varmax_threshold=0.1):
        self.model = model
        self.top_two_threshold = top_two_threshold
        self.varmax_threshold = varmax_threshold

    def forward(self, x):
        """
        Simulates the `forward` method of a PyTorch model.

        Parameters:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Logits (not predictions).
        """
        with torch.no_grad():
            logits = self.model(x)
        return logits

    def classify(self, x):
        """
        Classify inputs based on VarMax logic.

        Parameters:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Predicted labels.
        """
        logits = self.forward(x)
        softmax_outputs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(softmax_outputs, 2, dim=1)
        top_diff = (top_probs[:, 0] - top_probs[:, 1]).cpu().numpy()
        logits_np = logits.cpu().numpy()

        predictions = []
        for i in range(x.size(0)):
            if top_diff[i] > self.top_two_threshold:
                predictions.append(top_indices[i, 0].item())
            else:
                # Apply VarMax
                logit = logits_np[i]
                variance = np.var(np.abs(logit))
                if variance < self.varmax_threshold:
                    predictions.append(-1)  # Representing 'Unknown' as -1
                else:
                    predictions.append(top_indices[i, 0].item())
        return torch.tensor(predictions)

    def __call__(self, x):
        """
        Makes the VarMaxModel callable.

        Parameters:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Predicted labels.
        """
        return self.classify(x)

    def eval(self):
        """Dummy method for compatibility with PyTorch evaluation mode."""
        pass


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.noise_steps = noise_steps
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(DEVICE)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(DEVICE)
        
        # Define network layers
        self.fc1 = nn.Linear(input_dim + 1, 256).to(DEVICE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, input_dim).to(DEVICE)

    def noise_data(self, x, t):
        """
        Add noise to the data based on the diffusion process.
        """
        batch_size = x.shape[0]
        t = t.view(-1)
        beta_t = self.beta[t].view(-1, 1).to(DEVICE)
        alpha_t = self.alpha_hat[t].view(-1, 1).to(DEVICE)
        noise = torch.randn_like(x).to(DEVICE)
        x_noisy = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return x_noisy, noise

    def forward(self, x, t):
        """
        Forward pass for the diffusion model.
        """
        t_normalized = t.float() / self.noise_steps  # Normalize time step
        x_input = torch.cat([x, t_normalized.unsqueeze(1)], dim=1)  # Concatenate time step
        x = self.fc1(x_input)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def generate(self, x, t):
        """
        Generate adversarial examples using the reverse diffusion process.
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            # Ensure t is a scalar value or handle it correctly
            if isinstance(t, torch.Tensor):
                if t.numel() > 1:  # If t contains multiple elements
                    t = t[0].item()  # Use the first element as scalar
                else:
                    t = t.item()
            
            t_tensor = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
            x_noisy, _ = self.noise_data(x.to(DEVICE), t_tensor)
            predicted_noise = self(x_noisy, t_tensor)
            alpha_t = self.alpha[t_tensor].view(-1, 1).to(DEVICE)
            alpha_hat_t = self.alpha_hat[t_tensor].view(-1, 1).to(DEVICE)
            x_adv = (1 / torch.sqrt(alpha_t)) * (
                x_noisy - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            )
        return x_adv





class FGSM:
    def __init__(self, model, epsilon):
        self.model = model.to(DEVICE)  # Ensure model is on the correct device
        self.epsilon = epsilon

    def generate(self, x, y):
        # Ensure inputs are on the same device as the model
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Model evaluation mode
        self.model.eval()

        # Enable gradient computation
        x_adv = x.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = self.model(x_adv)
        loss = F.cross_entropy(outputs, y)

        # Backward pass to compute gradients
        self.model.zero_grad()
        loss.backward()

        # Compute the adversarial perturbation
        perturbation = self.epsilon * x_adv.grad.sign()

        # Create adversarial examples by applying the perturbation
        x_adv = x_adv + perturbation

        # Clip the adversarial examples to ensure they remain valid inputs
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

    

class GeneticAlgorithmAttack:
    def __init__(self, model, num_classes, population_size=20, mutation_rate=0.1, max_generations=10, fitness_function=None):
        self.model = model
        self.num_classes = num_classes
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.fitness_function = fitness_function

    def generate(self, x, y):
        self.model.eval()
        x_np = x.cpu().detach().numpy()
        y_np = y.cpu().detach().numpy()

        # Initialize population
        population = np.repeat(x_np, self.population_size // len(x_np), axis=0)
        perturbations = np.random.uniform(-0.1, 0.1, population.shape)
        population += perturbations
        population = np.clip(population, 0, 1)

        for generation in range(self.max_generations):
            subset_y_true = np.tile(y_np, len(population) // len(y_np) + 1)[:len(population)]

            fitness_scores = self._evaluate_fitness(population, subset_y_true)

            # Select top-performing individuals
            top_indices = np.argsort(fitness_scores)[-self.population_size:]
            top_individuals = population[top_indices]

            if len(top_individuals) == 0:
                print(f"Warning: No top individuals found in generation {generation}. Reinitializing population.")
                top_individuals = population  # Use entire population as fallback

            offspring = self._crossover(top_individuals)
            offspring = self._mutate(offspring)

            population = np.vstack((top_individuals, offspring))

        best_indices = np.argsort(fitness_scores)[-len(x_np):]
        best_adversarial_examples = population[best_indices]

        return torch.tensor(best_adversarial_examples, dtype=torch.float32).to(x.device)

    def _evaluate_fitness(self, population, y_true):
        assert len(population) == len(y_true), f"Population size ({len(population)}) must match y_true size ({len(y_true)})"

        population_tensor = torch.tensor(population, dtype=torch.float32).to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model(population_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        if self.fitness_function:
            return np.array([self.fitness_function(probs[i], y_true[i]) for i in range(len(y_true))])
        else:
            fitness_scores = np.max(probs, axis=1)
            fitness_scores[np.array(y_true) == np.argmax(probs, axis=1)] = 0  # Penalize correct classifications
            return fitness_scores

    def _crossover(self, parents):
        num_parents = parents.shape[0]

        if num_parents < 2:
            print("Warning: Not enough parents for crossover. Cloning existing parents.")
            return parents.copy()

        num_features = parents.shape[1]
        offspring = []

        for _ in range(self.population_size // 2):
            parent1_idx, parent2_idx = np.random.choice(num_parents, size=2, replace=False)
            parent1, parent2 = parents[parent1_idx], parents[parent2_idx]

            mask = np.random.rand(num_features) > 0.5
            child = np.where(mask, parent1, parent2)
            offspring.append(child)

        return np.array(offspring)

    def _mutate(self, offspring):
        mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
        mutations = np.random.uniform(-0.1, 0.1, offspring.shape)
        offspring[mutation_mask] += mutations[mutation_mask]
        return np.clip(offspring, 0, 1)


