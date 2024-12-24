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

import numpy as np
import torch
from torch.nn.functional import softmax

class VarMaxModel:
    """
    A model-like class that applies the classify_with_unknown logic
    for comparison with neural network models.
    """

    def __init__(self, model, top_two_threshold=0.5, varmax_threshold=0.1):
        """
        Initializes the VarMaxModel.
        
        Parameters:
            model (nn.Module): The base neural network model.
            top_two_threshold (float): Threshold for Top Two Difference.
            varmax_threshold (float): Threshold for VarMax variance.
        """
        self.model = model
        self.top_two_threshold = top_two_threshold
        self.varmax_threshold = varmax_threshold

    def forward(self, x):
        """
        Simulates the `forward` method of a PyTorch model, classifying
        inputs based on VarMax logic.

        Parameters:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Predicted labels.
        """
        with torch.no_grad():
            logits = self.model(x)
            softmax_outputs = softmax(logits, dim=1)
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


class FGSM:
    """
    Class to implement the Fast Gradient Sign Method (FGSM) for generating adversarial examples.
    """
    def __init__(self, model, epsilon):
        """
        Initializes the FGSM class.

        Parameters:
            model (torch.nn.Module): The trained model to attack.
            epsilon (float): The magnitude of the perturbation.
        """
        self.model = model
        self.epsilon = epsilon

    def generate(self, x, y):
        """
        Generates adversarial examples.

        Parameters:
            x (torch.Tensor): Original input samples.
            y (torch.Tensor): True labels corresponding to the input samples.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        # Ensure the model is in evaluation mode
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
    """
    Class to implement a Genetic Algorithm-based adversarial attack.
    """
    def __init__(self, model, num_classes, population_size=20, mutation_rate=0.1, max_generations=10, fitness_function=None):
        """
        Initializes the GeneticAlgorithmAttack class.

        Parameters:
            model (torch.nn.Module): The trained model to attack.
            num_classes (int): The number of output classes of the model.
            population_size (int): Number of adversarial examples in each generation.
            mutation_rate (float): Probability of mutation for each feature.
            max_generations (int): Maximum number of generations for the genetic algorithm.
            fitness_function (callable, optional): Custom fitness function. If None, misclassification confidence is used.
        """
        self.model = model
        self.num_classes = num_classes
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.fitness_function = fitness_function

    def generate(self, x, y):
        """
        Generates adversarial examples using a genetic algorithm.

        Parameters:
            x (torch.Tensor): Original input samples.
            y (torch.Tensor): True labels corresponding to the input samples.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Convert inputs to numpy for manipulation
        x_np = x.cpu().detach().numpy()
        y_np = y.cpu().detach().numpy()

        # Initialize population
        population = np.repeat(x_np, self.population_size, axis=0)

        # Apply random perturbations to create the initial population
        perturbations = np.random.uniform(-0.1, 0.1, population.shape)
        population += perturbations
        population = np.clip(population, 0, 1)  # Ensure inputs are valid

        for generation in range(self.max_generations):
            # Evaluate fitness of each individual in the population
            fitness_scores = self._evaluate_fitness(population, y_np)

            # Select the top individuals based on fitness scores
            top_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
            top_individuals = population[top_indices]

            # Perform crossover to create new offspring
            offspring = self._crossover(top_individuals)

            # Apply mutation to introduce diversity
            offspring = self._mutate(offspring)

            # Combine parents and offspring to form the new population
            population = np.vstack((top_individuals, offspring))

        # Return the best adversarial examples based on fitness
        best_indices = np.argsort(fitness_scores)[-x_np.shape[0]:]
        best_adversarial_examples = population[best_indices]

        return torch.tensor(best_adversarial_examples, dtype=torch.float32).to(x.device)

    def _evaluate_fitness(self, population, y_true):
        """
        Evaluates the fitness of each individual in the population.

        Parameters:
            population (numpy.ndarray): Population of adversarial examples.
            y_true (numpy.ndarray): True labels for the original inputs.

        Returns:
            numpy.ndarray: Fitness scores for each individual.
        """
        population_tensor = torch.tensor(population, dtype=torch.float32).to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model(population_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        if self.fitness_function:
            # Use custom fitness function
            return np.array([self.fitness_function(probs[i], y_true[i]) for i in range(len(y_true))])
        else:
            # Default: maximize the misclassification confidence
            fitness_scores = np.max(probs, axis=1)
            fitness_scores[y_true == np.argmax(probs, axis=1)] = 0  # Penalize correct classifications
            return fitness_scores

    def _crossover(self, parents):
        """
        Performs crossover on the parent population to create offspring.

        Parameters:
            parents (numpy.ndarray): Array of parent individuals.

        Returns:
            numpy.ndarray: Array of offspring.
        """
        num_parents = parents.shape[0]
        num_features = parents.shape[1]
        offspring = []

        for _ in range(self.population_size // 2):
            # Select two random parents
            parent1_idx, parent2_idx = np.random.choice(num_parents, size=2, replace=False)
            parent1, parent2 = parents[parent1_idx], parents[parent2_idx]

            # Perform uniform crossover
            mask = np.random.rand(num_features) > 0.5
            child = np.where(mask, parent1, parent2)
            offspring.append(child)

        return np.array(offspring)

    def _mutate(self, offspring):
        """
        Applies mutation to the offspring population.

        Parameters:
            offspring (numpy.ndarray): Array of offspring individuals.

        Returns:
            numpy.ndarray: Mutated offspring population.
        """
        mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
        mutations = np.random.uniform(-0.1, 0.1, offspring.shape)
        offspring[mutation_mask] += mutations[mutation_mask]
        return np.clip(offspring, 0, 1)  # Ensure inputs are valid

