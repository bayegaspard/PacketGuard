# defense.py

import torch
import torch.optim as optim
from config import DEVICE
from torch.nn import CrossEntropyLoss

# def ttpa_improved(model, x_adv, num_steps=800, learning_rate=0.001, alpha=0.9, confidence_threshold=0.7):
#     """
#     Improved TTOPA using a hybrid loss function and pseudo-labeling with confidence thresholding.
    
#     Parameters:
#         model (nn.Module): Trained classification model.
#         x_adv (torch.Tensor): Adversarial examples tensor.
#         num_steps (int): Number of adaptation steps.
#         learning_rate (float): Learning rate for optimizer.
#         alpha (float): Weighting factor between entropy and confidence losses.
#         confidence_threshold (float): Threshold for confidence in pseudo-labeling.
    
#     Returns:
#         torch.Tensor: Adapted adversarial examples.
#     """
#     model.eval()
#     x_adapted = x_adv.clone().detach().requires_grad_(True).to(DEVICE)

#     optimizer_ttap = optim.Adam([x_adapted], lr=learning_rate)

#     for step_num in range(num_steps):
#         optimizer_ttap.zero_grad()
#         outputs = model(x_adapted)
#         probabilities = torch.nn.functional.softmax(outputs, dim=1)

#         # Hybrid Loss: Entropy Minimization + Confidence Maximization
#         entropy_loss = -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))
#         confidence_loss = 1 - torch.mean(torch.max(probabilities, dim=1)[0])
#         hybrid_loss = alpha * entropy_loss + (1 - alpha) * confidence_loss

#         # Pseudo-Labeling with Confidence Thresholding
#         max_probs, pseudo_labels = torch.max(probabilities, dim=1)
#         mask = max_probs >= confidence_threshold

#         if mask.sum() > 0:
#             loss_supervised = CrossEntropyLoss()(outputs[mask], pseudo_labels[mask])
#             total_loss = hybrid_loss + loss_supervised
#         else:
#             total_loss = hybrid_loss

#         # Input Regularization (L2)
#         l2_reg = torch.norm(x_adapted - x_adv)
#         total_loss += 0.0001 * l2_reg  # Regularization coefficient

#         total_loss.backward()

#         # Gradient Clipping
#         torch.nn.utils.clip_grad_norm_([x_adapted], max_norm=1.0)

#         optimizer_ttap.step()

#         with torch.no_grad():
#             x_adapted.clamp_(min=0, max=1)

#             if torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
#                 print(f"NaN or Inf detected in x_adapted at step {step_num}")
#                 x_adapted[torch.isnan(x_adapted)] = 0
#                 x_adapted[torch.isinf(x_adapted)] = 0

#     return x_adapted.detach()
# defense.py

# defense.py

import torch
import torch.optim as optim
import torch.nn as nn

def ttpa_improved(model, x_adv, num_steps=400, learning_rate=0.001, alpha=0.9, confidence_threshold=0.7):
    """
    Apply TTOPA defense to adversarial examples.

    Args:
        model (torch.nn.Module): Trained classifier model.
        x_adv (torch.Tensor): Adversarial examples tensor.
        num_steps (int): Number of optimization steps.
        learning_rate (float): Learning rate for optimization.
        alpha (float): Scaling factor.
        confidence_threshold (float): Threshold for confidence.

    Returns:
        torch.Tensor: Defended adversarial examples.
    """
    # Ensure x_adv is on the same device as the model
    device = next(model.parameters()).device
    x_defended = nn.Parameter(x_adv.clone().detach().to(device), requires_grad=True)

    optimizer_ttap = optim.Adam([x_defended], lr=learning_rate)

    for step in range(num_steps):
        optimizer_ttap.zero_grad()
        outputs = model(x_defended)
        
        # Example loss: maximize the confidence of the predicted class
        predicted_class = torch.argmax(outputs, dim=1)
        loss = -torch.mean(outputs[range(len(predicted_class)), predicted_class])
        
        loss.backward()
        optimizer_ttap.step()

        # Optional: Project the defended samples back to valid range if necessary
        # For example, if input features are normalized between certain bounds
        # x_defended.data = torch.clamp(x_defended.data, min=lower_bound, max=upper_bound)

        if step % 50 == 0:
            print(f"TTOPA Step {step}/{num_steps} completed.")

    # Detach the defended tensor from the computation graph and return
    return x_defended.detach()
