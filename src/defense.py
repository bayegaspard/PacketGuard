import torch
import torch.nn as nn
import torch.optim as optim

# Supervised TTOPA
def ttpa_improved(model, x_adv, y_true, num_steps=100, learning_rate=0.005, device='cpu'):
    """
    Test-Time Open Packet Adaptation (TTOPA) function for adversarial recovery.

    Parameters:
        model (nn.Module): Trained classification model.
        x_adv (torch.Tensor): Adversarial examples.
        y_true (torch.Tensor): True labels for the adversarial examples.
        num_steps (int): Number of adaptation steps.
        learning_rate (float): Learning rate for adaptation.
        device (str): Device to run the adaptation (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: Adapted examples after applying TTOPA.
        list: List of gradient norms per adaptation step.
        list: List of loss values per adaptation step.
    """
    # Ensure model is on the correct device
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Ensure y_true is a tensor and moved to the correct device
    if isinstance(y_true, list):
        y_true = torch.tensor(y_true, dtype=torch.long).to(device)
    else:
        y_true = y_true.to(device)

    # Clone, detach, and explicitly set requires_grad=True for x_adapted
    x_adapted = x_adv.clone().detach().to(device).requires_grad_(True)

    # Use an optimizer for adapting the adversarial examples
    optimizer = optim.Adam([x_adapted], lr=learning_rate)

    # Lists to store gradient norms and loss values for each step
    grad_norms = []
    losses = []

    criterion = nn.CrossEntropyLoss()

    for step in range(num_steps):
        # print(step)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_adapted)
        loss = criterion(outputs, y_true)

        # Backward pass
        loss.backward()
        grad_norm = x_adapted.grad.norm().item()  # Compute gradient norm
        grad_norms.append(grad_norm)
        losses.append(loss.item())

        # Gradient descent step
        optimizer.step()

        # Clamp the adapted examples to ensure valid input range
        with torch.no_grad():
            x_adapted.clamp_(min=0, max=1)

    return x_adapted.detach(), grad_norms, losses


# # 8. Define the Improved Unsupervised TTOPA Function
# def ttpa_improved_unsup(model, x_adv, num_steps=800, learning_rate=0.001, alpha=0.9, confidence_threshold=0.7, device='cpu'):
#     """
#     Improved TTOPA using a hybrid loss function and pseudo-labeling with confidence thresholding.
#     """
#     x_adapted = x_adv.clone().detach().requires_grad_(True).to(device)

#     optimizer_ttap = optim.Adam([x_adapted], lr=learning_rate)

#     for step_num in range(num_steps):
#         optimizer_ttap.zero_grad()
#         outputs = model(x_adapted)
#         probabilities = nn.functional.softmax(outputs, dim=1)

#         # Hybrid Loss: Entropy Minimization + Confidence Maximization
#         entropy_loss = -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))
#         confidence_loss = 1 - torch.mean(torch.max(probabilities, dim=1)[0])
#         hybrid_loss = alpha * entropy_loss + (1 - alpha) * confidence_loss

#         # Pseudo-Labeling with Confidence Thresholding
#         max_probs, pseudo_labels = torch.max(probabilities, dim=1)
#         mask = max_probs >= confidence_threshold

#         if mask.sum() > 0:
#             loss_supervised = criterion(outputs[mask], pseudo_labels[mask])
#             total_loss = hybrid_loss + loss_supervised
#         else:
#             total_loss = hybrid_loss



#         # Gradient Clipping
#         torch.nn.utils.clip_grad_norm_([x_adapted], max_norm=1.0)

#         optimizer_ttap.step()

#         with torch.no_grad():
#             x_adapted.clamp_(min=0, max=1)

#             if torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
#                 print(f"NaN or Inf detected in x_adapted at step {step_num}")
#                 x_adapted[torch.isnan(x_adapted)] = 0
#                 x_adapted[torch.isinf(x_adapted)] = 0

#         # Optional: Remove or adjust the logging frequency
#         # if step_num % 10 == 0:
#         #     print(f"Adaptation step {step_num}, total_loss: {total_loss.item()}")

#     return x_adapted.detach()


# 12. Define the Unsupervised TTOPA Function
def ttpa_unsupervised(model, x_adv, num_steps=30, learning_rate=0.005, device='cpu'):
    """
    Unsupervised Test Time Open Packet Adaptation (TTOPA) using an optimizer.
    """
    x_adapted = x_adv.clone().detach().requires_grad_(True).to(device)
    
    # Use an optimizer for x_adapted
    optimizer_ttap = optim.Adam([x_adapted], lr=learning_rate)
    
    # Lists to store gradient norms and losses
    grad_norms = []
    losses = []
    
    for step_num in range(num_steps):
        optimizer_ttap.zero_grad()
        outputs = model(x_adapted)
        probabilities = nn.functional.softmax(outputs, dim=1)

        # Input Regularization (L2)
        l2_reg = torch.norm(x_adapted - x_adv)
        total_loss += 0.0001 * l2_reg  # Regularization coefficient

        # Entropy Minimization Loss
        loss = -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))
        
        loss.backward()

        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([x_adapted], max_norm=1.0)
        
        grad_norm = x_adapted.grad.norm().item()
        grad_norms.append(grad_norm)
        losses.append(loss.item())
        
        optimizer_ttap.step()
        
        # Ensure x_adapted stays within valid range
        with torch.no_grad():
            x_adapted.clamp_(min=0, max=1)
        
    return x_adapted.detach(), grad_norms, losses