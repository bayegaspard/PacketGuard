import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from config import DEVICE

def ttpa_improved(model, x_adv, num_steps=800, learning_rate=0.001, alpha=0.9, confidence_threshold=0.7):
    """
    Improved TTOPA using a hybrid loss function and pseudo-labeling with confidence thresholding.
    
    Parameters:
        model (nn.Module): Trained classification model.
        x_adv (torch.Tensor): Adversarial examples tensor.
        num_steps (int): Number of adaptation steps.
        learning_rate (float): Learning rate for optimizer.
        alpha (float): Weighting factor between entropy and confidence losses.
        confidence_threshold (float): Threshold for confidence in pseudo-labeling.
    
    Returns:
        torch.Tensor: Adapted adversarial examples.
    """
    model.eval()
    x_adapted = x_adv.clone().detach().requires_grad_(True).to(DEVICE)

    optimizer_ttap = torch.optim.Adam([x_adapted], lr=learning_rate)

    for step_num in range(num_steps):
        optimizer_ttap.zero_grad()
        outputs = model(x_adapted)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Hybrid Loss: Entropy Minimization + Confidence Maximization
        entropy_loss = -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))
        confidence_loss = 1 - torch.mean(torch.max(probabilities, dim=1)[0])
        hybrid_loss = alpha * entropy_loss + (1 - alpha) * confidence_loss

        # Input Regularization (L2)
        l2_reg = torch.norm(x_adapted - x_adv)
        total_loss = hybrid_loss + 0.0001 * l2_reg  # Regularization coefficient

        # Perform backward pass with `retain_graph=True`
        total_loss.backward(retain_graph=True)

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_([x_adapted], max_norm=1.0)

        optimizer_ttap.step()

        # Recreate x_adapted with clamping to avoid in-place operation
        with torch.no_grad():
            x_adapted = torch.clamp(x_adapted, 0, 1).detach().requires_grad_(True)

    return x_adapted.detach()
