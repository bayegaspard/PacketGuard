import torch
import torch.nn as nn
import torch.optim as optim


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
        print(step)
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

