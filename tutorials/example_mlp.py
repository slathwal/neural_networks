import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----- MODEL -----
class MiniMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ----- HOOKS -----
activations = {}
gradients = {}

def activation_hook(name):
    def hook(module, input, output):
        print(f"[Forward] {name} → shape: {output.shape}")
        activations[name] = output.detach().cpu()
    return hook

def gradient_hook(name):
    def hook(grad):
        print(f"[Backward] {name} → grad shape: {grad.shape}")
        gradients[name] = grad.detach().cpu()
    return hook


# ----- TRAINING -----
def train(model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        x = torch.randn(16, 20)       # input: [B, D]
        y = torch.randn(16, 1)        # target: [B, 1]

        optimizer.zero_grad()
        out = model(x)

        out.register_hook(gradient_hook("output"))  # backward hook on final output

        loss = loss_fn(out, y)
        print(f"\nEpoch {epoch+1}, Loss = {loss.item():.4f}")
        loss.backward()
        optimizer.step()


# ----- PLOTTING -----
def plot_histogram(tensor, title):
    tensor = tensor.view(-1).numpy()
    plt.hist(tensor, bins=40, alpha=0.7)
    plt.title(title)
    plt.grid(True)
    plt.show()


# ----- MAIN -----
if __name__ == "__main__":
    model = MiniMLP()

    # Register forward hooks
    model.fc1.register_forward_hook(activation_hook("fc1"))
    model.fc2.register_forward_hook(activation_hook("fc2"))

    train(model, epochs=3)

    # Visualize results
    for name, act in activations.items():
        print(f"\nPlotting activation from {name}")
        plot_histogram(act, f"Activation Histogram - {name}")

    for name, grad in gradients.items():
        print(f"\nPlotting gradient from {name}")
        plot_histogram(grad, f"Gradient Histogram - {name}")
