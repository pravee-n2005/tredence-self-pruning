"""
Self-Pruning Neural Network — Tredence AI Engineering Case Study
================================================================
Name  : Praveen Adithya B 
Task    : Train a feed-forward network on CIFAR-10 that learns to prune
          its own weights during training via learnable sigmoid gates and
          an L1 sparsity regularisation term.

Run:
    python train.py

Requirements:
    pip install torch torchvision matplotlib
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear — a drop-in replacement for nn.Linear with learnable gates
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A linear layer where every weight has an associated learnable scalar gate.

    Forward pass:
        gates        = sigmoid(gate_scores)          # values in (0, 1)
        pruned_w     = weight * gates                # element-wise mask
        out          = x @ pruned_w.T + bias

    During training, an L1 penalty on the gate values drives many of them to
    exactly zero, effectively removing the corresponding weights from the
    computation graph (they contribute nothing to the output).

    Gradient flow:
        Both `weight` and `gate_scores` are nn.Parameter objects, so PyTorch's
        autograd tracks gradients through the element-wise product automatically.
        sigmoid is differentiable everywhere, guaranteeing smooth gradient flow
        back into gate_scores.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard learnable parameters ──────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # ── Gate scores: same shape as weight, initialised near 0.5 gate ──
        # Initialising gate_scores to 0 means sigmoid(0) = 0.5, so gates
        # start at a neutral "half-open" state and the optimiser is free to
        # push them toward 0 (pruned) or 1 (kept).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform initialisation for the weight (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Squash gate_scores into (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # 2. Element-wise multiplication — zero gate ≡ pruned weight
        pruned_weights = self.weight * gates

        # 3. Standard affine transform (equivalent to F.linear)
        return F.linear(x, pruned_weights, self.bias)

    def gate_values(self) -> torch.Tensor:
        """Return the current gate activations (detached, for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below `threshold`."""
        gates = self.gate_values()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Network definition
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A three-hidden-layer MLP for CIFAR-10 classification.
    All linear projections use PrunableLinear so every weight can be gated out.

    Architecture:
        Input (3072) → 512 → 256 → 128 → 10
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # flatten
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        """Iterator over all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Because gates = sigmoid(gate_scores) ∈ (0,1), the L1 norm is simply
        their sum.  Minimising this sum drives gate values toward zero, which
        is the pruning objective.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage of weights pruned across the whole network."""
        pruned = total = 0
        for layer in self.prunable_layers():
            g = layer.gate_values()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return 100.0 * pruned / total if total > 0 else 0.0

    def all_gate_values(self) -> torch.Tensor:
        """Concatenate all gate values — used for histogram analysis."""
        parts = [layer.gate_values().flatten() for layer in self.prunable_layers()]
        return torch.cat(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training & evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, lam, device):
    model.train()
    total_loss = cls_loss_sum = sp_loss_sum = correct = n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimiser.zero_grad()
        logits = model(imgs)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()
        loss     = cls_loss + lam * sp_loss

        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()

        # Accumulate metrics
        total_loss    += loss.item()  * imgs.size(0)
        cls_loss_sum  += cls_loss.item() * imgs.size(0)
        sp_loss_sum   += sp_loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n       += imgs.size(0)

    return {
        "loss"     : total_loss   / n,
        "cls_loss" : cls_loss_sum / n,
        "sp_loss"  : sp_loss_sum  / n,
        "acc"      : 100.0 * correct / n,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        n       += imgs.size(0)
    return 100.0 * correct / n


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(lam: float, epochs: int, device, train_loader, test_loader,
                   verbose: bool = True):
    """Train the self-pruning network for a given λ and return results."""
    model = SelfPruningNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, train_loader, optimiser, lam, device)
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        sparsity = model.global_sparsity()

        if test_acc > best_acc:
            best_acc = test_acc

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(
                f"  Lambda={lam:.0e} | Epoch {epoch:>3}/{epochs} | "
                f"Loss={metrics['loss']:.4f}  CE={metrics['cls_loss']:.4f}  "
                f"Sp={metrics['sp_loss']:.1f} | "
                f"TrainAcc={metrics['acc']:.1f}%  TestAcc={test_acc:.1f}%  "
                f"Sparsity={sparsity:.1f}%  [{elapsed:.0f}s]"
            )

    final_acc     = evaluate(model, test_loader, device)
    final_sparsity = model.global_sparsity()
    gate_vals      = model.all_gate_values().cpu()

    return {
        "lambda"   : lam,
        "test_acc" : final_acc,
        "sparsity" : final_sparsity,
        "gate_vals": gate_vals,
        "model"    : model,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_distributions(results, save_path="gate_distributions.png"):
    """
    One subplot per λ value showing the histogram of final gate values.
    A successful run shows a sharp spike at 0 (pruned) and a spread of
    surviving gates away from 0.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        gates = res["gate_vals"].numpy()
        ax.hist(gates, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_title(
            f"λ = {res['lambda']:.0e}\n"
            f"Acc = {res['test_acc']:.1f}%  |  Sparsity = {res['sparsity']:.1f}%",
            fontsize=11,
        )
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1,
                   label="prune threshold (0.01)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Distribution of Learned Gate Values after Training", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot saved → {save_path}]")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    EPOCHS     = 1           # Reduced to 1 for faster execution (Increase to 30-50 for higher accuracy)
    LAMBDAS    = [1e-5, 1e-4, 1e-3]   # low / medium / high sparsity pressure

    train_loader, test_loader = get_dataloaders(batch_size=256)

    results = []
    for lam in LAMBDAS:
        print(f"\n{'='*70}")
        print(f"  Starting experiment  Lambda = {lam:.0e}")
        print(f"{'='*70}")
        res = run_experiment(lam, EPOCHS, device, train_loader, test_loader)
        results.append(res)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("-"*55)
    for r in results:
        print(f"{r['lambda']:<12.0e} {r['test_acc']:>14.2f} {r['sparsity']:>14.2f}")
    print("="*55)

    # ── Gate distribution plot ───────────────────────────────────────────────
    plot_gate_distributions(results, save_path="gate_distributions.png")

    print("\nDone. Results saved.")


if __name__ == "__main__":
    main()
