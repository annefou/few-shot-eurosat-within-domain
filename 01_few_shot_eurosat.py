# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Few-Shot Classification on EuroSAT Sentinel-2 Imagery
#
# ## Replication context
#
# **Original paper**: Snell, Swersky & Zemel (2017),
# [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175), NeurIPS.
# Tested on mini-ImageNet (natural photos). Accuracy: 49.4% (1-shot, 5-way).
#
# **Cross-domain benchmark**: Guo et al. (2020),
# [A Broader Study of Cross-Domain Few-Shot Learning](https://doi.org/10.1007/978-3-030-58583-9_8), ECCV.
# Tests transfer from mini-ImageNet to EuroSAT (satellite imagery) among other domains.
#
# **Our question**: How well does Prototypical Networks perform on EuroSAT
# when we simulate a rare-class scenario — common land cover types as
# base classes, rare habitats as novel classes with only a few examples?
#
# ## EuroSAT dataset
#
# | Property | Value |
# |----------|-------|
# | Source | Sentinel-2 Level-1C |
# | Bands | RGB (this experiment) |
# | Images | 27,000 (64×64 px, 10m GSD) |
# | Classes | 10 land use / land cover types |
# | Size | ~94 MB |

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

# %% [markdown]
# ## Configuration

# %%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Few-shot settings
N_WAY = 5       # number of classes per episode
K_SHOT = 5      # support examples per class
N_QUERY = 15    # query examples per class

# CI mode: fewer episodes for faster runs on CPU (set CI=true in env)
CI_MODE = os.environ.get("CI", "").lower() in ("true", "1")
N_EPISODES = 100 if CI_MODE else 600      # evaluation episodes
N_TRAIN_EPISODES_CFG = 500 if CI_MODE else 2000  # training episodes

if CI_MODE:
    print("CI mode: reduced episodes for faster execution")

# %% [markdown]
# ## 1. Load EuroSAT and inspect

# %%
transform = transforms.Compose([
    transforms.Resize((84, 84)),  # standard few-shot size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = EuroSAT(root=str(DATA_DIR), download=False, transform=transform)

print(f"Total images: {len(dataset)}")
print(f"Classes ({len(dataset.classes)}):")
for i, cls in enumerate(dataset.classes):
    count = sum(1 for _, label in dataset if label == i)
    print(f"  {i}: {cls} ({count})")

# %% [markdown]
# ## 2. Define base / novel class split
#
# We simulate the Natura 2000 scenario: common habitat types are "base"
# (abundant training data), rare types are "novel" (few-shot).
#
# **Base classes** (7): AnnualCrop, Forest, Highway, Industrial, Pasture,
# Residential, SeaLake — common, well-represented land cover.
#
# **Novel classes** (3): HerbaceousVegetation, PermanentCrop, River —
# treated as "rare" habitats with only K examples available.

# %%
CLASS_NAMES = dataset.classes

BASE_CLASSES = [0, 1, 3, 4, 5, 7, 9]  # AnnualCrop, Forest, Highway, Industrial, Pasture, Residential, SeaLake
NOVEL_CLASSES = [2, 6, 8]  # HerbaceousVegetation, PermanentCrop, River

print("Base classes (training):")
for c in BASE_CLASSES:
    print(f"  {CLASS_NAMES[c]}")
print("\nNovel classes (few-shot):")
for c in NOVEL_CLASSES:
    print(f"  {CLASS_NAMES[c]}")

# Build class-to-indices mapping
class_indices = defaultdict(list)
for idx in range(len(dataset)):
    _, label = dataset[idx]
    class_indices[label].append(idx)

for c in sorted(class_indices.keys()):
    print(f"  {CLASS_NAMES[c]}: {len(class_indices[c])} images")

# %% [markdown]
# ## 3. Embedding network
#
# Following ProtoNet (Snell et al.), we use a simple 4-block CNN
# (Conv-BN-ReLU-MaxPool) as the embedding backbone. This is the
# standard architecture used in the original paper.

# %%
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ProtoNetCNN(nn.Module):
    """4-block CNN embedding network (ProtoNet standard)."""

    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # flatten


model = ProtoNetCNN().to(DEVICE)
embed_dim = model(torch.randn(1, 3, 84, 84).to(DEVICE)).shape[1]
print(f"Embedding dimension: {embed_dim}")

# %% [markdown]
# ## 4. Episode sampling
#
# Each few-shot "episode" samples N_WAY classes, then K_SHOT support
# images and N_QUERY query images per class.

# %%
def sample_episode(class_indices, classes, n_way, k_shot, n_query, dataset):
    """Sample a few-shot episode."""
    selected = random.sample(classes, n_way)

    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for i, cls in enumerate(selected):
        indices = random.sample(class_indices[cls], k_shot + n_query)
        support_idx = indices[:k_shot]
        query_idx = indices[k_shot:]

        for idx in support_idx:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(i)

        for idx in query_idx:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(i)

    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels


# Test episode
s_img, s_lbl, q_img, q_lbl = sample_episode(
    class_indices, BASE_CLASSES, N_WAY, K_SHOT, N_QUERY, dataset
)
print(f"Support: {s_img.shape}, Query: {q_img.shape}")

# %% [markdown]
# ## 5. Train on base classes
#
# Train the embedding network using episodic training on base classes.
# Each training step is a few-shot episode: compute prototypes from
# support set, classify queries by nearest prototype.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
N_TRAIN_EPISODES = N_TRAIN_EPISODES_CFG

print(f"Training on {len(BASE_CLASSES)} base classes for {N_TRAIN_EPISODES} episodes...")
model.train()
losses = []

for ep in range(N_TRAIN_EPISODES):
    s_img, s_lbl, q_img, q_lbl = sample_episode(
        class_indices, BASE_CLASSES, N_WAY, K_SHOT, N_QUERY, dataset
    )
    s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
    q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)

    # Compute embeddings
    s_emb = model(s_img)
    q_emb = model(q_img)

    # Compute prototypes (mean embedding per class)
    prototypes = torch.stack([
        s_emb[s_lbl == c].mean(dim=0) for c in range(N_WAY)
    ])

    # Classify queries by distance to prototypes
    dists = torch.cdist(q_emb, prototypes)  # (n_query*n_way, n_way)
    log_probs = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_probs, q_lbl)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (ep + 1) % 500 == 0:
        avg_loss = np.mean(losses[-500:])
        acc = (log_probs.argmax(dim=1) == q_lbl).float().mean().item()
        print(f"  Episode {ep+1}/{N_TRAIN_EPISODES}: loss={avg_loss:.3f}, acc={acc:.1%}")

print("Training complete.")

# %% [markdown]
# ## 6. Evaluate on novel classes (few-shot)
#
# The key test: can the model trained on base classes generalize to
# novel classes (HerbaceousVegetation, PermanentCrop, River) with
# only K examples?

# %%
model.eval()
accuracies = []

# Use all 10 classes for evaluation episodes (5-way from all classes)
# This tests generalization to novel classes
ALL_CLASSES = list(range(10))

print(f"Evaluating: {N_EPISODES} episodes, {N_WAY}-way {K_SHOT}-shot")
print(f"  (episodes can include both base and novel classes)\n")

with torch.no_grad():
    for ep in range(N_EPISODES):
        s_img, s_lbl, q_img, q_lbl = sample_episode(
            class_indices, ALL_CLASSES, N_WAY, K_SHOT, N_QUERY, dataset
        )
        s_img, q_img = s_img.to(DEVICE), q_img.to(DEVICE)
        q_lbl = q_lbl.to(DEVICE)

        s_emb = model(s_img)
        q_emb = model(q_img)

        prototypes = torch.stack([
            s_emb[s_lbl.to(DEVICE) == c].mean(dim=0) for c in range(N_WAY)
        ])

        dists = torch.cdist(q_emb, prototypes)
        preds = (-dists).argmax(dim=1)
        acc = (preds == q_lbl).float().mean().item()
        accuracies.append(acc)

mean_acc = np.mean(accuracies)
ci95 = 1.96 * np.std(accuracies) / np.sqrt(N_EPISODES)

print(f"Results ({N_WAY}-way {K_SHOT}-shot, {N_EPISODES} episodes):")
print(f"  Accuracy: {mean_acc:.1%} ± {ci95:.1%}")

# %% [markdown]
# ## 7. Evaluate novel-only episodes
#
# More stringent test: episodes drawn only from the 3 novel classes
# (3-way classification, since we only have 3 novel classes).

# %%
novel_accs = []
n_novel_way = min(N_WAY, len(NOVEL_CLASSES))

with torch.no_grad():
    for ep in range(N_EPISODES):
        s_img, s_lbl, q_img, q_lbl = sample_episode(
            class_indices, NOVEL_CLASSES, n_novel_way, K_SHOT, N_QUERY, dataset
        )
        s_img, q_img = s_img.to(DEVICE), q_img.to(DEVICE)
        q_lbl = q_lbl.to(DEVICE)

        s_emb = model(s_img)
        q_emb = model(q_img)

        prototypes = torch.stack([
            s_emb[s_lbl.to(DEVICE) == c].mean(dim=0) for c in range(n_novel_way)
        ])

        dists = torch.cdist(q_emb, prototypes)
        preds = (-dists).argmax(dim=1)
        acc = (preds == q_lbl).float().mean().item()
        novel_accs.append(acc)

novel_mean = np.mean(novel_accs)
novel_ci = 1.96 * np.std(novel_accs) / np.sqrt(N_EPISODES)

print(f"Novel-only results ({n_novel_way}-way {K_SHOT}-shot):")
print(f"  Accuracy: {novel_mean:.1%} ± {novel_ci:.1%}")

# %% [markdown]
# ## 8. Compare shot counts (1, 5, 20)

# %%
shot_results = {}
for k in [1, 5, 20]:
    accs = []
    with torch.no_grad():
        for ep in range(N_EPISODES):
            s_img, s_lbl, q_img, q_lbl = sample_episode(
                class_indices, ALL_CLASSES, N_WAY, k, N_QUERY, dataset
            )
            s_img, q_img = s_img.to(DEVICE), q_img.to(DEVICE)
            q_lbl = q_lbl.to(DEVICE)

            s_emb = model(s_img)
            q_emb = model(q_img)

            prototypes = torch.stack([
                s_emb[s_lbl.to(DEVICE) == c].mean(dim=0) for c in range(N_WAY)
            ])

            dists = torch.cdist(q_emb, prototypes)
            preds = (-dists).argmax(dim=1)
            acc = (preds == q_lbl).float().mean().item()
            accs.append(acc)

    mean = np.mean(accs)
    ci = 1.96 * np.std(accs) / np.sqrt(N_EPISODES)
    shot_results[k] = (mean, ci)
    print(f"  {N_WAY}-way {k}-shot: {mean:.1%} ± {ci:.1%}")

# %% [markdown]
# ## 9. Results summary and plot

# %%
import json

results = {
    "method": "Prototypical Networks (Snell et al. 2017)",
    "original_paper_doi": "10.48550/arXiv.1703.05175",
    "benchmark_paper_doi": "10.1007/978-3-030-58583-9_8",
    "dataset": "EuroSAT (Sentinel-2 RGB, 27k images, 10 classes)",
    "device": DEVICE,
    "n_episodes": N_EPISODES,
    "base_classes": [CLASS_NAMES[c] for c in BASE_CLASSES],
    "novel_classes": [CLASS_NAMES[c] for c in NOVEL_CLASSES],
    "results": {
        f"{N_WAY}way_{k}shot": {"accuracy": f"{m:.3f}", "ci95": f"{c:.3f}"}
        for k, (m, c) in shot_results.items()
    },
    "novel_only": {
        f"{n_novel_way}way_{K_SHOT}shot": {
            "accuracy": f"{novel_mean:.3f}",
            "ci95": f"{novel_ci:.3f}",
        }
    },
}

with open(RESULTS_DIR / "few_shot_eurosat_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {RESULTS_DIR / 'few_shot_eurosat_results.json'}")

# %%
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: accuracy vs shot count
    ax = axes[0]
    shots = sorted(shot_results.keys())
    means = [shot_results[k][0] for k in shots]
    cis = [shot_results[k][1] for k in shots]
    ax.errorbar(shots, means, yerr=cis, fmt="o-", capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Number of shots (K)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ProtoNet {N_WAY}-way on EuroSAT")
    ax.set_xticks(shots)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Panel B: training loss curve
    ax = axes[1]
    window = 50
    smoothed = [np.mean(losses[max(0, i - window):i + 1]) for i in range(len(losses))]
    ax.plot(smoothed, linewidth=1)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (base classes)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Prototypical Networks on EuroSAT Sentinel-2",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "few_shot_eurosat.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {RESULTS_DIR / 'few_shot_eurosat.png'}")
except ImportError:
    print("matplotlib not available")

# Save model
torch.save(model.state_dict(), RESULTS_DIR / "protonet_eurosat.pth")
print(f"Model saved to {RESULTS_DIR / 'protonet_eurosat.pth'}")
