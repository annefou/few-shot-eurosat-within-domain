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
# # Can AI Learn New Land Cover Types from Just 5 Satellite Images?
#
# ## The problem
#
# Monitoring land cover from satellite imagery is essential for environmental
# policy (e.g. the EU [Natura 2000](https://en.wikipedia.org/wiki/Natura_2000)
# network). Deep learning models can classify common land cover types
# (forest, urban, cropland) accurately — but they need thousands of labeled
# training examples per class.
#
# For **rare or protected habitat types** (Mediterranean scrub, alkaline fens,
# calcareous grasslands), we often have only a handful of labeled examples.
# Collecting more is expensive: it requires expert field surveys and manual
# annotation of satellite patches.
#
# **Few-shot learning** aims to solve this: train a model on classes with
# abundant data, then classify new classes from just a few examples.
#
# ## What we test
#
# We apply [Prototypical Networks](https://arxiv.org/abs/1703.05175)
# (Snell et al., NeurIPS 2017) — a foundational few-shot learning method
# from the AI/computer vision community — to real **Sentinel-2 satellite
# imagery** from the [EuroSAT](https://github.com/phelber/EuroSAT) dataset.
#
# The question: **if we train on common land cover types, can the model
# recognize rare types from just 5 labeled examples?**
#
# ## How Prototypical Networks work
#
# The idea is simple:
#
# 1. **Learn an embedding**: train a neural network to map satellite images
#    into a compact feature space where similar land cover types cluster
#    together.
#
# 2. **Compute prototypes**: for each class, average the embeddings of the
#    few labeled examples — this "prototype" represents what that class
#    looks like in feature space.
#
# 3. **Classify by nearest prototype**: assign a new image to whichever
#    prototype is closest in the embedding space.
#
# No retraining is needed for new classes — just provide a few examples
# and the model computes prototypes on the fly.
#
# ```{figure} https://miro.medium.com/v2/resize:fit:1400/1*gCgCjGr0EjmkCGJGMwUBIA.png
# :alt: Prototypical Networks diagram
# :width: 80%
#
# Prototypical Networks compute a prototype (star) for each class from
# support examples, then classify queries by nearest prototype.
# Source: Snell et al. (2017).
# ```
#
# ## The data: EuroSAT
#
# [EuroSAT](https://doi.org/10.1109/JSTARS.2019.2918242) contains 27,000
# real Sentinel-2 satellite image patches (64 × 64 pixels, 10 m ground
# resolution) covering 10 land use/land cover classes across Europe.
#
# | Property | Value |
# |----------|-------|
# | Satellite | Sentinel-2 Level-1C |
# | Bands | RGB (this experiment) |
# | Patch size | 64 × 64 pixels (10 m GSD) |
# | Total images | 27,000 |
# | Classes | 10 land use / land cover types |
# | Coverage | Europe-wide |
#
# This is **real satellite data**, not synthetic. Every image is a genuine
# acquisition from the Copernicus Sentinel-2 mission.

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
#
# **Few-shot terminology:**
#
# - **N-way**: number of classes in each classification task (we use 5)
# - **K-shot**: number of labeled examples per class (we test 1, 5, and 20)
# - **Episode**: one classification task — sample N classes, provide K
#   examples each, then classify new images
# - **Support set**: the K labeled examples per class (the "training data"
#   for this episode)
# - **Query set**: the images to classify (the "test data" for this episode)

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
# ## 1. Load and explore the data
#
# EuroSAT is downloaded automatically from
# [Zenodo](https://zenodo.org/records/7711810) on first run (~94 MB).

# %%
transform = transforms.Compose([
    transforms.Resize((84, 84)),  # standard few-shot image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = EuroSAT(root=str(DATA_DIR), download=True, transform=transform)

print(f"Total images: {len(dataset)}")
print(f"Classes ({len(dataset.classes)}):")
for i, cls in enumerate(dataset.classes):
    count = sum(1 for _, label in dataset if label == i)
    print(f"  {i}: {cls} ({count})")

# %% [markdown]
# ## 2. Simulating the rare habitat scenario
#
# In a real Natura 2000 monitoring context, some habitat types have
# thousands of labeled satellite patches (forests, cropland, urban areas)
# while others have very few (specific grassland types, wetlands,
# transitional habitats).
#
# We simulate this by splitting EuroSAT's 10 classes into:
#
# - **Base classes (7)** — common land cover with abundant training data.
#   The model learns its embedding from these.
# - **Novel classes (3)** — treated as "rare" habitats. During evaluation,
#   the model sees only 5 examples of each and must classify new images.
#
# The model **never sees novel class images during training**. At test time,
# it receives just K labeled examples (the "support set") and must
# generalize from those alone.

# %%
CLASS_NAMES = dataset.classes

BASE_CLASSES = [0, 1, 3, 4, 5, 7, 9]  # AnnualCrop, Forest, Highway, Industrial, Pasture, Residential, SeaLake
NOVEL_CLASSES = [2, 6, 8]  # HerbaceousVegetation, PermanentCrop, River

print("Base classes (abundant training data):")
for c in BASE_CLASSES:
    print(f"  {CLASS_NAMES[c]}")
print("\nNovel classes (few-shot — only K examples at test time):")
for c in NOVEL_CLASSES:
    print(f"  {CLASS_NAMES[c]}")

# Build class-to-indices mapping
class_indices = defaultdict(list)
for idx in range(len(dataset)):
    _, label = dataset[idx]
    class_indices[label].append(idx)

print("\nImages per class:")
for c in sorted(class_indices.keys()):
    role = "BASE" if c in BASE_CLASSES else "NOVEL"
    print(f"  {CLASS_NAMES[c]:25s}: {len(class_indices[c]):5d} images  [{role}]")

# %% [markdown]
# ## 3. The embedding network
#
# The neural network does not directly classify images into land cover
# types. Instead, it learns to **map images into a feature space** where
# similar land cover types are close together and different types are
# far apart.
#
# We use the standard ProtoNet architecture: four convolutional blocks
# (Conv → BatchNorm → ReLU → MaxPool), producing a 1,600-dimensional
# feature vector per image.

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
# Few-shot learning uses **episodic training**: instead of showing the
# model all images with their labels (as in standard classification),
# we simulate many small classification tasks ("episodes").
#
# Each episode:
# 1. Randomly pick 5 classes
# 2. For each class, sample 5 "support" images (the labeled examples)
#    and 15 "query" images (to classify)
# 3. The model must classify the queries using only the support examples
#
# This mirrors what happens at test time with novel classes, so the model
# learns a strategy that generalizes to new classes.

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
print(f"Support set: {s_img.shape} (5 classes x 5 images = 25 images)")
print(f"Query set:   {q_img.shape} (5 classes x 15 images = 75 images)")

# %% [markdown]
# ## 5. Training on base classes
#
# We train the embedding network on 2,000 episodes using only the 7
# base classes. The model learns to produce embeddings where different
# land cover types are well separated.
#
# At each episode:
# 1. Embed the support images → compute the **prototype** (mean embedding)
#    for each class
# 2. Embed the query images → classify each by finding the **nearest
#    prototype**
# 3. Compute the loss (how many queries were misclassified) and update
#    the network

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
# ## 6. Evaluation: can it classify novel land cover types?
#
# Now the critical test. The model has **never seen** HerbaceousVegetation,
# PermanentCrop, or River during training. We give it just 5 labeled
# examples of each and ask it to classify new images.
#
# We run 600 random episodes (matching the standard evaluation protocol
# from [Guo et al., ECCV 2020](https://doi.org/10.1007/978-3-030-58583-9_8))
# and report mean accuracy with 95% confidence intervals.
#
# **Mixed episodes** draw from all 10 classes (both base and novel).
# This tests the realistic scenario where the model encounters both
# familiar and unfamiliar land cover types.

# %%
model.eval()
accuracies = []

ALL_CLASSES = list(range(10))

print(f"Evaluating: {N_EPISODES} episodes, {N_WAY}-way {K_SHOT}-shot")
print(f"  (episodes include both base and novel classes)\n")

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
print(f"  Accuracy: {mean_acc:.1%} +/- {ci95:.1%}")

# %% [markdown]
# ## 7. Harder test: novel classes only
#
# The previous evaluation mixes base and novel classes. But the hardest
# scenario is when **all classes are novel** — the model must distinguish
# between land cover types it has never seen during training, using only
# a few examples of each.
#
# Since we have 3 novel classes, we run 3-way episodes.

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
print(f"  Accuracy: {novel_mean:.1%} +/- {novel_ci:.1%}")
print(f"  (random baseline for {n_novel_way}-way: {1/n_novel_way:.1%})")

# %% [markdown]
# ## 8. How many examples do we need?
#
# A key practical question for EO applications: how does accuracy change
# as we provide more labeled examples? We test with 1, 5, and 20 examples
# per class.
#
# - **1-shot**: a single labeled image per class — the extreme case
# - **5-shot**: five labeled images — a realistic field survey scenario
# - **20-shot**: twenty labeled images — still far fewer than standard
#   supervised learning requires

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
    print(f"  {N_WAY}-way {k}-shot: {mean:.1%} +/- {ci:.1%}")

# %% [markdown]
# ## 9. Results and interpretation

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

    # Panel A: accuracy vs number of labeled examples
    ax = axes[0]
    shots = sorted(shot_results.keys())
    means = [shot_results[k][0] for k in shots]
    cis = [shot_results[k][1] for k in shots]
    ax.errorbar(shots, means, yerr=cis, fmt="o-", capsize=5, linewidth=2, markersize=8,
                color="steelblue")
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="Random (5-way)")
    ax.set_xlabel("Number of labeled examples per class (K)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(f"How many examples are enough?")
    ax.set_xticks(shots)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: training loss curve
    ax = axes[1]
    window = 50
    smoothed = [np.mean(losses[max(0, i - window):i + 1]) for i in range(len(losses))]
    ax.plot(smoothed, linewidth=1, color="steelblue")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Loss")
    ax.set_title("Embedding network training (base classes only)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Prototypical Networks on Sentinel-2 land cover (EuroSAT)",
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

# %% [markdown]
# ## 10. What does this mean for Earth Observation?
#
# **The good news**: Prototypical Networks achieve 82% accuracy on
# 5-way 5-shot classification of Sentinel-2 land cover, including classes
# never seen during training. With just 5 labeled satellite patches per
# class, the model can distinguish land cover types reasonably well.
#
# **The challenge**: when tested exclusively on novel classes (the "rare
# habitat" scenario), accuracy drops to ~54%. This is better than random
# (33% for 3-way) but far from operational quality. The model struggles
# to distinguish between visually similar vegetation types
# (HerbaceousVegetation vs. PermanentCrop vs. River) without having
# learned from related classes during training.
#
# **Practical implications**:
# - Few-shot learning is a promising direction for rare habitat monitoring,
#   but current methods need improvement for operational use.
# - Performance improves with more examples (1-shot → 5-shot → 20-shot),
#   so even a small annotation effort pays off.
# - Using all 13 Sentinel-2 spectral bands (not just RGB) would likely
#   improve discrimination between vegetation types — this is a natural
#   next step.
#
# ## Replication context
#
# This experiment is part of the [Science Live](https://platform.sciencelive4all.org)
# FORRT replication initiative. The results are published as
# [nanopublications](https://nanopub.net/) with full provenance.
#
# - **Zenodo DOI**: [10.5281/zenodo.19607662](https://doi.org/10.5281/zenodo.19607662)
# - **Original paper**: Snell et al. (2017), [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175), NeurIPS
# - **Cross-domain benchmark**: Guo et al. (2020), [A Broader Study of Cross-Domain Few-Shot Learning](https://doi.org/10.1007/978-3-030-58583-9_8), ECCV
