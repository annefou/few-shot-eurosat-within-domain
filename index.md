---
title: Few-Shot EuroSAT Within-Domain
subtitle: Can AI learn new land cover types from just 5 satellite images?
---

## Why this matters

Monitoring Europe's protected habitats from satellite imagery requires classifying land cover types — but rare habitats have very few labeled examples. Training a deep learning model typically requires thousands of images per class. What if we could teach a model to recognize new habitat types from just **5 labeled satellite patches**?

This project tests exactly that, using real Sentinel-2 satellite imagery and a method called [Prototypical Networks](https://arxiv.org/abs/1703.05175) from the AI research community.

## What we did

1. **Took 27,000 real Sentinel-2 images** from the [EuroSAT](https://github.com/phelber/EuroSAT) dataset, covering 10 land cover types across Europe
2. **Split into common and rare classes**: 7 common types (forest, cropland, urban...) for training, 3 "rare" types (herbaceous vegetation, permanent crop, river) held back
3. **Trained a Prototypical Network** to learn what different land cover types "look like" — using only the common classes
4. **Tested on the rare classes** with just 1, 5, or 20 labeled examples each

## Results

| Labeled examples per class | Accuracy |
|---------------------------|----------|
| 1 image | 71.5% ± 1.0% |
| 5 images | 82.1% ± 0.8% |
| 20 images | 84.3% ± 0.7% |
| 5 images (rare classes only) | 53.8% ± 0.7% |

**Key finding**: with just 5 labeled images per class, the model reaches 82% accuracy when classifying a mix of common and rare land cover types. But when tested exclusively on rare types it has never seen during training, accuracy drops to 54% — better than random (33%) but not yet operational quality.

## What this means

- **Few-shot learning works** for satellite imagery — it's not limited to the natural photos it was designed for
- **More examples help**: going from 1 to 5 to 20 labeled images steadily improves accuracy
- **The gap for rare classes is real**: models struggle when all classes are unfamiliar, suggesting that some domain-specific training data is still needed
- **Next step**: using all 13 Sentinel-2 spectral bands (not just RGB) should improve discrimination between vegetation types

## Reproducibility

This experiment is fully reproducible:

- **Code**: [GitHub repository](https://github.com/annefou/few-shot-eurosat-within-domain) with Jupytext notebook, Snakemake pipeline, Dockerfile
- **Data**: [EuroSAT](https://zenodo.org/records/7711810) — downloaded automatically (94 MB)
- **CI**: GitHub Actions runs the full experiment on every push
- **Archive**: [Zenodo DOI 10.5281/zenodo.19607662](https://doi.org/10.5281/zenodo.19607662)
- **Runtime**: ~5 minutes on a laptop GPU (Apple M1 Pro), ~13 minutes on CPU

## FORRT replication on Science Live

This work is published as a chain of [nanopublications](https://nanopub.net/) on [Science Live](https://platform.sciencelive4all.org) — small, machine-readable, cryptographically signed scientific assertions.

The chain follows the [FORRT](https://forrt.org/) (Framework for Open and Reproducible Research Training) replication workflow:

1. **Paper quotation** — the original claim from the AI paper, annotated
2. **AIDA sentence** — the claim distilled into a single atomic, independent, declarative, absolute statement
3. **FORRT Claim** — the specific scientific claim being tested
4. **FORRT Replication Study** — what we did, how, and why it differs from the original
5. **FORRT Replication Outcome** — the result: does the claim hold for satellite imagery?

<!-- TODO: add nanopub URIs once published
| Step | Nanopublication |
|------|----------------|
| Paper quotation | [view](https://platform.sciencelive4all.org/np/?uri=...) |
| AIDA sentence | [view](https://platform.sciencelive4all.org/np/?uri=...) |
| FORRT Claim | [view](https://platform.sciencelive4all.org/np/?uri=...) |
| FORRT Study | [view](https://platform.sciencelive4all.org/np/?uri=...) |
| FORRT Outcome | [view](https://platform.sciencelive4all.org/np/?uri=...) |
-->

### Want to publish your own replication?

Science Live makes it easy to turn any replication study into FAIR nanopublications:

1. **Find a claim** in a published paper you want to test
2. **Run your replication** and publish the code (GitHub + Zenodo)
3. **Create nanopubs** on [platform.sciencelive4all.org](https://platform.sciencelive4all.org): quotation → AIDA sentence → claim → study → outcome
4. **Your result is now discoverable, citable, and machine-readable**

Each nanopub is cryptographically signed with your [ORCID](https://orcid.org/), permanently stored on the [nanopub network](https://nanopub.net/), and linked to the original paper, your code, and your data.

## Quick start

```bash
# Clone and run
git clone https://github.com/annefou/few-shot-eurosat-within-domain
cd few-shot-eurosat-within-domain
mamba env create -f environment.yml
mamba activate few-shot-eurosat
python 01_few_shot_eurosat.py
```

Or with Docker:
```bash
docker pull ghcr.io/annefou/few-shot-eurosat-within-domain
docker run ghcr.io/annefou/few-shot-eurosat-within-domain
```
