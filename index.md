---
title: Few-Shot EuroSAT Within-Domain
subtitle: Common to Rare Land Cover Transfer with Prototypical Networks
---

## Overview

This project tests whether [Prototypical Networks](https://arxiv.org/abs/1703.05175) (Snell et al., NeurIPS 2017) can classify rare land cover types from Sentinel-2 satellite imagery with only a handful of labeled examples.

We train on 7 common EuroSAT land cover classes and evaluate few-shot transfer to 3 "novel" classes, simulating a [Natura 2000](https://en.wikipedia.org/wiki/Natura_2000) habitat monitoring scenario where abundant training data exists for common habitats but rare, ecologically important types have very few annotations.

## Results

| Setting | Accuracy |
|---------|----------|
| 5-way 1-shot | 71.5% ± 1.0% |
| 5-way 5-shot | 82.1% ± 0.8% |
| 5-way 20-shot | 84.3% ± 0.7% |
| 3-way 5-shot (novel only) | 53.8% ± 0.7% |

## Replication context

Part of the [Science Live](https://platform.sciencelive4all.org) FORRT replication initiative, published as nanopublications with full provenance.

- **Zenodo DOI**: [10.5281/zenodo.19607662](https://doi.org/10.5281/zenodo.19607662)
- **Dataset**: [EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 Sentinel-2 images, 10 classes

## Quick start

```bash
mamba env create -f environment.yml
mamba activate few-shot-eurosat
python 01_few_shot_eurosat.py
```
