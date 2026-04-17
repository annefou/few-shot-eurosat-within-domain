# Few-Shot EuroSAT Within-Domain: Common to Rare Land Cover Transfer

[![Run Few-Shot Experiment](https://github.com/annefou/few-shot-eurosat-within-domain/actions/workflows/run-experiment.yml/badge.svg)](https://github.com/annefou/few-shot-eurosat-within-domain/actions/workflows/run-experiment.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19607662.svg)](https://doi.org/10.5281/zenodo.19607662)
[![Jupyter Book](https://img.shields.io/badge/Jupyter%20Book-live-orange)](https://annefou.github.io/few-shot-eurosat-within-domain/)

Can a model trained on common land cover types (forest, residential, highway) classify rare habitat types (herbaceous vegetation, permanent crop, river) with only a handful of labeled examples?

This project applies [Prototypical Networks](https://arxiv.org/abs/1703.05175) (Snell et al., NeurIPS 2017) — an AI method that learns to classify new categories from very few examples — to real [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) satellite imagery from the [EuroSAT](https://github.com/phelber/EuroSAT) dataset. It simulates a practical [Natura 2000](https://en.wikipedia.org/wiki/Natura_2000) monitoring scenario where abundant training data exists for common habitats but only a few annotated examples are available for rare, ecologically important habitat types.

## Results

| Labeled examples per class | Accuracy |
|---------------------------|----------|
| 1 image | 71.5% ± 1.0% |
| 5 images | 82.1% ± 0.8% |
| 20 images | 84.3% ± 0.7% |
| 5 images (rare classes only) | 53.8% ± 0.7% |

With just 5 labeled images per class, the model reaches 82% accuracy when classifying a mix of common and rare land cover types. When tested exclusively on rare types never seen during training, accuracy drops to 54% — better than random (33%) but not yet operational quality.

## FORRT Nanopublications

This work is published as a chain of [nanopublications](https://nanopub.net/) on [Science Live](https://platform.sciencelive4all.org) following the [FORRT](https://forrt.org/) replication framework:

| Step | Nanopublication |
|------|----------------|
| Paper quotation | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RA7g02DAdAV4zUy1_u7_lD7GnTDPkbnpyUIsfGDmv-yyE) |
| AIDA sentence | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAnb5--dk_Xp5iV1nZWXLdY_TrvLReMfYwaHcaRyUAI8I) |
| FORRT Claim | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/np/RAPEj_VkTBh17NfklyuB0klzxobwyp2dsx6vJRByVs148) |
| FORRT Replication Study | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RALGj6JatoumnOiiTUie18OUEVGpeAKuhvUkLpAa2Wjjc) |
| FORRT Replication Outcome | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAUS6GbT3Bu-Np0Ue73q58G_c2HilLhh95Y2b8W18o--M) |

## Quick start

```bash
git clone https://github.com/annefou/few-shot-eurosat-within-domain
cd few-shot-eurosat-within-domain
mamba env create -f environment.yml
mamba activate few-shot-eurosat
python 01_few_shot_eurosat.py
```

Or with Docker:

```bash
docker build -t few-shot-eurosat-within .
docker run few-shot-eurosat-within
```

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 real Sentinel-2 satellite image patches (64×64 px, 10 m ground resolution), 10 land use/land cover classes across Europe. Downloaded automatically on first run (~94 MB).

## Method

Prototypical Networks learn to map satellite images into a feature space where similar land cover types cluster together. At test time, the model computes a "prototype" (average feature vector) from the few labeled examples of each class, then classifies new images by finding the nearest prototype. No retraining is needed for new classes.

We train on 7 common land cover classes and evaluate on 3 held-back "rare" classes with 1, 5, or 20 labeled examples.

## Related

- [few-shot-eurosat-cross-domain](https://github.com/annefou/few-shot-eurosat-cross-domain) — Strict replication of Guo et al. (2020, ECCV): train on mini-ImageNet, test on EuroSAT

## Citation

```bibtex
@software{fouilloux2026fewshot_within,
  author = {Fouilloux, Anne},
  title = {Few-Shot EuroSAT Within-Domain: Common to Rare Land Cover Transfer},
  year = {2026},
  doi = {10.5281/zenodo.19607662},
  url = {https://github.com/annefou/few-shot-eurosat-within-domain}
}
```

## License

MIT — see [LICENSE](LICENSE).
