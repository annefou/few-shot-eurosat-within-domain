# Few-Shot EuroSAT Within-Domain: Common to Rare Land Cover Transfer

Within-domain few-shot classification on [EuroSAT](https://github.com/phelber/EuroSAT) Sentinel-2 imagery using [Prototypical Networks (Snell et al., 2017, NeurIPS)](https://arxiv.org/abs/1703.05175).

**Research question**: Can a model trained on common land cover types (forest, residential, highway) classify rare habitat types (herbaceous vegetation, permanent crop, river) with only a handful of labeled examples?

This simulates a practical Natura 2000 monitoring scenario where abundant training data exists for common habitats but only a few annotated examples are available for rare, ecologically important habitat types.

## Results

| Setting | Accuracy |
|---------|----------|
| 5-way 1-shot | 71.5% ± 1.0% |
| 5-way 5-shot | 82.1% ± 0.8% |
| 5-way 20-shot | 84.3% ± 0.7% |
| 3-way 5-shot (novel only) | 53.8% ± 0.7% |

## Quick start

```bash
mamba env create -f environment.yml
mamba activate few-shot-eurosat
snakemake --cores 1
```

## Docker

```bash
docker build -t few-shot-eurosat-within .
docker run few-shot-eurosat-within
```

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 Sentinel-2 images (64×64 px), 10 land use/land cover classes. Downloaded automatically on first run.

## Method

Prototypical Networks learn an embedding space where classification is performed by computing distances to class prototypes (centroids). We train on 7 "base" land cover classes and evaluate few-shot transfer to 3 "novel" classes simulating rare habitats.

## Related

- [few-shot-eurosat-cross-domain](https://github.com/annefou/few-shot-eurosat-cross-domain) — Strict replication of Guo et al. (2020): train on mini-ImageNet, test on EuroSAT

## Citation

If you use this work, please cite:

```bibtex
@software{fouilloux2026fewshot_within,
  author = {Fouilloux, Anne},
  title = {Few-Shot EuroSAT Within-Domain: Common to Rare Land Cover Transfer},
  year = {2026},
  url = {https://github.com/annefou/few-shot-eurosat-within-domain}
}
```

## License

MIT — see [LICENSE](LICENSE).
