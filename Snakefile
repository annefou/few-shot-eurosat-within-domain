# Snakefile — Few-Shot EuroSAT Replication
#
# Usage:
#   snakemake --cores 1       # run experiment
#   snakemake --cores 1 -n    # dry run

RESULTS = "results"

rule all:
    input:
        f"{RESULTS}/few_shot_eurosat_results.json",
        f"{RESULTS}/few_shot_eurosat.png",

rule run_experiment:
    output:
        f"{RESULTS}/few_shot_eurosat_results.json",
        f"{RESULTS}/few_shot_eurosat.png",
        f"{RESULTS}/protonet_eurosat.pth",
    log:
        f"{RESULTS}/logs/01_few_shot_eurosat.log",
    shell:
        """
        mkdir -p {RESULTS}/logs
        jupytext --to notebook --execute 01_few_shot_eurosat.py 2>&1 | tee {log}
        """
