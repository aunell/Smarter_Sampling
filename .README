# PAC_Judge

## Overview
PAC_Judge provides both theoretical and empirical guarantees for improved sampling methods to accurately estimate the **Intraclass Correlation Coefficient (ICC)** of human and LLM judge rankings of non-verifiable text.  
Our goal is to improve the robustness and efficiency of LLM-judge evaluations, particularly in **data-scarce regimes**.

---

## Getting Started

### Running the LLM Judge
Use the following script to obtain LLM judge scores for a specific dataset:

```bash
python get_judge_scores.py \
    --output_file "../results/judge_scores/${DATASET}/results_${DATASET}_2.json" \
    --sample_size 400 \
    --dataset "$DATASET"
```

### Running Experiments on Real Data
The scripts below perform experiments over all real datasets, varying the budget ratio of human to LLM rankings across different selection strategies. The scripts generate evaluation metrics and plots of the results:

```bash
python ../_1_real_data_experiment_simulation_cleaned.py
python ../_2_confidence_interval_rde.py
```
_1_real_data_experiment_simulation_cleaned.py: Runs experiments across datasets and selection strategies.

_2_confidence_interval_rde.py: Computes and plots confidence interval widths for the rollouts.

Dataset Classes
PAC_Judge uses a flexible dataset abstraction:

pacDataset (../datasets/pac_dataset.py):
Base class for all datasets. Handles data loading, prompt creation, and result formatting.

SummevalDataset (../datasets/summeval.py):
Loads the Summeval dataset, flattens it, and generates prompts for LLM evaluation. Prompts are tailored to each evaluation dimension.

Adding a New Dataset:
To add a new dataset, subclass pacDataset and implement:

extract_data(): Load and preprocess the dataset into a standardized DataFrame.

create_prompt(row): Generate evaluation prompts for each row of data.

## Dataset Information 
MSLR: https://github.com/allenai/mslr-annotated-dataset/blob/main/data/data_with_overlap_scores.json 
HANNA: https://github.com/dig-team/hanna-benchmark-asg/blob/main/hanna_stories_annotations.csv 
SummEval: https://huggingface.co/datasets/mteb/summeval 
MedVAL: https://arxiv.org/pdf/2507.03152
