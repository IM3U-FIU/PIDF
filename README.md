# PIDF: Partial Information Decomposition for Data Interpretability and Feature Selection

This repository contains the code used to generate the results presented in the paper:

> **Partial Information Decomposition for Data Interpretability and Feature Selection**  
> *[Charles Westphal], et al.*  
> [arXiv:2405.19212](https://arxiv.org/abs/2405.19212)
> To appear in AiStats'25.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Citation](#citation)

## Introduction

This repository contains the code used to generate the results in our work on Partial Information Decomposition for Data Interpretability and Feature Selection. Our method, PIDF, offers a novel approach to understanding how features interact to predict a target. Unlike traditional feature importance techniques that assign a single importance value to each feature, PIDF breaks down the information into three distinct components. Specifically, it quantifies the mutual information between a feature and the target, the synergistic information that a feature produces in conjunction with other features, and the redundant information shared among features. By clearly presenting these three quantities as shown in the following schematic, PIDF provides insights into feature interactions that enhance both interpretability and feature selection.
 
 ![CWgraph (1).pdf](https://github.com/user-attachments/files/19166260/CWgraph.1.pdf)
30a-f55d135d0d57)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/c-s-westphal/PIDF.git
   cd PIDF

2. **Create Virtual Environment:**

   ```bash
   python3 -m PIDF_venv venv
   source PIDFD_venv/bin/activate

3. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt  


## Usage

### Synthetic Data Experiments
Results:
[all_datasets (6).pdf](https://github.com/user-attachments/files/19166671/all_datasets.6.pdf)
To generate the above results on synthetic data run:

    ```bash
    python main.py --name RVQ --num_iters 20000 --feature_selection False
    python main.py --name SVQ --num_iters 20000 --feature_selection False
    python main.py --name MSP --num_iters 20000 --feature_selection False
    python main.py --name WT --num_iters 20000 --feature_selection False
    python main.py --name TERC1 --num_iters 20000 --feature_selection False
    python main.py --name TERC2 --num_iters 20000 --feature_selection False
    python main.py --name UBR --num_iters 20000 --feature_selection False
    python main.py --name SG --num_iters 20000 --feature_selection False

### Housing Experiments

Results:
[housing.pdf](https://github.com/user-attachments/files/19166801/housing.pdf)
To generate the above results on housing data run:

    ```bash
    python main.py --name housing --num_iters 20000 --feature_selection False  

### BRCA Experiments

Results:
[housing.pdf](https://github.com/user-attachments/files/19166801/housing.pdf)
To generate the above results on BRCA data run:

    ```bash
    python main.py --name brca_small --num_iters 20000 --feature_selection False  

### Neurons Experiments

Results:
[neurons_average_line (1).pdf](https://github.com/user-attachments/files/19166835/neurons_average_line.1.pdf)
To generate the above results on neuron data run the following with multiple seeds:

    ```bash
    python main.py --name timme_neurons_day4 --num_iters 20000 --feature_selection False
    python main.py --name timme_neurons_day7 --num_iters 20000 --feature_selection False
    python main.py --name timme_neurons_day12 --num_iters 20000 --feature_selection False
    python main.py --name timme_neurons_day16 --num_iters 20000 --feature_selection False
    python main.py --name timme_neurons_day20 --num_iters 20000 --feature_selection False
    python main.py --name timme_neurons_day25 --num_iters 20000 --feature_selection False
    python main.py --name timme_neurons_day31 --num_iters 20000 --feature_selection False
    python main.py --name timme_neurons_day33 --num_iters 20000 --feature_selection False

  These are the most computationally expensive experiments to run, we recommend you first collect results using our scalable method (set scalable=True in class init).

### Feature Selection Experiments
To generate the above results on synthetic data run:

    ```bash
    python main.py --name RVQ --num_iters 500 --feature_selection True
    python main.py --name SVQ --num_iters 500 --feature_selection True
    python main.py --name MSP --num_iters 500 --feature_selection True
    python main.py --name WT --num_iters 500 --feature_selection True
    python main.py --name TERC1 --num_iters 500 --feature_selection True
    python main.py --name TERC2 --num_iters 500 --feature_selection True
    python main.py --name UBR --num_iters 500 --feature_selection True
    python main.py --name SG --num_iters 500 --feature_selection True

## Datasets

BRCA: 
Neurons: 
- **BRCA:** [Download Link](https://github.com/HydroML/UMFI)
- **Neurons:** [Download Link](https://potterlab.gatech.edu/potter-lab-data-code-and-designs/)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@proceedings{westphal2025partial,
      title={Partial Information Decomposition for Data Interpretability and Feature Selection}, 
      author={Charles Westphal and Stephen Hailes and Mirco Musolesi},
      year={2025},
      booktitle={AISTATS'25},
}
