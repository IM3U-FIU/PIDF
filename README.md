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
 
![CWgraph2](https://github.com/user-attachments/assets/978f58e3-1044-4e5b-99b4-658d447b69d0)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/IM3U-FIU/PIDF.git
   cd PIDF


3. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt  


## Usage

### Synthetic Data Experiments
Results:

![all_datasets](https://github.com/user-attachments/assets/5fc7c2ab-5a65-4335-b31a-0e6a3d07941e)

To generate the above results on synthetic data run:

    ```bash
    python main.py RVQ 20000 False
    python main.py SVQ 20000 False
    python main.py MSP 20000 False
    python main.py WT 20000 False
    python main.py TERC1 20000 False
    python main.py TERC2 20000 False
    python main.py UBR 20000 False
    python main.py SG 20000 False

### Housing Experiments

Results:

![housing](https://github.com/user-attachments/assets/9ab5e013-7275-4bf4-8c23-ab897ebbf0ba)

To generate the above results on housing data run:

    ```bash
    python main.py housing 20000 False  

### BRCA Experiments

Results:

![brca_small](https://github.com/user-attachments/assets/c9dc73d1-ad39-4c50-8993-610ef57d15a2)

To generate the above results on BRCA data run:

    ```bash
    python main.py brca_small 20000 False  

### Neurons Experiments

Results:

![neurons_average_line](https://github.com/user-attachments/assets/f890bbd3-e69a-4a0a-9a3b-929b0916a9da)

To generate the above results on neuron data run the following with multiple seeds:

    ```bash
    python main.py timme_neurons_day4 20000 False
    python main.py timme_neurons_day7 20000 False
    python main.py timme_neurons_day12 20000 False
    python main.py timme_neurons_day16 20000 False
    python main.py timme_neurons_day20 20000 False
    python main.py timme_neurons_day25 20000 False
    python main.py timme_neurons_day31 20000 False
    python main.py timme_neurons_day33 20000 False

  These are the most computationally expensive experiments to run, we recommend you first collect results using our scalable method (set scalable=True in class init).

### Feature Selection Experiments
To generate the above results on synthetic data run:

    ```bash
    python main.py RVQ 500 True
    python main.py SVQ 500 True
    python main.py MSP 500 True
    python main.py WT 500 True
    python main.py TERC1 500 True
    python main.py TERC2 500 True
    python main.py UBR 500 True
    python main.py SG 500 True

## Datasets

- **BRCA:** [Download Link](https://github.com/HydroML/UMFI)
- **Neurons:** [Download Link](https://potterlab.gatech.edu/potter-lab-data-code-and-designs/)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{westphal2025partial,
      title={Partial Information Decomposition for Data Interpretability and Feature Selection}, 
      author={Charles Westphal and Stephen Hailes and Mirco Musolesi},
      year={2025},
      booktitle={AISTATS'25},
}
