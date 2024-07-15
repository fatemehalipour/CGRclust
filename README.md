# CGRclust: Chaos Game Representation for Twin Contrastive  Clustering of Unlabelled DNA Sequences

## Overview
[CGRclust](https://arxiv.org/abs/2407.02538) is a novel unsupervised clustering framework that utilizes Chaos Game Representation (CGR) of DNA sequences combined with twin contrastive learning and convolutional neural networks (CNNs). This method is designed to efficiently and accurately cluster unlabelled DNA sequences, overcoming the limitations of traditional DNA sequence classification and clustering methods that rely on sequence alignment and biological annotations.



## Features
- **Unsupervised Learning**: Leverages a novel approach using twin contrastive clustering that doesn't rely on labelled data.
- **Chaos Game Representation (CGR)**: Utilizes CGR to transform DNA sequences into two-dimensional images that are then used for clustering.
- **Convolutional Neural Networks**: Employs CNNs to enhance feature extraction from CGR images, improving the clustering performance.
- **High Accuracy and Scalability**: Demonstrates superior clustering accuracy and scalability across various genomic datasets, including those of viruses and mitochondrial DNA from multiple species.

## Installation

```bash
git clone https://github.com/yourgithubusername/CGRclust.git
cd CGRclust
pip install -r requirements.txt
```

## Clustering
```bash
python3 src/cluster.py --dataset="01_Cypriniformes.fasta"
```

### Clustering parameters

| Parameter                     | Description                                              | Default Value               |
|-------------------------------|----------------------------------------------------------|-----------------------------|
| `--dataset`                   | Choose a fasta file in the data directory                | `"01_Cypriniformes.fasta"`  |
| `--k`                         | k-mer size, an integer between 6-8                       | `6`                         |
| `--weak_mutation_rate`        | Weak mutation rate for augmented data                    | `1e-4`                      |
| `--strong_mutation_rate`      | Strong mutation rate for augmented data                  | `1e-2`                      |
| `--weak_fragmentation_perc`   | Weak fragmentation percentage for augmented data         | `None`                      |
| `--strong_fragmentation_perc` | Strong fragmentation percentage for augmented data       | `None`                      |
| `--number_of_pairs`           | Number of augmented data pairs to generate               | `1`                         |
| `--number_of_models`          | Number of models                                         | `5`                         |
| `--lr`                        | Learning rate                                            | `7e-5`                      |
| `--weight_decay`              | Weight decay                                             | `1e-4`                      |
| `--temp_ins`                  | Instance temperature                                     | `0.1`                       |
| `--temp_clu`                  | Cluster temperature                                      | `1.0`                       |
| `--num_epochs`                | Number of epochs                                         | `150`                       |
| `--batch_size`                | Batch size                                               | `512`                       |
| `--embedding_dim`             | Embedding dimension                                      | `512`                       |
| `--feature_dim`               | Feature dimension                                        | `128`                       |
| `--random_seed`               | Random seed                                              | `0`                         |
| `--weight`                    | Weight parameter ($\alpha$) that balances instance-level and cluster-level losses                                                 | `0.7`                       |
