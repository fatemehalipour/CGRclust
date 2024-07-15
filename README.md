# CGRclust: Chaos Game Representation for Twin Contrastive  Clustering of Unlabelled DNA Sequences

## Overview
CGRclust is a novel unsupervised clustering framework that utilizes Chaos Game Representation (CGR) of DNA sequences combined with twin contrastive learning and convolutional neural networks (CNNs). This method is designed to efficiently and accurately cluster unlabelled DNA sequences, overcoming the limitations of traditional DNA sequence classification and clustering methods that rely on sequence alignment and biological annotations.

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

