# Residue-Residue Contact Prediction Model

This repository contains the implementation of a machine learning model for predicting residue-residue contacts within protein sequences. The model is based on extending the ESM2 architecture, incorporating additional structural data to improve the prediction of contact pairs in a given protein. The project aims to enhance understanding of protein folding and interactions by accurately predicting which residues are in close spatial contact.


## Objective

The goal of this project is to predict residue-residue contacts within protein sequences, leveraging both ESM2 model embeddings and additional structural data to provide more accurate predictions. A residue-residue contact is defined as a pair of residues that are spatially close (distance between Cα atoms below 8 Å).

The model aims to output a **binary contact map**, where a value of `1` indicates a contact between a pair of residues, while a value of `0` indicates no contact.

## Approach

### Model Architecture

The model builds upon the **ESM2 architecture** by using its pre-trained embeddings and extending the model to integrate structural data from similar protein sequences. The main components include:

- **Graph Neural Network (GNN)**: A GCN-based model to capture complex residue-residue relationships from both sequence and structural data.
- **ESM2 Embeddings**: The ESM2 model is used to obtain high-quality sequence embeddings, which are further enhanced with structural data.

The model is trained to output a binary classification for each residue pair, indicating whether or not they are in contact.

### Dataset

- The **Protein Data Bank (PDB)** was used as the source of sequence and 3D structural data.
- Training is implemented using the **PyTorch Geometric** library to handle graph structures efficiently.

### Training and Evaluation

- The training process is implemented in `training.py` and leverages **batch training** to handle large datasets.
- The **Binary Cross-Entropy with Logits Loss** (`BCEWithLogitsLoss`) is used for training the model to address the binary classification nature of the task.
- The model is trained using **Adam optimizer** with standard hyperparameters, which can be tuned for better performance.

### Hyperparameters

- **Learning Rate**: 0.001
- **Hidden Dimension**: 128

These hyperparameters were chosen based on preliminary experimentation, and further tuning can lead to better model performance.

### Results Analysis

- The **analyze.ipynb** notebook is used to evaluate the model's performance

## Installation

To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

Ensure you have **Python 3.8+** installed

   ```bash
   python training.py
   ```

## Future Work

- **Hyperparameter Tuning**: Further tuning of hyperparameters to improve model performance.
- **Model Extension**: Explore using additional structural features or alternative architectures such as attention mechanisms for improved contact prediction.

## References

- **ESM2**: [ESM2 Architecture by Facebook AI Research (FAIR)](https://github.com/facebookresearch/esm)
- **Protein Data Bank**: [PDB Website](https://www.rcsb.org/)
