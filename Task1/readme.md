# Evaluation Task 1: WikiArt Multi-Task Classification

Implemented a custom Convolutional Recurrent Neural Network (CRNN) to jointly classify the **Artist**, **Genre**, and **Style** of paintings from the WikiArt dataset.

## Project Highlights
* **Architecture:** A ResNet feature extractor bridged to a BiLSTM sequence model to capture both spatial textures and sequential brushstroke cadences.
* **Multi-Task Learning:** Three joint classification heads optimized simultaneously, utilizing cross-task regularization.
* **Metric Focus:** Evaluated using **Macro F1** to rigorously account for the severe class imbalance inherent in the WikiArt dataset.
* **Anomaly Detection:** Includes an automated pipeline to identify high-confidence misclassifications (label noise/outliers) within the validation set.

## Final Results (50 Epochs)
The model was trained for 50 epochs, saving the optimal weights based on the highest Average Macro F1 across all three tasks.

| Task | Macro F1 Score |
| :--- | :--- |
| **Artist** | 0.545 |
| **Style** | 0.433 |
| **Genre** | 0.419 |
| **Average F1**| **0.466** |
