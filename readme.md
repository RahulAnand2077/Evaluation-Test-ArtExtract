# Deep Learning for Art Classification and Retrieval

This repository contains the evaluation tasks by HumanAI Foundation for the Google Summer of Code (GSoC) application. The project focuses on building highly accurate, scalable Deep Learning pipelines for the analysis of fine art, tackling both multi-task classification and Content-Based Image Retrieval (CBIR).

> **Note on Documentation:** This README provides a high-level summary of the final architectures and metrics. For a comprehensive, theoretical breakdown of ***why*** specific architectural decisions were made (e.g., ConvNeXt vs. ResNet feature extraction, Triplet Margin dynamics, and Self-Attention A/B testing), please refer to the heavily commented Markdown inside each task's `pipeline.ipynb` notebook.

## Table of Contents
- [Task 1: Multi-Task Classification (CRNN)](#task-1-multi-task-classification-crnn)
- [Task 2: Content-Based Image Retrieval (CBIR)](#task-2-content-based-image-retrieval-cbir)

---

## Task 1: Multi-Task Classification (CRNN)

**Objective:** Build a Convolutional-Recurrent Neural Network (CRNN) to simultaneously predict the **Artist**, **Genre**, and **Style** of a given painting.

### Architectural Strategy
*(Detailed architectural justifications and A/B testing theory are available in `/Task1/pipeline.ipynb`)*

To fulfill the specific "convolutional-recurrent" requirement while achieving state-of-the-art efficiency, the proposed pipeline utilizes a **ConvNeXt-Tiny + 2-Layer Bi-GRU** architecture:
* **The Spatial Extractor (ConvNeXt):** Replaces dated ResNet backbones, utilizing massive 7x7 depthwise convolutions to extract rich, hierarchical visual features.
* **The Sequential Scanner (Bi-GRU):** By flattening the spatial feature grid, the network treats the canvas as a sequence. The Bidirectional GRU scans the painting simultaneously top-to-bottom and bottom-to-top, effectively capturing the sequential, directional flow of the artist's brushstrokes.
* **Multi-Task Joint Optimization:** A single context vector feeds three distinct classification heads. Optimizing the joint loss ($\mathcal{L}_{total} = \mathcal{L}_{artist} + \mathcal{L}_{genre} + \mathcal{L}_{style}$) acts as a massive cross-task regularizer, preventing individual heads from overfitting to spurious local details.

### Results (Deterministic Evaluation)
Metrics are calculated using **Macro F1** on a rigid `CenterCrop` validation set to handle severe class imbalances and ensure purely deterministic scoring.

| Architecture | Artist F1 | Genre F1 | Style F1 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet + Bi-LSTM** | 0.5794 | 0.4243 | 0.4358 | Baseline CRNN. |
| **ViT-B/16** | 0.5429 | 0.5023 | 0.4146 | SOTA Baseline (Breaks CRNN constraint & Trained for ***1 Epoch***). |
| **ConvNeXt + Bi-GRU** | **0.8429** | **0.6440** | **0.6690** | **Proposed Architecture.** |

---

## Task 2: Content-Based Image Retrieval (CBIR)

**Objective:** Build a model to find similarities in paintings (e.g., portraits with a similar face or pose) using the National Gallery of Art open dataset.

### Methodology: Siamese Triplet Network
*(Detailed theoretical breakdown of Metric Learning and the ViT vs. CNN experiment is available in `/Task2/pipeline.ipynb`)*

Standard classification is incompatible with continuous metric spaces. This task is framed as a Deep Metric Learning problem using pseudo-labeled triplets (Anchor, Positive, Negative) derived from metadata.
* **The Projector:** The network acts purely as a spatial feature extractor, mapping images onto a 256-dimensional $L_2$ normalized hypersphere.
* **Triplet Margin Loss:** Prevents representation collapse by forcing the network to ensure the Positive sample is strictly closer to the Anchor than the Negative sample by a mathematical margin ($\alpha=1.0$).

### Results (Deterministic Evaluation)
To determine the optimal spatial projector, local Convolutional filters were tested against global Self-Attention. Performance is evaluated using **Recall@5** (simulating a real-world visual search engine).

| Architecture | Epochs | Triplet Loss | Recall@5 |
| :--- | :--- | :--- | :--- |
| **Custom ResNet (CNN)** | 30 | 0.6651 | 91.38% |
| **ViT-B/16 (Self-Attention)** | 30 | 0.0101 | **100.00%** |

**Conclusion:** The Vision Transformer achieved absolute convergence. Visual inference confirms the Triplet Loss successfully organized internal cluster geometries, grouping portraits with similar poses, subjects, and stylistic features tightly together in the continuous vector space.

---
