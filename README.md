# Homework-4
## Student Information
**Name:**  Boreddy Harshavardhan Reddy

**Course:** Natural Language Processing  

---

## Overview
This repository contains implementations for three programming tasks related to Recurrent Neural Networks, Attention, and Transformer architectures.  
All code is written in **PyTorch** and designed to run smoothly on **Google Colab** with GPU acceleration.

---

##  Q1 – Character-Level RNN Language Model
In this task, a character-level text generator is built using an LSTM.  
Key features:
- Custom toy dataset + larger 50–200 KB text file  
- Model: *Embedding → LSTM → Linear → Softmax*  
- Teacher forcing & cross-entropy loss  
- Temperature-based sampling (τ = 0.7, 1.0, 1.2)  
- Plots for training/validation loss  
- Generated text samples (200–400 characters)

**Goal:** Understand sequence modeling, hidden state dynamics, and sampling temperature effects.

---

##  Q2 – Mini Transformer Encoder
A simplified Transformer Encoder was implemented from scratch, including:
- Tokenization & learned embeddings  
- Sinusoidal positional encodings  
- Scaled-dot product self-attention  
- Multi-head attention (2–4 heads)  
- Feed-forward network  
- Add & Norm (Residual + LayerNorm)  
- Visualization of attention weights (heatmaps)

**Goal:** Understand how self-attention builds contextual embeddings.

---

##  Q3 – Scaled Dot-Product Attention
Implements the core attention function:
Includes:
- Random Q, K, V testing  
- Attention weight matrix  
- Output vectors  
- Demonstration of softmax instability without scaling  
- Comparison before/after √dₖ normalization  

**Goal:** Observe why scaling is required for stable gradients.
## Conclusion
This assignment demonstrates:
- How RNNs learn sequential patterns at the character level  
- How Transformer encoders represent contextual meaning  
- How scaled attention improves numerical stability  

The notebooks provide clear, modular implementations suitable for experimentation and further extension.
