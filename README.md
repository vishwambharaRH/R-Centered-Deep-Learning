# Physiologically Guided R-Centered Deep Learning for ECG Wave Delineation Across Heterogeneous Databases

This repository contains the experimental implementation and analysis supporting our ICBBE 2026 paper, *“Physiologically Guided R-Centered Deep Learning for ECG Wave Delineation Across Heterogeneous Databases.”*

The project investigates whether incorporating physiological structure into deep learning can improve ECG waveform delineation, particularly for the *P and T waves, under cross-dataset conditions. The central idea is to use the **R peak as a physiologically meaningful temporal reference* and combine this representation with a CNN–BiLSTM sequence-labeling architecture.

## Repository Overview

The repository is organized around three primary notebooks, corresponding to the major stages of the experimental work:

### 1. Dataset & Preprocessing

This notebook prepares the ECG datasets used throughout the study. It handles record-level splitting, signal preprocessing, annotation processing, sampling-rate harmonization, normalization, and R-centered window construction.

The experiments use the *QT Database (QTDB)* and the *Lobachevsky University Database (LUDB)*. QTDB is used primarily for model development, while LUDB provides an independent database for evaluating cross-dataset generalization and target-domain adaptation.

### 2. Model Development & Ablation

This notebook contains the CNN–BiLSTM model development and the controlled experiments used to study the contribution of different physiological and temporal representations.

In addition to baseline experiments, a matched *A0–A4 ablation study* evaluates R-centered guidance, extended temporal context, an explicit temporal-position channel, and lightweight supervised LUDB adaptation under a common experimental protocol.

The results show that R-centered guidance and additional temporal context can improve performance, while adding further temporal constraints does not necessarily provide a consistent benefit.

### 3. Reviewer Evaluation & Final Analysis

The final notebook contains the additional experiments performed in response to reviewer feedback. These include *event-level P/T evaluation, boundary-error analysis, independent R-peak auditing, per-record analysis, and investigation of the relationship between R-detector errors and downstream P/T delineation performance*.

These analyses provide a more detailed assessment than sample-wise segmentation metrics alone and help evaluate the robustness of the proposed R-centered approach across heterogeneous ECG databases.

## Key Findings

Overall, the experiments indicate that *physiological guidance can improve ECG delineation, but more physiological constraints are not inherently better*. Appropriate R-centered representation provides a useful balance between physiological structure and data-driven learning, while lightweight target-domain adaptation can further improve cross-dataset performance.

The repository is intended to provide the experimental basis for the reported results, analyses, and conclusions in the accompanying ICBBE 2026 paper.

## Datasets

•⁠  ⁠QT Database (QTDB)
•⁠  ⁠Lobachevsky University Database (LUDB)

Both datasets are publicly available through their respective PhysioNet/database sources and should be obtained according to their original licensing and usage requirements.