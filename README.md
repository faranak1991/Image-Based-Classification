# Deep Learning for Quantitative Classification of Biomolecular Condensate Material States [app](https://image-based-classification-faranak-hatami.streamlit.app/)

This project presents a deep learning framework for the automated classification of **biomolecular condensate material states** directly from microscopy images.  
The study focuses on distinguishing condensates across five classes:

- **Droplets**
- **Gels**
- **Aggregates**
- **Amorphous Dense Liquids (ADLs)**
- **Solution**

## Motivation

Biomolecular condensates can adopt multiple material states, and these states are functionally important in both normal physiology and disease.  
In practice, condensate state is often assigned by manual inspection of bright-field microscopy images. However, manual labeling is:

- subjective
- low-throughput
- dependent on user expertise
- sensitive to inter-observer variability
- difficult for weak-contrast or morphologically ambiguous images

This project addresses these limitations by providing an **AI-based, automated, and high-throughput classification framework** for condensate-state prediction.

## What this study does

We developed and evaluated a machine learning pipeline for classifying condensate material states from microscopy images.  
The framework includes:

- dataset curation across multiple condensate systems and experimental conditions
- image preprocessing and normalization
- classical machine learning baselines
- deep learning models
- transfer learning with pretrained CNN backbones
- ensemble modeling for robust prediction
- uncertainty-aware output probabilities

The best-performing approach is a **weighted ensemble** of three fine-tuned convolutional neural networks:

- **DenseNet121**
- **ResNet18**
- **EfficientNet-B0**

## Main findings

- Fine-tuned CNNs achieved strong and balanced performance across condensate classes.
- The weighted ensemble produced the most stable results and reduced cross-class confusion.
- The model was especially useful for **challenging and ambiguous morphologies**, including cases that are difficult to classify by human inspection.
- The framework helps reduce:
  - morphology-driven bias
  - dependence on expert judgment
  - inter-observer disagreement
  - manual labeling burden

## Application

A public web app is available for predicting condensate state from microscopy images:

**Streamlit app:**  
https://image-based-classification-faranak-hatami.streamlit.app/

Users can upload microscopy images and obtain automated predictions for condensate material state.

## Potential use cases

This framework can support:

- large-scale image annotation
- perturbation studies
- mutational screening
- condensate phenotyping
- drug screening
- reproducible labeling across laboratories

## Output

For each uploaded microscopy image, the model predicts the most likely condensate class and can provide probability-based confidence across the five states.

## Relevance

This work provides a scalable and objective approach for studying condensate behavior and material-state transitions from imaging data, with potential applications in both basic biophysics and disease-related condensate research.

## Citation

If you use this framework, please cite the associated study/manuscript once available.

## Contact

For questions about the study, dataset, or app, please contact the project author(s).

