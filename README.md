# DeepGazeIIIE (ACW) - Enhanced Visual Saliency Prediction

Welcome to the official repository for our implementation of the DeepGazeIIIE (ACW) model, an extension of the state-of-the-art DeepGazeIIIE architecture for visual saliency prediction. This repository contains all the necessary code, data processing scripts, and model configurations to reproduce the experiments described in our research.

## Overview
The goal of this project is to predict human visual attention by identifying salient regions in images. Our approach builds upon the DeepGazeIIE architecture by introducing category-specific weighting matrices across backbones, optimizing saliency map generation, and addressing computational constraints with innovative upsampling techniques. While our model achieved competitive results in certain metrics, it serves as a foundation for further exploration into domain-specific saliency modeling.

## Key Features

- Category-Specific Weighting: Assigns adaptive weights to backbones based on image categories (e.g., "Black and White," "Cartoon").
- Progressive Upsampling: Improves spatial resolution of saliency maps through multiple bicubic interpolation stages.
- Probabilistic Framework: Incorporates a log-likelihood-based loss function for end-to-end optimization.
- Dynamic Backbone Adaptation: Utilizes multiple backbones (ResNext50, DenseNet201, EfficientNetB5, ShapeNetC) initialized with random seeds for diversified predictions.
- Evaluation on CAT2000 Dataset: Trained and evaluated using the CAT2000 dataset, leveraging 1280 training images and 400 test images.

## Results
Our model was evaluated using metrics such as Information Gain (IG), Normalized Scanpath Saliency (NSS), Area Under the Curve (AUC), and others on the CAT2000 dataset. Below is a summary of our performance:

Model	IG	AUC	sAUC	NSS	CC	KLDiv	SIM
DeepGazeIIIE (SOTA)	0.1893	0.8692	0.6677	2.1122	0.8189	0.3448	0.706
DeepGazeIIIE ACW	0.1795	0.8883	0.4946	1.2663	0.345	8.3372	0.3427
Note: Our evaluation metrics were calculated using custom implementations and may not fully align with benchmark scores.
