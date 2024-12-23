# DeepGazeIIE (ACW) - Enhanced Visual Saliency Prediction

Welcome to the official repository for our implementation of the DeepGazeIIE (ACW) model, an extension of the state-of-the-art DeepGazeIIE architecture for visual saliency prediction. This repository contains all the necessary code, data processing scripts, and model configurations to reproduce the experiments described in our research.

## Overview
The goal of this project is to predict human visual attention by identifying salient regions in images. Our approach builds upon the DeepGazeIIE architecture by introducing category-specific weighting matrices across backbones, optimizing saliency map generation, and addressing computational constraints with innovative upsampling techniques. While our model didn't achieved competitive results in certain metrics, it serves as a foundation for further exploration into domain-specific saliency modeling. Our results demonstrated the effectiveness of assigning adaptive weights to different categories, highlighting how various backbones respond uniquely to certain types of stimuli. This finding suggests a broader potential for saliency models to specialize in specific domains, opening new avenues for specific applications like health (e.g., diagnosing visual impairments or designing visual aids), marketing (e.g., optimizing advertisements for target demographics), or autonomous systems (e.g., enhancing robotic navigation in diverse environments).

## Key Features

- Category-Specific Weighting: Assigns adaptive weights to backbones based on image categories (e.g., "Black and White," "Cartoon").
- Fine tuned Upsampling: Improves spatial resolution of saliency maps through multiple interpolation stages.
- Probabilistic Framework: Probailistic saliency map modeling.
- Dynamic Backbone Adaptation: Utilizes multiple backbones (ResNext50, DenseNet201, EfficientNetB5, ShapeNetC) initialized with random seeds for diversified predictions.
- Evaluation on CAT2000 Dataset: Trained and evaluated using the MIT CAT2000 dataset.

Feel free to read the full [report](https://github.com/AdrianGalvanDiaz/DeepGazeIIE_ACW/blob/main/Where%20we%20look_%20Approach%20to%20Predicting%20Visual%20Attention%20in%20Images%20using%20Deep%20Learning.docx%20(1).pdf) for a more detailed explanation.

## Results
Our model was evaluated using metrics such as Information Gain (IG), Normalized Scanpath Saliency (NSS), Area Under the Curve (AUC), and others on the CAT2000 dataset. Below is a summary of our performance:

| **Model**         | **IG**  | **AUC**  | **sAUC** | **NSS**  | **CC**   | **KLDiv** | **SIM**   |
|--------------------|---------|----------|----------|----------|----------|-----------|-----------|
| DeepGazeIIE (SOTA)| 0.1893  | 0.8692   | 0.6677   | 2.1122   | 0.8189   | 0.3448    | 0.706     |
| **DeepGazeIIE ACW**| 0.1795  | 0.8883   | 0.4946   | 1.2663   | 0.345    | 8.3372    | 0.3427    |

Note: Our evaluation metrics were calculated using custom implementations and may not fully align with [benchmark scores](https://saliency.tuebingen.ai/results_CAT2000.html). If you want to learn what each metric means I suggest you visit this [very detailed paper](http://olivalab.mit.edu/Papers/08315047.pdf). 

## Visual Examples
Below are examples of saliency maps generated by our model compared to the state-of-the-art:

### Our model performance 
![alt text](<images/Screenshot 2024-11-22 112425.png>)
![alt text](<images/Screenshot 2024-11-22 112501.png>)

### DeepGazeIIE SOTA performance 
![alt text](<images/Screenshot 2024-11-22 112443.png>)
![alt text](<images/Screenshot 2024-11-22 112518.png>)

For additional examples, refer to the [gauss_weights_upscaling.ipynb](https://github.com/AdrianGalvanDiaz/DeepGazeIIE_ACW/blob/main/deepgaze_pytorch/gauss_weights_upscaling.ipynb) file. Debugs were done in this file, ignore them and scroll down. That file makes visualizations comparing de SOTA DeepGazeIIE and our model. 

## Citations
If you use this repository, please cite the following:

- [DeepGazeIIE](https://openaccess.thecvf.com/content/ICCV2021/html/Linardos_DeepGaze_IIE_Calibrated_Prediction_in_and_Out-of-Domain_for_State-of-the-Art_Saliency_ICCV_2021_paper.html)
- [DeepGazeIIE ACW](https://github.com/AdrianGalvanDiaz/DeepGazeIIE_ACW/blob/main/Where%20we%20look_%20Approach%20to%20Predicting%20Visual%20Attention%20in%20Images%20using%20Deep%20Learning.docx%20(1).pdf)
- [CAT2000](https://arxiv.org/abs/1505.03581)

## Contributors

### Adrián Galván Díaz
[Github](https://github.com/AdrianGalvanDiaz) | 
[LinkedIn](https://www.linkedin.com/in/adrian-galvan-15780826a/)