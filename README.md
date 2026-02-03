
# Machine Learning Integrated in Wavelet Shrinkage (MLWS)

## Overview
The goal of this project is to develop a new signal denoising framework called Machine Learning Integrated Wavelet Shrinkage (MLWS). This method combines wavelet thresholding with machine learning to enhance noise suppression and signal reconstruction. Unlike traditional single-threshold shrinkage techniques, MLWS introduces two adaptive thresholds to classify wavelet coefficients into signal, noise, and uncertain regions. Machine learning classifiers are then applied to decide the inclusion of uncertain coefficients based on their magnitudes and neighboring structures. Extensive simulations across standard benchmark signals demonstrate that MLWS achieves competitive or superior performance compared to existing shrinkage methods.

## Methods
The proposed method combines classical wavelet shrinkage with machine learning to enhance signal denoising performance. The approach introduces two adaptive thresholds to classify wavelet coefficients and employs machine learning models to refine uncertain regions. The following components are used in this method:

1. **Wavelet Transform:** The input signal is decomposed into multiple resolution levels using the discrete wavelet transform (DWT). This provides a multiscale representation where signal and noise components can be separated more effectively.
   
2. **Dual-Threshold Classification:** Two thresholds (λ₁, λ₂) are used to categorize wavelet coefficients into three groups:

  - Strong coefficients (signal-dominant)
  - Weak coefficients (noise-dominant)
  - Uncertain coefficients (intermediate range)

3. **Machine Learning Integration:** For coefficients in the uncertain zone, classification algorithms (Logistic Regression, SVM, Random Forest, Decision Tree, Neural Network) are used to decide whether to retain or shrink each coefficient based on statistical and contextual features.

4. **Adaptive Parameter Selection:** The optimal thresholds and classifier hyperparameters are selected through grid search and cross-validation to minimize the average mean squared error (AMSE).

5. **Signal Reconstruction:** The denoised signal is reconstructed using the inverse wavelet transform (IDWT), ensuring both smoothness and edge preservation.

## MATLAB Implementation
The following steps explain the procedure used to perform signal denoising using the proposed Machine Learning Wavelet Shrinkage (MLWS) framework.

1. **Load Data:** The test signals (Blocks, Bumps, HeaviSine, and Doppler) are generated or loaded from existing signal libraries in **WaveLab850**.

2. **Wavelet Decomposition:** Each signal is decomposed using a discrete wavelet transform (**dwtr.m**) into multiple resolution levels. The decomposition coefficients are then processed for thresholding.

3. **Threshold Selection:** Two thresholds, λ₁ and λ₂, are defined as functions of clog(n). A grid search over multiple c-values is performed to minimize the Average Mean Squared Error (AMSE) (**WaveletDenoise.m**).

4. **Machine Learning Classification:** Five classifiers; Logistic Regression (**LogisticRegModel.m**), Support Vector Machine (**SVMModel.m**), Random Forest (**RFModel.m**), Decision Tree (**TreeModel.m**), and Neural Network (**NNModel.m**) are used to classify the wavelet coefficients into signal or noise regions.

5. **Signal Reconstruction:** Using the selected coefficients, the signal is reconstructed via the Inverse Discrete Wavelet Transform (**idwtr.m**).

6. **Performance Evaluation:** The denoising performance is evaluated using AMSE under multiple noise levels (SNR = 3, 5, and 7) and compared across classifiers(**Final_code.m**).

## Outcomes

1. The proposed **Machine Learning Wavelet Shrinkage (MLWS)** framework demonstrates improved denoising accuracy across multiple signal types and noise levels.

2. Compared to traditional shrinkage techniques, MLWS achieves lower Average Mean Squared Error (AMSE) through adaptive classifier-based coefficient selection.
