# Image Pixelation and Reconstruction

This project idea came as an extension of a Python II course project at Johannes Kepler University (JKU), aimed at exploring advanced image processing techniques. The primary objective is to develop a system that can pixelate images and then use a trained UNet model to reconstruct the missing pixels.

## Project Overview

### Current Progress

#### Part 1: Model Training

The first part focuses on training models using the UNet architecture.
For image resolution of 64x64 pixels:
- **64x64 images**: The images are pixelated and then reconstructed by the trained model.

The model predicts missing pixels in pixelated images, demonstrating the effectiveness of the UNet architecture for this task. Performance metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) are used to evaluate the model's performance.

Currently, a model for 64x64 pixel images has been trained using a rented instance from vast.ai.

### Future Work

#### Part 1: Model Training
- **128x128 images**: Collect data and train the model on 128x128 pixel images
- 
#### Part 2: Web Application

The second part aims to create a user-friendly web application with the following features:
- **Image Upload**: Users can upload their own images.
- **Random Pixelation**: The uploaded image will be randomly pixelated.
- **Model Prediction**: The trained UNet model will predict the missing pixels in the pixelated image.
- **Performance Evaluation**: Display the performance metrics (MSE, PSNR, SSIM) to assess the reconstruction quality.
