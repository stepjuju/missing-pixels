# Image Pixelation and Reconstruction
**Status:** 🚧 in progress 🚧

This project idea came as an extension of a Python II course project at Johannes Kepler University (JKU), aimed at exploring advanced image processing techniques. The primary objective is to pixelate images and then use a trained CNN model to reconstruct the missing pixels.

## Project Overview
For training models on 64x64 pixels images dataset from the course was used. \
For training models on 128x128 pixels images dataset from https://press.liacs.nl/mirflickr/mirdownload.html was used.

### Current Progress

#### Part 1: Model Training

The first part focuses on training models using the UNet architecture.
- **64x64 images**: The images are pixelated and then reconstructed by the trained model.
- **128x128 images**: Train the model on images of various sizes

The model predicts missing pixels in pixelated images, demonstrating the effectiveness of the UNet architecture for this task. Performance metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) are used to evaluate the model's performance.

### Future Work

#### Part 1: Model Training
- **Arbitrary size of the images** Train a model on arbitrary sized images.
  
#### Part 2: Web Application

The second part aims to create a user-friendly web application with the following features:
- **Image Upload**: Users can upload their own images.
- **Random Pixelation**: The uploaded image will be randomly pixelated.
- **Model Prediction**: The trained UNet model will predict the missing pixels in the pixelated image.
- **Performance Evaluation**: Display the performance metrics (MSE, PSNR, SSIM) to assess the reconstruction quality.
