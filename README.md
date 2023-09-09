# Kvasir GI image classification using MobileOne architecture in PyTorch lightning framework

This repository contains code for the Kvasir Gastrointestinal (GI) tract image classification task. The goal is to get familiarized with Lightning framework and mobileOne architecture.

## Workflow

**1. Dataset details:** The Kvasir dataset consists of images, annotated and verified by medical doctors (experienced endoscopists), including several classes showing anatomical landmarks, phatological findings or endoscopic procedures in the GI tract. The anatomical landmarks include Z-line, pylorus, and cecum, while the pathological finding includes esophagitis, polyps, and ulcerative colitis. In addition, there are another two class images related to removal of lesions, "dyed and lifted polyp" and the "dyed resection margins". The dataset consist of the images with different resolution from 720x576 up to 1920x1072 pixels and organized in a way where they are sorted in separate folders named accordingly to the content. Some of the included classes of images have a green picture in picture illustrating the position and configuration of the endoscope inside the bowel, by use of an electromagnetic imaging system (ScopeGuide, Olympus Europe) that may support the interpretation of the image.

**2. Loading the dataset:** I have unzipped the data from my google drive as I have stored the zip file (kvasir-v2.zip) in my google drive. In the data folder, each subfolder represents each class.

**3. EDA:**: To check the class distribution, I counted the number of images in each class within the Kvasir GI dataset. I found that there are 8 Classes and there are 1000 images for each of the clas. I showed that by plotting a bar chart. I have also displayed sample images from each class in a grid. The classes are the following:

- dyed-lifted-polyps
- dyed-resection-margins
- esophagitis
- normal-cecum
- normal-pylorus
- normal-z-line
- polyps
- ulcerative-colitis

**4. Creating Data Module:**: I have created the data module class which is responsible for managing the Kvasir GI dataset, including preprocessing and splitting into training, validation, and test sets. I initialized the data module by specifying essential parameters such as the data directory, batch size, and random seed. Then I defined a set of data transformations for the training dataset, including resizing images, random rotations, flips, color jittering, converting to tensors, and normalization. For the validation and test datasets, I defined a separate set of transformations, which includes resizing, converting to tensors, and normalization. These transformations ensure that the validation and test data are prepared consistently. I split the Kvasir GI dataset into training, validation, and test subsets while maintaining a fixed random seed for reproducibility. I have taken 80% of the total samples(8000)for training, 10% for validation, and 10% for testing. I have applied the defined data transformations to the respective subsets. I have created dataloader for training, validation, and test. I have also defined method to calculate and store the number of images in each class for the training, validation, and test subsets.

**5. Model Building:** I have constructed image classification model within the PyTorch Lightning Module called ImageClassifier. This module is responsible for defining the model architecture, specifying metrics, and handling training and evaluation steps. I initialized the model by selecting the MobileOne architecture with s3 and s4 variant from the timm library. The forward method in the ImageClassifier module is where I defined the forward pass of the model. Given an input image, the forward pass computes the model's predictions (logits) for each class. These logits will later be used to compute the loss and evaluate the model's performance.

In the training_step method, I specified the logic for a single training step.
This includes passing a batch of training data through the model, computing the loss (cross-entropy), and logging training-related metrics. Metrics logged during training include training loss and training accuracy. Within the validation_step method,the validation loss and validation accuracy are computed and logged. In test_step, the model's performance is evaluated on a held-out test dataset. Test loss, test accuracy, top-k accuracy, and F1 score are computed and logged. I configured the optimizer used for training, typically using an Adam optimizer. The learning rate is set based on the specified hyperparameters. The speed of training was also fast with these models on colab GPU.

**6. Model training,validation and WandB integration:** After initializing datamodule and classification model using the ImageClassifier class, I initialize Wandb logger. I configured a model checkpoint callback named checkpoint_callback to save models during training. Then, I initialized a PyTorch Lightning trainer named trainer for training and evaluation. The accelerator is set to "gpu" to leverage GPU acceleration if available.
I enabled mixed-precision training by setting the precision to "16-mixed". This speeds up training and Reduces GPU Memory Usage. Then I used trainer.fit to train the model for the specified number of epochs. The trained model will be saved based on the best validation accuracy achieved during training.

**7. Model Evaluation on test data:** Then I used trainer.test on test data and returned accuracy, Top 3 Accuracy, Top 5 accuracy, F1 score. I also plotted the confusion matrix.

**8. Result:**

| Model        | Accuracy | Top 3 Accuracy | Top 5 Accuracy | F1 score |
| ------------ | -------- | -------------- | -------------- | -------- |
| mobileone_s3 | 91.625%  | 99.75%         | 100%           | 0.91625  |
| mobileone_s4 | 91.875%  | 99.875%        | 100%           | 0.91875  |

## Conclusion:

This project demonstrates the entire workflow of building a deep learning model in PyTorch Lightning framework. The achieved accuracy on the testing dataset reflects the models capability to generalize and classify images accurately. I have compared accuracy of two variants of mobileOne model (s3 and s4). Among the models, mobileone_s4 achieved slighlty more accuracy than s3 variant. But, s3 variant has less parameters. So, it is more memory efficient.

Note: I have added code feature extraction. I will try to perform traditional ml techniques on extracted feature.
