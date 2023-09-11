# Kvasir GI image Feature extraction, Classification and Clustering

This repository contains notebooks for feature extraction process on the Kvasir Gastrointestinal (GI) tract image data. Feature extraction was done in two ways. Using the extracted features I have performed classification task and clustering analysis.


## Workflow

**1. Dataset details:** The Kvasir dataset consists of images, annotated and verified by medical doctors (experienced endoscopists), including several classes showing anatomical landmarks, phatological findings or endoscopic procedures in the GI tract. The anatomical landmarks include Z-line, pylorus, and cecum, while the pathological finding includes esophagitis, polyps, and ulcerative colitis. In addition, there are another two class images related to removal of lesions, "dyed and lifted polyp" and the "dyed resection margins". The dataset consist of the images with different resolution from 720x576 up to 1920x1072 pixels and organized in a way where they are sorted in separate folders named accordingly to the content. Some of the included classes of images have a green picture in picture illustrating the position and configuration of the endoscope inside the bowel, by use of an electromagnetic imaging system (ScopeGuide, Olympus Europe) that may support the interpretation of the image.

**2. Loading the dataset:** I have unzipped the data from my google drive as I have stored the zip file (kvasir-v2.zip) in my google drive. In the data folder, each subfolder represents each class.

**3. Feature Extraction:** I have performed feature extraction in two following ways. After creating the mobileone_s4 model from timm library, I removed the final classification layer. After extracting the feature, I saved it as pickle file containing feature and label of each image.

* **MobileOne architecture with my best trained weights:** Here, I used the base mobileOne model with pretrained imagenet weights for extracting features.

* **MobileOne architecture with imagenet weights:** During the training with mobileOne_s4, I saved the best checkpoint based on val accuracy. I load the weights from the checkpoint for this feature extraction task.

**4. Classification and Clustering:** First, I loaded feature files. This part of the task was performed in following three ways. For classifcation, I have used SVM, KNN, Random Forest, Adaboost, and LightGBM. I also used K-means clustering and demonstrated the elbow plot.

* **Using the feature file generated from mobileone_s4 and my best trained weights:** 
* **Using the feature file generated from mobileone_s4 and imagenet weights:**
* **Concatenating feature files generated from MobileOne_s4, Efficientvit_m3, and Fastvit_t8:** 

**5. Result:**

**Using the best weights achieved after training with mobileOne_s4:**

| Model           | Accuracy   | F1 Score   |
| --------------- | ---------- | ---------- |
| SVM             | 0.978333   | 0.978342   |
| LightGBM        | 0.975833   | 0.975844   |
| KNN             | 0.975833   | 0.975854   |
| Random Forest   | 0.976666   | 0.976679   |
| AdaBoost        | 0.957500   | 0.957531   |

**Using the imagenet weights:**

| Model           | Accuracy   | F1 Score   |
| --------------- | ---------- | ---------- |
| SVM             | 0.889166   | 0.888754   |
| LightGBM        | 0.876666   | 0.876485   |
| KNN             | 0.741666   | 0.736347   |
| Random Forest   | 0.840833   | 0.840499   |
| AdaBoost        | 0.590833   | 0.578524   |

**Using the concatenated feature files generated from MobileOne_s4,FastVit_t8, and EfficientVit_m3:**

| Model           | Accuracy   | F1 Score   |
| --------------- | ---------- | ---------- |
| SVM             | 0.985833   | 0.985837   |
| LightGBM        | 0.977500   | 0.977508   |
| KNN             | 0.982500   | 0.982485   |
| Random Forest   | 0.975833   | 0.975845   |
| AdaBoost        | 0.956666   | 0.956749   |

## Conclusion:

This project demonstrates the entire workflow of extracting feature and performing classification and clustering using ML models and using those features. I have observed that I got much higher accuracy with the features those are generated using my best trained weights than imagenet weights on mobileone_s4. After comparing all three results, I found more improved accuracy with concatenated features. In all cases, SVM achieved higher accuracy.
