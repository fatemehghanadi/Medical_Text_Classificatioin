# Medical_Text_Classificatioin
Medical Text Classification Project 🩺📚 Explore efficient methods for classifying medical texts into predefined categories using both machine learning (e.g., SVM, Random Forest) and deep learning (e.g., 1D CNN, pretrained BERT models). Includes a unique dataset, performance comparisons, and a custom 1D CNN for optimized results.


This repository contains a project focused on classifying medical text into various predefined categories. The project was developed as part of an advanced information retrieval course.

## Project Overview

Text classification in the medical field plays a crucial role in organizing large datasets and providing quick access to critical information for researchers, doctors, and other health professionals. This project explores methods for medical text classification using both traditional machine learning and deep learning techniques.

## Dataset

The dataset consists of summaries of medical texts, each assigned to one of the following categories:
1. Neoplasms
2. Digestive System Diseases
3. Nervous System Diseases
4. Cardiovascular Diseases
5. General Pathological Conditions

The dataset contains 11,550 training samples and 2,888 test samples, for a total of 14,438 records. Data preprocessing involved converting text into numerical arrays while preserving the structure of the sentences.

## Methods and Models

The project evaluates supervised and unsupervised text classification methods:

### Traditional Machine Learning Models
- Logistic Regression
- Support Vector Machines (SVM)
- Naive Bayes
- Random Forest
- Gradient Boosting
- AdaBoost
- Ensemble Models

### Deep Learning Models
#### Proposed 1D Convolutional Network
A custom 1D Convolutional Neural Network (CNN) was designed for this project. Key features include:
- Preprocessing to convert sentences into arrays of numerical values.
- Use of convolutional layers to combine local information and classify the data.

Performance:
- F1 Score: 60.18%
- Total Parameters: 1.8M

#### Pretrained BERT Models
- **`bert_en_uncased`**: Achieved an accuracy of 48%.
- **`Bio_ClinicalBERT-finetuned-medicalcondition`**: Fine-tuned for medical conditions, achieved an accuracy of 63.46%.
- **`MedicalArticlesClassificationModelMultiLabel`**: Fine-tuned for multi-label classification, achieved an accuracy of 63.50%.

## Results

| Model                                    | F1 Score (%) |
|------------------------------------------|--------------|
| Logistic Regression                      | 52.5         |
| SVM                                      | 50.2         |
| Naive Bayes                              | 48.7         |
| Random Forest                            | 53.8         |
| Gradient Boosting                        | 52.3         |
| AdaBoost with Random Forest              | 49.6         |
| Proposed 1D Convolutional Network        | 60.18        |
| `Bio_ClinicalBERT-finetuned-medicalcondition` | 63.46   |
| `MedicalArticlesClassificationModelMultiLabel` | 63.50   |


## Confusion Matrix

The confusion matrix below represents the performance of the **MedicalArticlesClassificationModelMultiLabel** on the test dataset. Each row corresponds to the true class, and each column represents the predicted class.

|              | Predicted Class 0 | Predicted Class 1 | Predicted Class 2 | Predicted Class 3 | Predicted Class 4 |
|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| **Actual 0** | 0.83               | 0.03                | 0.039                | 0.017                | 0.084                |
| **Actual 1** | 0.15                | 0.65               | 0.027                 | 0.03                 | 0.14                |
| **Actual 2** | 0.088                | 0.029                | 0.64               | 0.073                | 0.17                |
| **Actual 3** | 0.028                 | 0.013                | 0.048                | 0.79               | 0.12                |
| **Actual 4** | 0.17               | 0.12               | 0.12               | 0.19               | 0.4               |

The confusion matrix highlights the model's strengths in accurately predicting some classes, while showing areas where predictions for other classes could be improved. This provides a clear view of class-wise performance for multi-label classification.


## Conclusion

The proposed 1D CNN demonstrated competitive performance compared to more parameter-heavy models like BERT, achieving substantial accuracy with only 1.8 million parameters. Traditional machine learning models also provided reasonable performance with much faster training times.

## References
1. Schopf, T., Braun, D., & Matthes, F. (2022). Evaluating unsupervised text classification: zero-shot and similarity-based approaches. arXiv preprint [arXiv:2211.16285](https://arxiv.org/abs/2211.16285).
2. Dataset: [Medical Abstracts TC Corpus](https://github.com/sebischair/medical-abstracts-tc-corpus)
3. Pretrained BERT Models: Available on [Hugging Face](https://huggingface.co).
