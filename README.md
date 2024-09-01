# Predicting Student Performance in Diagnostic Assessments Using Machine Learning

This repository contains the final project for **CSC311 Winter 2024**, focusing on predicting student performance in diagnostic assessments using various machine learning techniques. This project was designed my Professor Alice Gao, completed by **Jiaxi Li**, **Jiawei Gong**, and **Nuo Xu**.

## Project Overview

Online education platforms often struggle to measure students' understanding of course material. To address this, many platforms use diagnostic questions to assess students' knowledge. Our project aims to predict whether a student will correctly answer a diagnostic question based on their previous answers and the responses of other students.

We implemented several machine learning algorithms, modified existing models to improve prediction accuracy, and analyzed their performance in this context.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Implemented Algorithms](#implemented-algorithms)
- [Modifications and Improvements](#modifications-and-improvements)
- [Results](#results)
- [How to Run](#how-to-run)
- [Contributors](#contributors)

## Data

The dataset was obtained from the Eedi platform and includes responses from 542 students to 1,774 diagnostic questions. The data is divided into three primary files:

- **`train_data.csv`**: The training dataset.
- **`valid_data.csv`**: The validation dataset.
- **`test_data.csv`**: The test dataset.
- **`train_sparse.npz`**: A sparse matrix representing the training data.

Additionally, metadata files provide more context about the questions:

- **`question_meta.csv`**: Contains information about the questions.
- **`student_meta.csv`**: Contains demographic information about the students.

## Implemented Algorithms

### 1. Collaborative Filtering with k-Nearest Neighbor (kNN)
We implemented both user-based and item-based collaborative filtering to predict students' responses. The best performance was achieved with user-based collaborative filtering with **k=11**, yielding a test accuracy of **68.42%**.

### 2. Item Response Theory (IRT)
We implemented a one-parameter IRT model to predict the likelihood of a correct response based on the student's ability and question difficulty. Our model achieved a test accuracy of **70.84%**.

### 3. Matrix Factorization
We used Singular-Value Decomposition (SVD) and Alternating Least Squares (ALS) with Stochastic Gradient Descent (SGD) to factorize the sparse matrix of student responses. The ALS model with regularization achieved a test accuracy of **69.91%**.

## Modifications and Improvements

In Part B of the project, we modified the ALS model to include regularization and switched from SGD to mini-batch gradient descent. These modifications aimed to reduce overfitting and improve convergence speed.

## Results

The modifications led to marginal improvements in accuracy. The best-performing model was the regularized ALS with a mini-batch size of 8, achieving a test accuracy of **69.91%**. However, the enhancements primarily improved the model's stability rather than significantly boosting accuracy.

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JacquelineLee028/MachineLearningProject.git
   cd MachineLearningProject
   Certainly! Here’s the "How to Run" section formatted properly in Markdown:
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the algorithms**:
   - For kNN:
     ```bash
     python part_a/knn.py
     ```
   - For IRT:
     ```bash
     python part_a/item_response.py
     ```
   - For Matrix Factorization:
     ```bash
     python part_a/matrix_factorization.py
     ```
4. **Explore the results in the `output/` directory.**

## Contributors

- **Nuo Xu**: Implemented the Collaborative Filtering model and contributed to the modifications in Part B.
- **Jiaxi Li**: Implemented the Matrix Factorization model and contributed to the modifications in Part B.
- **Jiawei Gong**: Implemented the Item Response Theory model.


