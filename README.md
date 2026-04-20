# fake-seller-detection
ML fraud detection system for Amazon marketplace using Random Forest &amp; XGBoost in R

![Language](https://img.shields.io/badge/Language-R-blue)
![Model](https://img.shields.io/badge/Models-RandomForest%20%7C%20XGBoost-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview
An end-to-end machine learning system that detects fraudulent sellers and fake reviews on the Amazon marketplace. Developed as part of MBA (Business Intelligence & Data Analytics) coursework at Amity International Business School.

**Business Problem:** Fake reviews and fraudulent sellers erode customer trust and cause revenue loss for e-commerce platforms. This system automates fraud detection using classification models.

## Key Results
| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Forest | 94% | Best on balanced dataset |
| XGBoost | ~90% | Fastest inference time |

## Tech Stack
- **Language:** R
- **Models:** Random Forest, XGBoost
- **Libraries:** dplyr, ggplot2, caret, randomForest, xgboost
- **Visualisation:** ggplot2 fraud probability plots

## Project Structure
```
│── app.R                # Main Shiny Application
│── data/                # CSV Input Files
│── visuals/             # Dashboard Screenshots
│── README.md
```

## How to Run
```
# Install dependencies
install.packages(c("dplyr", "ggplot2", "caret", "randomForest", "xgboost"))

# Run preprocessing
source("scripts/preprocessing.R")

# Train models
source("scripts/model_rf.R")
source("scripts/model_xgb.R")
```

## Visualisations
*Fraud probability score distributions — Random Forest vs XGBoost*

*(Add your ggplot2 output images here)*

## Author
**Dakshayani Verma** | MBA (BI & Analytics) | [LinkedIn](https://linkedin.com/in/dakshayaniverma)
