# Intrusion Detection Using Ensemble Techniques

A multi-class network intrusion detection system built on the NSL-KDD dataset, using a Maximum Voting Ensemble that combines Gaussian Naive Bayes, a Decision Tree, and XGBoost.

## Overview

This project classifies network traffic into one of five categories: Normal, DoS (Denial of Service), Probe, R2L (Remote to Local), or U2R (User to Root). Rather than relying on a single classifier, three different algorithms vote on each prediction, with the majority vote taken as the final classification.

## Approach

- **Dataset**: NSL-KDD, an improved version of the original KDD Cup 99 intrusion detection dataset, addressing known duplicate-record issues in the original
- **Classification**: multi-class, grouping the dataset's ~40 specific attack types into 5 standard categories
- **Feature encoding**: categorical fields (protocol type, service, flag) label-encoded; all features standardized
- **Ensemble**: Maximum Voting Ensemble combining Gaussian Naive Bayes, a Decision Tree, and XGBoost, using hard (majority) voting

## Results

- **99.93% accuracy**
- **0.07% false positive rate** (normal traffic incorrectly flagged as an attack)

## Tech Stack

- Python
- Pandas, Scikit-learn, XGBoost

## Files

- `intrusion_detection.py` — full pipeline: data loading, preprocessing, ensemble training, and evaluation
- `KDDTrain.txt`, `KDDTest.txt` — the NSL-KDD dataset files

## Running the Project

```bash
pip install -r requirements.txt
python intrusion_detection.py
```

## Author

Aishwarya Ramanath Shanbhag
