[![CI/CD — API Service](https://github.com/Tinupaulose111/Fair_value/actions/workflows/api-cicd.yml/badge.svg)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/api-cicd.yml)
[![Drift Detection (Weekly)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/drift.yml/badge.svg)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/drift.yml)
[![Model Retrain & Deploy (Triggered by Drift)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/trainer.yml/badge.svg)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/trainer.yml)
# Fair Value Prediction – Production ML System

An end-to-end machine learning system that predicts intrinsic stock value using financial fundamentals, with automated drift detection, retraining, and deployment.
## Business Problem

Retail investors often rely on complex financial ratios (EPS, Book Value, Cash Flow, etc.) to evaluate stocks.
This project simplifies decision-making by learning a data-driven intrinsic value from multiple financial indicators,
aligned with value-investing principles to reduce downside risk.

## System Architecture

The system follows a continuous ML lifecycle:
Data ingestion → Feature engineering → Model training → Drift detection → Automated retraining → Deployment → Live inference

api/            → Flask API for inference
trainer/        → Training & retraining logic
drift/          → Drift detection scripts
core/           → Shared preprocessing logic
model/          → Active model artifact
.github/workflows/ → CI/CD automation

Data Collection
      ↓
Data Cleaning & Feature Engineering
      ↓
Model Training & Evaluation
      ↓
Drift Detection (Scheduled)
      ↓
Retraining Triggered (if drift)
      ↓
New Model vs Current Model Comparison
      ↓
   deploy
      ↓
Model Copied to VM
      ↓
API Container Restart
      ↓
Live Predictions with New Model
