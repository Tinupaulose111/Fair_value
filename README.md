[![CI/CD — API Service](https://github.com/Tinupaulose111/Fair_value/actions/workflows/api-cicd.yml/badge.svg)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/api-cicd.yml)
[![Drift Detection (Weekly)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/drift.yml/badge.svg)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/drift.yml)
[![Model Retrain & Deploy (Triggered by Drift)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/trainer.yml/badge.svg)](https://github.com/Tinupaulose111/Fair_value/actions/workflows/trainer.yml)
# Fair Value Prediction – Production ML System

An end-to-end machine learning system that predicts intrinsic stock value using financial fundamentals, with automated drift detection, retraining, and deployment.


## Business Problem

Retail investors often rely on multiple complex financial ratios (EPS, Book Value, Cash Flow, etc.)
to estimate intrinsic stock value, which increases decision complexity and risk.

This project simplifies the process by learning a **data-driven intrinsic value**
from multiple financial indicators, aligned with **value-investing principles**
to support lower-risk, long-term investment decisions.

## Solution Overview

The system continuously learns from financial data and automatically adapts
to changing data distributions using drift detection and retraining automation.

Key characteristics:
- Multi-factor intrinsic value modeling
- Automated drift detection (KS test)
- Automated retraining and deployment
- Hot model replacement
- Cloud-hosted inference API

  ## System Architecture

*High-level architecture showing data ingestion, drift detection, automated retraining, and deployment on Oracle Cloud VM.*

## Live Demo

🎥 Short demo showing live prediction via browser and automated ML workflow:

🎥 **Live Prediction Demo**
Shows real-time inference via browser using the deployed API:  
▶️ https://youtu.be/ZfAlvFwOjuI

🎥 **Automation & Workflow Demo**
Shows drift detection, automated retraining, and deployment via GitHub Actions:  
▶️ https://youtu.be/AOA4wUh8k90


The diagram below represents the actual production flow implemented in this system.

[![Demo Video](architecture/fairvalue_flowchart.png)](https://youtu.be/ZfAlvFwOjuI)

## Model Development & Validation

The intrinsic value model was developed using a rigorous, multi-stage modeling and statistical validation pipeline, combining traditional financial analysis with machine learning.

Financial data ingestion using Yahoo Finance (yfinance)

Data cleaning and sanity checks on financial ratios

Feature engineering using multiple fundamental indicators

Log and power transformations to stabilize variance

Baseline statistical modeling using OLS regression

Multicollinearity assessment using Variance Inflation Factor (VIF)

K-Fold cross-validation to ensure robustness and generalization

Smearing factor correction to reduce bias from log-transformed targets

Final model training using XGBoost with hyperparameter tuning

Model interpretability using SHAP analysis

Model Performance (Cross-Validated):

R² ≈ 0.82

MAPE ≈ 14%

Metrics are reported using cross-validation to reflect generalization performance rather than single-split results.


## Deployment & Optimization

- Flask-based inference API for real-time prediction
- Gunicorn application server for production-grade concurrency
- Multi-stage Docker builds for optimized container size
- Docker image size reduced from ~2.7 GB to ~1.4 GB
- Faster container startup and reduced cloud resource usage
- Deployed on Oracle Cloud VM
- Hot model replacement via container restart without rebuild

