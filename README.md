This repository contains the code used for my bachelor's thesis and the corresponding arXiv publication, [*ECG for Laboratory Values: A Comprehensive Physiological and Technical Analysis*](https://arxiv.org/abs/2610.28507).

It includes the Python and R scripts used for data preprocessing, evaluation and visualization of the experiments presented in the work.
The scripts used throughout the bachelor thesis were collected, cleaned up and organized into this repository, while preserving the original experimental implementations.

The experiments focus on predicting laboratory values from ECG recordings and patient metadata. NT-proBNP is the primary laboratory value, with additional experiments involving glucose, hemoglobin and potassium.

The model architecture and training pipeline are based on the MDS-ICU framework, available at:

https://github.com/AI4HealthUOL/MDS-ICU

The experiments use data from MIMIC-IV, MIMIC-IV-ECG and MIMIC-IV-ED. These datasets are not included in this repository and must be accessed separately.