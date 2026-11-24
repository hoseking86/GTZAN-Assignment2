
# Machine Learning and AI - GTZAN Dataset Music Genre Classification

Development environment: Google Colab
Python version: 3.13.18

## Project Overview

This project was created to accompany the Machine Learning and AI Assignment 2.

The requirements of the project involved the implementation and accompanying documentation to compare two chosen machine learning classification models.

For this project, the two models chosen were Decision Tree (DT) and Random Forest (RF). The models were implemented to help determine how accurately they could classify music genres using extracted numerical audio features.

In order to implement the models, the process was carried out using Python, Google Colab and scikit-learn.

## Dataset

The GTZAN dataset comprises 1,000 audio recordings across ten music genres, with each recording lasting approximately 30 seconds.

Following the feature extraction process, 999 recordings were successfully processed.

The features were extracted using Librosa and include tempo, spectral features, chroma and Mel-frequency cepstral coefficients (MFCCs).

The dataset was split into 70% training data and 30% testing data using stratified sampling to maintain similar genre proportions.

The dataset is not included in the repository and must be obtained and loaded separately before running the notebook.

## Machine Learning Models

The two classification models implemented for this project were:

- Decision Tree (DT)
- Random Forest (RF)

Both models were implemented using scikit-learn pipelines, including SimpleImputer as a precautionary preprocessing step.

## Model Optimisation

Five-fold stratified cross-validation was used to evaluate the initial model configurations.

GridSearchCV was used to investigate different hyperparameter combinations and identify the best-performing settings for each model.

Learning curves were also generated to investigate training and validation accuracy and evaluate potential overfitting.

## Testing and Evaluation

The optimised models were evaluated using the held-out 30% test dataset.

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrices
- 95% Wilson confidence intervals

The final testing results showed:

- Decision Tree (DT): Test accuracy of 49.33%
- Random Forest (RF): Test accuracy of 56.00%

## Requirements

The Google Colab project was developed and tested using:

- Google Colab
- Python 3.13.18
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Librosa
- SciPy
- Joblib
- tqdm

The specific library versions are recorded in `requirements.txt`.

To install the required libraries, run the following command:

    pip install -r requirements.txt

## Running the Notebook

1. Open the project notebook `GTZAN_Assignment2.ipynb` in Google Colab.
2. Ensure the GTZAN dataset is stored in an accessible location.
3. Update the dataset file path if required.
4. Mount Google Drive when prompted.
5. Run the notebook cells in the order they have been structured.
6. Review the model optimisation, learning curves and final testing results.

The notebook was developed and tested using the following dataset location:

    /content/drive/MyDrive/datasets/music_genre

## Reproducibility

Both models were implemented using a fixed random state of 42 and the same stratified training and testing split to support reproducibility.

Five-fold stratified cross-validation was used during model evaluation and optimisation. The held-out test dataset remained separate until final testing.

The `requirements.txt` file records the Python library versions used during development.

The Google Colab notebook was tested in a fresh Colab environment. However, results may vary slightly if the project is executed using different library versions or computing environments.
