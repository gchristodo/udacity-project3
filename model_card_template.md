# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The purpose of this model is to predict whether an employee has an annual income of more than 50k per annum.
We trained a RandomForest Classifier using GridSearch for hyperparameter tuning. The optimal hyperparameters are:
## Intended Use
The intended use of this model is to predict whether an employee has an annual income of more than 50k per annum.
It was created for testing and deployment purposes in scope of Udacity's 3rd Project.
## Training Data
The data can be downloaded from https://archive.ics.uci.edu/dataset/20/census+income. It consists of 14 features (8 categorical, 6 numerical) and the target column (salary), 32561 rows and it doesn't have any missing values. It is a typical binary classification problem, since the target variable consists only of two classes. The target variable is imbalanced, something that it is handled during model implementation via stratify strategy.

The dataset didn't need any complex preprocessing except for some whitespace trimming, OHE for the categorical features and binarization of the target variable. The dataset was split in an 80-20% ratio for training and testing purposes.
## Evaluation Data
We used the test set (20% of the original dataset) to conduct our evaluation. The trained encoder and binarizer were used to transform the categorical features and target variable respectively.
## Metrics
We used precision, recall and fbeta as classification performance metrics. We also calculated the confusion matrix. The results were the following:
    - precision: 0.78
    - recall: 0.642
    - fbeta: 0.704
    - Confusion matrix: [[4661,  284], [ 560, 1008]]

## Ethical Considerations
According to Tai Le Quy et all (https://arxiv.org/pdf/2110.00530.pdf) this dataset isn't fairness-aware.
## Caveats and Recommendations
Extraction was done by Barry Becker from the 1994 Census database. Thus it is outdated and it doesn't accurately represent the current status quo of the population.
However it can be used for training purposes in ML classification problems.