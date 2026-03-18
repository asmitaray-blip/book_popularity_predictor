# Book Popularity Predictor

Overview

This project develops a machine learning pipeline to classify books into popularity categories using a combination of textual and numerical features. The model leverages book titles processed through TF-IDF alongside structured metadata such as ratings and review counts.

The objective is to demonstrate how heterogeneous data (text + tabular features) can be integrated to build an effective classification system.


Dataset

* Source: Goodreads Books Dataset (Kaggle)
* The dataset contains metadata for books, including ratings, reviews, and publication details.

Features Used

* `title` (text feature)
* `average_rating`
* `ratings_count`
* `text_reviews_count`
* `num_pages`

Problem Formulation

The original dataset does not include explicit popularity labels. Therefore, a target variable was engineered based on `ratings_count`:

* Low: < 10,000
* Medium: 10,000 – 50,000
* High: > 50,000

This converts the problem into a multi-class classification task.

Methodology

Data Preprocessing

* Selected relevant columns
* Renamed inconsistent column headers
* Removed missing values to ensure data integrity

Feature Engineering

Text Representation

* Applied TF-IDF vectorization on book titles
* Limited to the top 500 features to control dimensionality

Numerical Features

* Directly used structured attributes such as ratings and review counts

#### Feature Integration

* Combined sparse TF-IDF features with dense numerical features using horizontal stacking (`hstack`)

---

## Model

### Random Forest Classifier

* Number of estimators: 100
* Random state fixed for reproducibility

Random Forest was selected due to its robustness, ability to handle mixed feature types, and resistance to overfitting.

---

## Evaluation

The model was evaluated using:

* Accuracy Score
* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix

These metrics provide both overall performance and class-wise prediction quality.

---

## Visualization

* Confusion Matrix to analyze classification performance
* Popularity distribution across classes
* Feature importance derived from the Random Forest model


## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

Limitations

* The target variable is derived from `ratings_count`, which is also used as an input feature, potentially introducing leakage.
* Titles alone may not capture full semantic context compared to full descriptions.

Future Work

* Replace title-based features with full book descriptions
* Remove data leakage by excluding `ratings_count` from input features
* Perform hyperparameter tuning
* Evaluate advanced models such as Gradient Boosting or XGBoost
* Address potential class imbalance

## Author

Asmita Ray
