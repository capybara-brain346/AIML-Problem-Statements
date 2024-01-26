# Procedure To Perform Feature Engineering
1. **Understanding the Problem Statement**: Understand the problem or question you are trying to solve or answer with the dataset.

2. **Data Collection**: Collect the data that you think might help you in solving the problem or answering the question.

3. **Data Cleaning**: Clean the data to handle missing values, remove duplicates, filter out outliers, and correct inconsistent data types.

4. **Feature Extraction**:
    - **Numerical Features**: You can create new features from existing ones. For example, you can create a feature that represents the "total income" of a person by adding their "salary" and "other income".
    - **Categorical Features**: You can convert categorical variables into numerical variables using techniques like one-hot encoding, label encoding, etc.
    - **Date and Time Features**: You can extract information like "day of the week", "month", "year", "hour", etc. from date/time features.
    - **Text Features**: You can use techniques like Bag of Words, TF-IDF, Word Embeddings, etc. to convert text data into numerical features.

5. **Feature Scaling**: Standardize or normalize numerical features if necessary. This is especially important for algorithms that use distance measures, like K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).

6. **Feature Selection**: Use statistical tests, importance measures from machine learning models, or other techniques to select the most important features.

7. **Model Training**: Train your machine learning model using the engineered features.

8. **Evaluation and Iteration**: Evaluate the performance of your model. If it's not satisfactory, you might need to go back and engineer other features, choose different transformations, or try other techniques.

