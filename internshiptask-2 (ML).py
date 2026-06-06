import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


data = {
    "Review": [
        "I love this product",
        "Excellent service",
        "Very good quality",
        "Amazing experience",
        "Best purchase ever",
        "I hate this item",
        "Worst product",
        "Very bad service",
        "Not satisfied",
        "Terrible experience"
    ],

    "Sentiment": [
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative"
    ]
}
df=pd.DataFrame(data)
print()
print()

print("DATASET:\n")
print(df)

# INPUT AND OUTPUT
X=df['Review']
Y=df['Sentiment']

# TF-IDF VECTORIZATION converting text into numerical
tfidf=TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)
print()
print("\nTF-IDF Matrix Shape:")
print(X_tfidf.shape)

# SPLIT TRAINING AND TESTING DATA
X_train,X_test,Y_train,Y_test=train_test_split(X_tfidf,Y,
                                      test_size=0.3,random_state=42)


# STATISTICAL MODEL
# LOGISTIC REGRESSION
#works with numbers
model=LogisticRegression()

model.fit(X_train,Y_train)

# PREDICTION
y_pred=model.predict(X_test)

print("Predicted Model is: ")
print(y_pred)
print()

# ACCURACY
print("Accuracy_score is:")
accuracy=accuracy_score(Y_test,y_pred)
print(accuracy)

print()

# CLASSIFICATION REPORT
print("Classification Report:")
print(classification_report(Y_test, y_pred, zero_division=0))

print()

# CONFUSION MATRIX
print("Confusion_matrix:")
print(confusion_matrix(Y_test,y_pred))

