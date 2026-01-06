import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the data
train_df = pandas.read_csv("Corona_NLP_train.csv", encoding='latin-1')
test_df = pandas.read_csv("Corona_NLP_test.csv")

# Data preprocessing
train_df = train_df.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1)

# Prepare sentences and labels
sentences = train_df['OriginalTweet'].tolist()
labels = train_df['Sentiment'].tolist()

# Label Encoding
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Create the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto'))
])

# Train the pipeline
pipeline.fit(sentences, labels_encoded)

# Make predictions on test data
test_sentences = test_df['OriginalTweet'].tolist()
predictions_encoded = pipeline.predict(test_sentences)

# Decode predictions back to original labels
predictions = label_encoder.inverse_transform(predictions_encoded)

# Evaluate the pipeline
X_train, X_test, y_train, y_test = train_test_split(sentences, labels_encoded, test_size=0.3, random_state=42)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Predictions:", predictions)

# Save the pipeline and label encoder
with open('sentiment_pipeline.pkl', 'wb') as f:
    pickle.dump((pipeline, label_encoder), f)

# Load the pipeline and label encoder
with open('sentiment_pipeline.pkl', 'rb') as f:
    loaded_pipeline, loaded_label_encoder = pickle.load(f)

# Make predictions using loaded pipeline
loaded_predictions_encoded = loaded_pipeline.predict(test_sentences)
loaded_predictions = loaded_label_encoder.inverse_transform(loaded_predictions_encoded)
print("Loaded predictions:", loaded_predictions)