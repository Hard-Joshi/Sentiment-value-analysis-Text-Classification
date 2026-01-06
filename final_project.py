import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Load the pre-trained pipeline and label encoder
try:
    with open('sentiment_pipeline.pkl', 'rb') as f:
        loaded_pipeline, loaded_label_encoder = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'sentiment_pipeline.pkl' not found. Please ensure the model has been trained and saved.")
    st.stop()

def predict_sentiment(text):
    """Predicts the sentiment of a given text using the loaded pipeline."""
    if loaded_pipeline and loaded_label_encoder:
        prediction_encoded = loaded_pipeline.predict([text])
        prediction = loaded_label_encoder.inverse_transform(prediction_encoded)[0]
        return prediction
    else:
        return "Model not loaded."

def main():
    st.title("Sentiment Analysis of Tweets")
    st.subheader("Predict the sentiment of a given text related to the coronavirus pandemic.")

    user_input = st.text_area("Enter your tweet/text here:", "")

    if st.button("Predict Sentiment"):
        if user_input:
            sentiment = predict_sentiment(user_input)
            st.write(f"**Predicted Sentiment:** {sentiment}")
        else:
            st.warning("Please enter some text to analyze.")

    st.sidebar.header("About")
    st.sidebar.info(
        "This application performs sentiment analysis on text data related to the coronavirus pandemic."
        " It uses a pre-trained Logistic Regression model with TF-IDF vectorization."
    )

if __name__ == "__main__":
    main()