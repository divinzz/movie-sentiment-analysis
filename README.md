# IMDB Movie Review Sentiment Analysis with LSTM

This repository contains a sentiment analysis model that uses Long Short-Term Memory (LSTM) networks to classify IMDB movie reviews as positive or negative. The dataset used in this project contains 50,000 movie reviews, with an equal distribution of positive and negative sentiment labels. The goal is to predict the sentiment of unseen reviews by leveraging Natural Language Processing (NLP) techniques.

## Project Overview

In this project, we build and train a sentiment analysis model using a Recurrent Neural Network (RNN), specifically the LSTM architecture, which is known for handling sequential data like text effectively. The IMDB dataset used contains two primary columns: `review` (the text of the review) and `sentiment` (the label, either positive or negative).

### Key Steps in the Project:

1. **Data Preprocessing**: 
   - The dataset is loaded from a CSV file and cleaned to ensure it's ready for model training.
   - Text is tokenized using the Keras `Tokenizer`, converting the words into numerical sequences. Padding is applied to ensure that all sequences have the same length.
   - Sentiment labels are converted into binary format, where `positive` is labeled as 1 and `negative` as 0.

2. **Exploratory Data Analysis (EDA)**: 
   - A WordCloud visualization is generated to highlight the most frequent words in positive reviews.
   - A histogram of review lengths is plotted to analyze the distribution of review sizes.
   - Class distribution is visualized using a count plot, confirming an even split between positive and negative reviews.

3. **Model Building**: 
   - An LSTM model is built using the Keras library. The architecture consists of an embedding layer, an LSTM layer for sequence learning, and a dense output layer with a sigmoid activation function to classify the sentiment.
   - The model is compiled with the Adam optimizer and binary cross-entropy loss function, which is suitable for binary classification tasks.

4. **Model Training**: 
   - The model is trained for 5 epochs with a batch size of 64, using a validation split of 0.2.
   - During training, accuracy and loss metrics are monitored for both training and validation datasets.

5. **Evaluation and Visualization**:
   - After training, the model's performance is evaluated on the test set. The evaluation includes calculating the test loss and accuracy.
   - The training process and evaluation results are visualized using accuracy and loss plots for both training and validation sets.
   - A confusion matrix and Receiver Operating Characteristic (ROC) curve are plotted to assess the model's classification performance.

6. **Prediction System**: 
   - A function is implemented to predict the sentiment of a new review. This function tokenizes and pads the review text before passing it through the trained model. The sentiment (positive or negative) is returned based on the model's prediction.

## Results

- The model achieved a test accuracy of approximately **87.8%**, indicating its ability to effectively classify sentiments in movie reviews.
- The confusion matrix and ROC curve also show that the model performs well with balanced false positives and true positives, reflecting its overall reliability in sentiment classification.

## How to Use the Model

1. **Training**: 
   - Clone the repository and run the provided code to train the model on the IMDB dataset. The model is trained for 5 epochs, but feel free to experiment with different configurations.

2. **Prediction**:
   - To use the trained model for sentiment prediction, simply call the `predict_sentiment` function with a new review. The model will return whether the sentiment is `positive` or `negative`.

3. **Model Saving and Loading**:
   - After training, the model is saved to a file (`_trained_model.h5`), and you can reload it later using Keras' `load_model` function for inference or further training.

## Conclusion

This project demonstrates how to use LSTM networks to perform sentiment analysis on text data. It provides a useful foundation for building NLP applications that can classify sentiment in reviews, social media posts, or any other text-based datasets.
