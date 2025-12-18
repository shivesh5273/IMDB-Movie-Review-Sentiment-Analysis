# IMDB Movie Review Sentiment Analysis

This project applies classic machine learning techniques to predict whether a movie review from the IMDB dataset is **positive** or **negative**. The goal is to build a complete, end-to-end text classification pipeline and understand each step, not just train a model.

## What this project does

- Starts from raw IMDB review text files and combines them into a single cleaned CSV file.  
- Cleans the text using regular expressions, removes HTML tags, keeps useful emoticons, and normalizes the text.  
- Converts text into numerical features using Bag-of-Words and TF–IDF representations.  
- Trains a regularized logistic regression model and tunes its hyperparameters with 5-fold stratified cross-validation.  
- Experiments with out-of-core learning (mini-batches streamed from disk) to handle large datasets efficiently.  
- Uses Latent Dirichlet Allocation (LDA) to discover topic groups inside the movie reviews.

## Key ideas covered

- Text preprocessing for NLP (cleaning, tokenization, stemming, stop-word handling).  
- Feature extraction with Bag-of-Words and TF–IDF.  
- Regularization and model selection via grid search and cross-validation.  
- Out-of-core learning using incremental training.  
- Topic modelling with LDA on a bag-of-words matrix.

## Results (high-level)

- The tuned logistic regression model achieves around **90% accuracy** on a held-out test set of IMDB reviews.  
- The out-of-core setup with streamed mini-batches is slightly less accurate (around mid-80% accuracy) but much more memory-efficient and suitable for larger datasets.  
- LDA reveals interpretable topics such as family-related stories, horror themes, crime/police plots and cinema/acting-related themes.

Overall, this project shows how far we can go with “classical” machine learning and careful feature engineering for text before even touching deep learning or transformer-based models.
