# Coding exercise

> This is a coding exercise for the application to a data science position at [**Emadri**](www.emadri.com) as part of an internship / part-time opportunity (see [www.emadri.com/careers](http://www.emadri.com/careers) for details).
> * The guidelines below are kept at a high-level - the actual implementation is up to you.
> * We value both computing performance (robustness and speed of code) and human efficiency (how long it took you to implement it).

 The work must be completed in a Python notebook (`*.ipynb`) and an accompanying Python module with any relevant helper functions.  All code should be submitted to your personal git repository (github, BitBucket, ...).  Please include an estimate of how long it took you to complete the coding exercise at the top of the notebook.

## Data: download and process ratings data

Download the Beauty 5 core review dataset (~45MB) from http://jmcauley.ucsd.edu/data/amazon/.  The website also has Python code to read `reviews_Beauty_5.json.gz` into a pandas DataFrame using `getDF()`:

```{python}
ratings = getDF('reviews_Beauty_5.json.gz')
```

For sake of consistency use the following renaming and relabeling of IDs:
```{python}
ratings.rename(columns={'reviewerID': 'user_id', 
                        'asin': 'item_id', 
                        'reviewerName': 'user_name', 
                        'reviewText': 'review_text',
                        'summary': 'review_summary',
                        'overall': 'score'},
               inplace=True)

ratings.user_id = ratings.user_id.astype('category').cat.codes.values
ratings.item_id = ratings.item_id.astype('category').cat.codes.values
# Add IDs for embeddings.
ratings['user_emb_id'] = ratings['user_id']
ratings['item_emb_id'] = ratings['item_id']
```

Using this dataset implement two models (and evaluate them as you see fit):
* Text embeddings and classifier
* Recommender system

## Text embeddings and classifier

1. Use the `“review_summary”` field and represent it as an embedding vector using transfer learning from word2vec/glove vectors. To save time we suggest to use the [**spacy**](https://spacy.io/) module, which has built-in functionality to calculate the average vector of a sequence of words. E.g., 

  ```{python}
  import spacy
  nlp_en = spacy.load('en', vectors='en_glove_cc_300_1m')
  example_reviews = ['this creme is amazing', 'creme amazing', 'this shoe does not fit; it hurts', 'shoe hurts']
  example_embs = np.vstack([nlp_en(e).vector for e in example_reviews])
  sklearn.metrics.pairwise.cosine_similarity(example_embs)
  ```

2. Compare two of your favorite classifiers for $score \geq 4.0$ using the embedding vectors for `"review_summary"` as features. The purpose of the classification is to alert the user if their score does not match what they have written.  Choose your preferred evaluation metric (e.g., AUC, F1, precision / recall, accuracy, ...), explain why you use it (rather than others), and pick the best classifier. 

3. Other than model evaluation metrics, provide human-interpretable results (word clouds, 2D projections of embeddings, data-driven examples, ...) on whether this classifier does a reasonable job of mapping text to low/high scores.

## Recommender system

1. Implement a basic collaborative filtering recommendation model using [**keras**](www.keras.io) (or [**tensorrec**](https://github.com/jfkirk/tensorrec)) only based on `user_id` and `item_id` (no user or item features). Include hyperparameter tuning for size of embeddings: select best model out of {10, 50, 100} embedding sizes and show how you evaluate the models. [This example](https://hackernoon.com/deep-learning-for-recommendation-with-keras-and-tensorrec-2b8935c795d0) might be useful.

2. For this question, assume that `ratings` also contains an `"item_descripton"` (string) field. Following the same transfer learning approach as above, we could obtain an `"item_description_embedding"` for any (new) item. Explain in 2 slides (pdf) how you would use  `"item_description_embeddings"` and user features (3 features: age $\in [0, 100]$, gender $\in {M, F}$, average historical spend on products $\geq 0$) to build a content based recommender system. 
    * Do not actually implement it, but just give a high-level summary on how this would be done and its main differences (pros/cons) to the     collaborative filtering approach.
    * Would Emadri prefer to user the collaborative filtering or the content-based version? Why?


> If there is anything else you found worthwhile in the dataset that might be relevant from a business or user-satisfaction point of view, feel free to include it in the notebook.