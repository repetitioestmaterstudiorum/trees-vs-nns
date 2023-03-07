# trees-vs-nns

Inspired by the paper "Why do tree-based models still outperform deep learning on tabular data?" (https://arxiv.org/abs/2207.08815), this is an attempt to beat a previously built NN's accuracy of 67% in predicting product ratings of an online store (referring to an NN I built during a course at university).

## Teaser

## Data

The data has the following columns:

- ID
- productID
- reviewer age
- review title
- review text
- rating(1-5)
- positive feedbacks on review
- productdivision
- productdepartment

The first 3 dataset rows look like this:
| ID | Clothing ID | Age | Title | Review Text | Rating | Recommended IND | Positive Feedback Count | Division Name | Department Name | Class Name |
| ----- | ----------- | --- | ------------------------------------------------ | --------------------------------------------------- | ------ | --------------- | ----------------------- | -------------- | --------------- | ---------- |
| 7828 | 1082 | 32 | I wanted to love this | I really wanted to love this dress.... \\n\\ni a... | 3 | 0 | 0 | General | Dresses | Dresses |
| 3435 | 1056 | 38 | Watch out for dye staining your load of laundry! | I read the cleaning instructions label careful... | 1 | 0 | 6 | General Petite | Bottoms | Pants |
| 23135 | 1079 | 39 | Disappointing | This is such beautiful material, but the sleev... | 3 | 0 | 22 | General Petite | Dresses | Dresses |

An emphasis (and primary requirement in the exercise) was to use the review text.

Class imbalances: rating 5 occurs ~56% of the time.

The tabular data is available here: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

## NN approach to be beaten

The NN approach had the following characteristics:

- BERT base uncased as pretrained base (https://huggingface.co/bert-base-uncased)
- then a classifier with 1 hidden layer with 50 nodes, a dropout of 0.2, and one output layer
- classification with 5 classes: 0-4 (ratings 1-5)
- classifier training was done without freezing BERT layers
- AdamW optimizer with a linear scheduler
- class weights: [5.63418291 3.0677551 1.65477763 0.91569201 0.35712249] (classes 0-4 -> 1-5 rating)
- 2 training epochs
- batch size: 32
- learning rate: 0.00005
- dataset train/valid/test split: 80/10/10%

To train the NN, columns were combined into features by using the item number, title, review text, and age. Example feature:

"Review for item 767. No title: Absolutely wonderful - silky and sexy and comfortable. Reviewer age: 33 --> Review for item 767. No title: Absolutely wonderful - silky and sexy and comfortable. Reviewer age: 33"

Preprocessing:

- empty reviews -> "No review"
- empty title -> "No title"
- empty age -> "unknown"
- empty Clothing ID -> "unknown"
- removing trailing whitespace

Best result: 67% accuracy.

## Tree models

The (initial) idea is the following:

- use all data (columns)
- encode Division Name, Department Name, and Class Name columns (probably simply using a one-hot-encoding approach)
- train a statistical model for sentiment analysis (1-5) of the Title and Review Text columns (Scikit Learn's MultinomialNB, inspired by https://www.kaggle.com/code/burhanykiyakoglu/predicting-sentiment-from-clothing-reviews)
  - in case the sentiment analysis classifier for the title won't reach at least 80% accuracy, use Huggingface's sentiment-analysis pipeline: https://huggingface.co/docs/transformers/v4.25.1/en/task_summary#text-classification
- use one or more tree models, such as Scikit Learn's RandomForest and GradientBoostingTrees, and maybe also XGBoost and try to reach a better accuracy than 67% in predicting a product's rating from a data row

## Log / Learnings

- ~~instead of one hot encoding Division Name, Department Name, and Class Name columns, I just assign a number for each unique category -> it's simple and fast~~ -> use one hot encoding for label colums because data already comes split into train, valid, and test datasets -> this means that encoding needs to happen with the same encoder for each dataset (fit on train)
- MultinomialNB is more than 100 times faster and only slightly less accurate (~3%) for title based rating prediction than LogisticRegression with this data
- Logistic Regression is the Scikit-Learn supervised learning model with the best base accuracy for title-based rating prediction (without much hyperparameter tuning) for this data and preprocessing
- it's a surprise and seems wrong that the title based rating predictions are more accurate (~61%) than review text based ones (~55%). There must be the possibility to extract more out of review texts, since they're longer and contain more information?
- initial tree model (Random Forest) results on all data with sentiment (rating predictions) from title and review texts: ~65% accuracy
- dropping instead of filling empty rows increases review text sentiment prediction by .5%, but decreases title sentiment prediction by ~1%. It decreases Random Forest tabular accuracy by 2%, and increases accuracies for Logistic Regression, Gradient Boosting, and XGBoosst by ~.2-.5 %
- after a switch to one hot encoding of categories, ensuring consistent transformation across datasets (fit on train, transform on all datasets with the same encoder), accuracy results are unexpected... Random Forest: -3%, Logistic Regression: -.5%, Gradient Boosting: +.1%, XGBoost: -.7%. I can only conclude that either I made a mistake (but I double-checked that I didn't) or the perviously wrongly encoded categories happened to make the results slightly better by chance
- tabular accuracies are slighly worse for tree models and slighly better for Logistic Regression without the categorical columns data in X
- dropping the Reviev Text Sentiment (with ~55% accuracy predicting Rating) increases boosted tree models' accuracy slightly, Random Forest by 2%, and the Linear Regression does not converge anymore (with 10000 max_iter, like before) but the accuracy doesn't decrease more than 1%. Very interesting result
- dropping both Title and Review Text decreased the performance of all models (expected). Random Forest: 62->55%, Logistic Regression: 65->62%, Gradient Boosting: 64->60%, XGBoost: 65->62%
- adding vectorized data to the dataframe increases its dimensionality a lot, which is why I stopped computing Random Forest, Logistic Regression, and Gradient Boosting results for now
- adding the vectorized title as columns to the dataframe increases XGBoost's accuracy to 63%
- doing the same with the text leads to 64.5% accuracy with XGBoost. Finally, the review text, which contains more words than the title, seems to hold more information than the title. Running this took 3 hours, so the next step is using another vectorization for text and try to improve this (count vectorizer)
- using the count vectorizer resulted in the same amount of columns (212450 of which ~30 aren't a OHE of the text). Need to find a way to control the vector length. The result is a bit worse with ~63.5% accuracy
- using max_features=5000 for the tfidf_vectorizer, the training time is reduced from ~3h to 5s again for XGBoost, but accuracy is en par with the count vectorizer (and no max_features, ~3h train time)
  - with 1'000 features, training time is ~1s and the accuracy is 63.41%
  - with 5'000 features, training time is ~5s and the accuracy is 63.71%
  - with 7'000 features, training time is ~6s and the accuracy is 64.01%
  - with 8'000 features, training time is ~7s and the accuracy is 64.13%
  - with 9'000 features, training time is ~8s and the accuracy is 63.24%
  - with 10'000 features, training time is ~8s and the accuracy is 63.50%
  - with 30'000 features, training time is ~29s and the accuracy is 63.33%
  - with 50'000 features, training time is ~53s and the accuracy is 63.67%
  - with 100'000 features, training time is ~5min and the accuracy is 63.67%
  - with 150'000 features, training time is ~10min and the accuracy is 63.67%
- after increasing the ngram range from 1,2 to 1,3 for the review text
  - with 6'000 features, training time is ~6s and the accuracy is 63.62%
  - with 7'000 features, training time is ~6s and the accuracy is 63.71%
  - with 8'000 features, training time is ~7s and the accuracy is 63.62%
  - with 10'000 features, training time is ~8s and the accuracy is 63.41%
- adding back the title to the best result (10k features, ngram range 1,2): 35s, 64.82% accuracy
- it seems that the best performance can be reached by perviously training a model on predicting ratings independently on title, then review, and using these models' predictions as additional columns for the tabular data (like a sentiment), leads to the best results in this case. An interesting insight is how slow XGBoost is for high dimensional data compared to Multinomial Naive Bayes or Logistic Regression
- switching back to a multi-model approach: extract information from the title and review text columns, and add that to the tabular data, then use XGBoost for review classification
- after applying max_features while vectorizing, the the review text logistic regression resulted in higher rating sentiment prediction accuracy than that for the title. This is what I initially expected; that the review text, because it is longer, contains more information. Subsequently I found the optimum max_features for vectorizing the training data: 8000 max_features for title, 13000 for review text
- since Logistic Regression yields better in this case for rating sentiment prediction than XGBoost and other models I've tried, I used it also for the tabular data, and was surprised to see that with standard hyper parameters, Logistic Regression performs 1% better than the already slightly optimized XGBoost, yielding a new best score of 67.67%! If the same accuracy can be achieved on the test set in the end as well, this would already beat the 67% (initial goal of this project)
- hyperparameter optimization fox XGBoost yields a bit higher results than no tuning, but still not better than Logistic Regression for this task: ~66.90% (instead of 66.26% without tuning)
- after letting hyperopt run for a longer time, the accuracy increased up to ~67.41% with XGBoost
- finally, testing on test data shows that the accuracy drops by ~2% for both models
- after using wandb.sweep instead of hyperopt, XGBoost accuracy is ~67% on train, and 65.28% on test. wandb sweep data: https://wandb.ai/embereagle/trees-vs-nns/reports/-23-03-07-09-28-03---VmlldzozNzIxMDcz?accessToken=5mhhip55kxj8tgkwn6m1x40nss3xq9r91ek8xixwalq0hj8whenihek14vy0sprc

### Next Ideas

- again concat the (now much shortened) review text and title vectors to the tabular data, since Logistic Regression seems to perform best for both tasks (text to rating sentiment, tabular data to rating)
- use hyperopt for the Logistic Regression model
- alternative to hyperopt: https://www.youtube.com/watch?v=9zrmUIlScdY&list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk

## Key Takeaways

- This dataset is not a good dataset to compare NN (e.g. fine-tuning BERT) and tree models, because most of the relevant information is contained in natural language, and NNs are better at extracting meaning out of natural language, at this point in time
- Building encoders for categorical values takes quite some time. In real life, they have to be stored somehow to be reused in a production environment, to ensure that same inputs result in same outputs during training and inference time when pre-processing data
- Using XGBoost for text classification is not computationally efficient after a certain size. Other models, such as MultinomialNB (and to a lesser extend Logistic Regression), are better suited for this task
- Manually keeping track of results is tedious. Solution: Tools like wandb.ai
- When vectorizing (e.g. count vectorizer, tfidf vectorizer), higher accuracy in text classification can be achieved when applying max_features to vectorizers (limiting the size of vectors)
