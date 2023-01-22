# trees-vs-nns

Inspired by the paper "Why do tree-based models still outperform deep learning on tabular data?" (https://arxiv.org/abs/2207.08815), this is an attempt to beat a previously built NN's accuracy of 67% in predicting product ratings of an online store. The NN that I previously built was an NLP course project.

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

### Next Ideas

- make correlation matrices
- think about adding vectorized title and review text as columns instead of prior sentiment extraction
- think about data augmentation
- hyperparameter tuning with wandb
