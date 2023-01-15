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
| ID    | Clothing ID | Age | Title                                            | Review Text                                         | Rating | Recommended IND | Positive Feedback Count | Division Name  | Department Name | Class Name |
| ----- | ----------- | --- | ------------------------------------------------ | --------------------------------------------------- | ------ | --------------- | ----------------------- | -------------- | --------------- | ---------- |
| 7828  | 1082        | 32  | I wanted to love this                            | I really wanted to love this dress.... \\n\\ni a... | 3      | 0               | 0                       | General        | Dresses         | Dresses    |
| 3435  | 1056        | 38  | Watch out for dye staining your load of laundry! | I read the cleaning instructions label careful...   | 1      | 0               | 6                       | General Petite | Bottoms         | Pants      |
| 23135 | 1079        | 39  | Disappointing                                    | This is such beautiful material, but the sleev...   | 3      | 0               | 22                      | General Petite | Dresses         | Dresses    |

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
- class weights: [5.63418291 3.0677551  1.65477763 0.91569201 0.35712249] (classes 0-4 -> 1-5 rating)
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
- train a statistical model for sentiment analysis (negative, neutral, positive) of the Title and Review Text columns (Scikit Learn's MultinomialNB, inspired by https://www.kaggle.com/code/burhanykiyakoglu/predicting-sentiment-from-clothing-reviews, 1-2 negative, 3 neutral, 4-5 positive)
  - in case the sentiment analysis classifier for the title won't reach at least 80% accuracy, use Huggingface's sentiment-analysis pipeline: https://huggingface.co/docs/transformers/v4.25.1/en/task_summary#text-classification
- use one or more tree models, such as Scikit Learn's RandomForest and GradientBoostingTrees, and maybe also XGBoost and try to reach a better accuracy than 67% in predicting a product's rating from a data row
