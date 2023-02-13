# Model Card

## Model Details
The model is a DecisionTreeClassifier with min_samples_split=2, min_samples_leaf=1, and min_weight_fraction_leaf=0.0. 
## Intended Use
The prediction is for whether a person's annual earnings is greater or less than $50,000. 

## Data
Training Data: 80% of Census data
Evaluation Data: 20% of Census data
The data can be found here; https://archive.ics.uci.edu/ml/datasets/census+income


## Metrics
Metrics for this model are:
Precision: 0.6026119402985075
Recall: 0.6215522771007056
F-beta: 0.6119355857278181

## Ethical Considerations
Since bias in the training data, the model may occur a result contains harmful or offensive. It is important to carefully monitor and check the result and filter out any inappropriate feature before using it.

## Caveats and Recommendations
The data is messy. There is limitations of trainning data and technical limitations that is the model has limitations in processing for certain tasks, such as understanding salary.
Some values in some columns have unknown value which is set to '?'. '?' in workclass column has 1836 rows that is almost 6% of data. 