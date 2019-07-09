# Inference and Validation

Neural networks are good at training data but fail at testing data - because of **Overfitting**

## Overfitting
As you train more on training data, the network finds more correlations and patterns which may not be true of the entire dataset. And so, when these patterns are absent in the testing data, they fail.

For an accurate model, Measure performance of data on the **validation dataset**


**topk()** - return a tuple of top 'k' values and the top 'k' indices. Eg: topk(1) => top 1 => highest value in the set and it's index. 
