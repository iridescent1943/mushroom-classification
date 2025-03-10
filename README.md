## Summary

**Data Source:** Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T.

**The goal of this small machine learning project** was to develop a neural network to accurately classify mushrooms as either edible or poisonous. Based on all available results and analyses, a 10-neuron model using 13 input parameters and Adam optimizer (with an initial learning rate of 0.001), is considered the best and most cost-effective model obtained so far. It can achieve 100% classification accuracy while containing only one hidden layer and requiring only 13 mushroom features. This is highly advantageous in practice, as measuring mushroom features can be labour-intensive, time-consuming, and expensive. 

The **investigation process** involved:
1. Qualitative exploration of input-output relationships using line, bar and parallel plots to visualize data trends, and preliminary quantitative analysis to identify important features.
2.	Initial training with all input parameters.
3.	Second round of training with a reduced input set
4.	Third round of training with same reduced input set by adding additional hidden layer.
5.	Sobol sensitivity analysis to more accurately investigate the input-output relationship, revealing gill spacing as one of the main dominant features that impact poisonous status.

**It should be noted** that the original dataset contains 8124 instances, and all feature parameters are categorical data. Therefore, data was first encoded into numerical form before model training. While the data donor specifies the number of total subcategories for each mushroom feature in the associated documentation, it is noticed during training that not all subcategories are present in the original dataset. 

For example, while veil type can be either partial or universal, all data in the dataset are of the partial type, making it impossible to evaluate the influence of veil type on the poisonous status, thus veil type has to be removed from the ANOVA analysis as it is a constant column, and this is also why it is ranked lowest for its influence on poisonous status during Sobol sensitivity analysis for all inputs.

Similarly, while gill attachment can have 4 subcategories, gill spacing can have 3 categories, stalk root can have 7 subcategories, and ring type can have 8 subcategories, the data used for training only has 2, 2, 5, and 5 subcategories, respectively. Therefore, the best model obtained might not perform well when trying to identify mushrooms that have features that are not present in the training data. 

Accurate classification of a mushroom's poisonous status is extremely important in real life, and misclassification could lead to severe consequences (e.g., mistaking a poisonous mushroom as edible), it is not trivial to emphasise that **additional biochemical testing is needed** to determine whether a mushroom can be safely consumed or commercially used as food. Also, a larger, more diverse dataset that can accurately represents mushroom varieties found in nature would be beneficial for building a more robust and reliable model.

## Some Results Obtained During Training

*Bar plots of each feature against the target using the whole dataset.*
<img src="results\bar plots.png" alt="bar plots" align=center style="margin-bottom: 20px;" />

*Bar plots comparing the ANOVA F-value of mushroom features using the whole dataset. Note that the veil type feature is excluded as it contains identical values across the entire dataset, making F-value calculation impossible for this constant column.*
<img src="results\F values.png" alt="F-values" align=center style="margin-bottom: 20px;" />


*Confusion matrix plots for the best mode found so far with 100% classification accuracy - the number of true positives and true negatives (diagonal).* 
<img src="results\Confusion Matrix.png" alt="Confusion Matrix" align=center style="margin-bottom: 20px;" />


*Results of third round of training using the same reduced input set.*
<img src="results\table.png" alt="results_table" align=center style="margin-bottom: 20px;" />