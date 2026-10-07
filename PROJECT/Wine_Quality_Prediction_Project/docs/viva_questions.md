# Viva Questions and Short Answers

1. **What is the objective of the project?**  
   To predict wine quality from 11 physicochemical properties and compare Decision Tree, Random Forest, and Gradient Boosting.

2. **What type of ML problem is this?**  
   Supervised binary classification in this implementation.

3. **What is the target variable?**  
   The original `quality` score is converted to a binary target: 0 for scores below 6.5 and 1 for scores at least 6.5.

4. **Why remove duplicates?**  
   Duplicate rows can cause the same observation to appear in different partitions and can make evaluation less representative.

5. **Why use stratified splitting?**  
   It preserves the proportion of the two target classes in both training and testing sets.

6. **What is a Decision Tree?**  
   A tree-based model that recursively splits the feature space using criteria such as Gini impurity.

7. **What is Random Forest?**  
   An ensemble of decision trees trained using bootstrap samples and feature subsampling, with predictions aggregated across trees.

8. **What is Gradient Boosting?**  
   An ensemble method that builds trees sequentially, with later trees focusing on errors/residual structure from earlier trees.

9. **What is precision?**  
   Of the samples predicted as positive, the fraction that are actually positive.

10. **What is recall?**  
    Of the actual positive samples, the fraction correctly identified.

11. **What is F1-score?**  
    The harmonic mean of precision and recall: 2 × precision × recall / (precision + recall).

12. **Why compare three models?**  
    They use different ensemble strategies and provide a useful comparison of model behaviour on the same data.

13. **What does feature importance show?**  
    It indicates how much each input feature contributes to the split-based predictions of a tree-based model.

14. **What is one limitation?**  
    The dataset is imbalanced, with relatively few high-quality samples, so accuracy alone should not be interpreted in isolation.

15. **What can be done in future work?**  
    Use multiclass quality prediction, cross-validation and hyperparameter search, class-imbalance techniques, and external datasets for broader validation.
