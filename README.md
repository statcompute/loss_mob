#### Introduction

To mimic the py\_mob package (https://pypi.org/project/py-mob) for binary outcomes, the loss\_mob is a collection of python functions that would generate the monotonic binning and perform the variable transformation for loss or severity such that the Spearman correlation between the transformed $X$, i.e. $F(X_i)$, and $E(Y_i | X_i)$ is equal to 1. In case of loss models with $Ln()$ link function, the transformation is derived as $F(x)_i = Ln \frac{\sum_i Y / \sum_i Exposure}{\sum Y / \sum Exposure}$ in the training sample, where $Exposure$ is the number of cases and $i$ refers to the $ith$ bin groupped by $x$ values.  

Should you have any question or suggestion about the package, please feel free to drop me a line. 

#### Core Functions

```
loss_mob
  |-- qtl_bin()  : Iterative discretization based on quantiles of X.  
  |-- los_bin()  : Revised iterative discretization for records with Y > 0.
  |-- iso_bin()  : Discretization driven by the isotonic regression. 
  |-- rng_bin()  : Revised iterative discretization based on the value range of X.  
  |-- kmn_bin()  : Iterative discretization based on the k-means clustering of X.  
  |-- gbm_bin()  : Discretization based on the gradient boosting machine (GBM).  
  |-- cus_bin()  : Customized discretization based on pre-determined cut points.  
  |-- view_bin() : Displays the binning outcome in a tabular form. 
  |-- cal_newx() : Applies the variable transformation to a numeric vector based on the binning outcome.
  |-- chk_newx() : Verifies the transformation generated from the cal_newx() function.
  |-- mi_score() : Calculates the Mutual Information (MI) score between X and Y.
  `-- screen()   : Calculates Spearman and Distance Correlations between X and Y.
```

####  Authors

[WenSui Liu](mailto:liuwensui@gmail.com) is a seasoned data scientist with 15-year experience in the financial service industry. 

[Joyce Liu](mailto:joyce.jl.liu@gmail.com) is a college student majoring in Mathematics with a strong passion for data science.
