### Proposed Modeling Approach

Goal: predict event-free survival for patients who have undergone a bone-marrow transplant.
* Data set has *censored data*, meaning that some subjects were lost to follow-up before experiencing an event (death or relapse). Their true survival time in unknown. 
* Censored data requires specialized *survival-analysis* algorithms.

We will try two primary families of algorithms.

#### Cox Proportional Hazard Model

* Commonly used in survival analysis
* Assumes that the *log-hazard* function of an individual is linear in the model features. 
* Simple and interpretable, and will serve as a baseline.
* Implemented in both `xgboost` and `lifelines`.

#### Tree-Based Models

* Ensemble models based on decisions trees often perform well on tabular data.
* Specialized versions of these models have been developed for survival data.
* Implemented in `scikit-survival` library.

#### Other Considerations

In addition to fitting multiple models, we will consider:
* Different methods of imputing missing data. 
** Train a predictive model to impute missing values
* Feature Selection and Engineering
** Try dropping some highly-correlated features
** Try dropping low-variance features, or binning low-variance features together

