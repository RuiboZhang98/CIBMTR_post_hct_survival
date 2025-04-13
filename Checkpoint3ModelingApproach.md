### Proposed Modeling Approach

Goal: predict event-free survival for patients who have undergone a bone-marrow transplant.

* Data set has *censored data*, meaning that some subjects were lost to follow-up before experiencing an event (death or relapse). Their true survival time in unknown.
* Censored data requires specialized *survival-analysis* algorithms.

### Model description

* The true survival time is modeled to be a random variable $T$ that depends on the features $p_1 \dots p_n$.
* The survival function is defined as $S(t) = \mathbb{P}(T > t)$. The harzard function is defined as a conditional density of $T$, i.e. $h(t) = p(t|T > t) = -S'(t)/S(t)$. Thus, $S(t) = \exp\left(- \int_0^t h(s) {\mathrm d} s\right)$.

We will try two primary families of algorithms.

#### Cox Proportional Hazard Model (CPH)

* It is a parametric model on $h(t)$.
  * E.g. assumes that the *log-hazard* function of an individual is linear in the model features.
* Simple and interpretable, and will serve as a baseline.
* Implemented in both `xgboost` and `lifelines`.

#### Accelerated Failure Time Model (AFT)

* It is a parametric mode on $S(t)$.
  * E.g. assumes that $\log(S(t)) = \theta \cdot p + Z$ in which $theta$ is a vector of parameters, $p$ is a vector of features, and $Z$ is a noise term.
* It has more degree of freedom in terms of modeling.
  * $Z$ can be choose from Normal, Wellbull, etc.

In addition, these two class of models have their tree-based alteratives.

#### Tree-Based Models

* Ensemble models based on decisions trees often perform well on tabular data.
  * A decision tree is a parameteric map $\Gamma(\theta,p)$ that encodes parts of the model.
* Specialized versions of these models have been developed for survival data.
  * For CPH model, this means $\log(h(t)) = \Gamma(\theta,p)$. (Ruibo: Not entirely sure if this is true.)
  * For AFT model, this means $\log(S(t)) = \Gamma(\theta,p) + Z$
* Implemented in `scikit-survival` library.

#### Other Considerations

In addition to fitting multiple models, we experiment with the following:

* Different methods of imputing missing data.
  * Train predictive model to impute missing values
  * Try `IterativeImputer` from `scikit-learn`.
* Feature Selection and Engineering
  * Try dropping some highly-correlated features
  * Try dropping low-variance features, or binning low-variance features together
