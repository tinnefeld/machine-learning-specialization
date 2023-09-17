# Week1
## Supervised vs. unsupervised learning
### What is machine learning?
- "Field of study that gives computers the ability to learn without being explicitly programmed." - Arthur Samuel (1959)
- Machine learning algorithms
  - Supervised learning (used most in real-world apps)
  - Unsupervised learning
  - Recommender systems
  - Reinforcement learning
- These algorithms are just tools, important to know how to apply and when to use what 
### Supervised learning part 1 (regression)
- learning input to output mapping
- correct output label y for an input x
- applications: spam filtering, speech recognition, online advertising, self-driving, visual inspection
- first train models with examples, then brand new, un-seen input x
- house pricing example
- regression: fitting straight line, curve or more complex function?
- regression: "Predict a number from infinitely many possible outputs."
- predict a number
### Supervised learning part 2 (classification)
- second major type of supervised learning: classification
- cancer detection example: malignant vs benign
- different from regression as (small) finite number of possible outputs
- output classes or categories terms used interchangeably
- fit a boundary line through data
- predict a category
### Unsupervised learning part 1 (clustering)
- find something, structured or interesting in unlabeled data
- clustering algorithm puts unlabeled data into clusters
- Google news example: grouping news articles in clusters
- Algo has to figure out what are the news clusters on a day?
- DNA microarray clustering
- Group customers into segments
### Unsupervised learning part 2
- "Data only comes with inputs x, but not output labels y. Algorithm has to find structure in the data."
- Clustering: Group similar data points together.
- Anomaly detection: Find unusual data points.
- Dimensionality reduction: Compress data using fewer numbers.
### Jupyter notebooks
- Most commonly used tool
## Regression model
### Linear regression model part 1
- (linear) regression model predicts numbers
- data table with sample data contains inputs and outputs
- Terminology:
  - Training Set: Data used to train the model
  - $x$: "input variable" aka "feature"
  - $y$: "output variable" aka "target variable" or just "target"
  - $m$: number of training examples
  - $(x,y)$: single training example
  - $(x^{(i)},y^{(i)})$ = i<sup>th</sup> training example, i refers to index
### Linear regression model part 2
- training set contains features and targets
- learning algorithm uses this to create a function $f$
- function $f$ - also known as the model - creates a prediction based on feature $x$
- the prediction is called $\hat{y}$ (y-hat) or estimated $y$
- Function representation $f_{w,b}(x)=wx+b$ ( or simpler just $f(x)$)
  - linear regression with one (single feature $x$) variable
  - aka Univariate linear regression
- Linear function (aka straight line) is easy to work with, good to get started
### Linear regression: cost function formula
- cost function is telling us "how well the model is doing"
- parameters $w, b$ sometimes also called coefficients or weights
- y-intercept is the value where the line crosses the y axis
- slope of the line is the "increase" of height per unit, value of $w$ gives the slope
- Find $w,b$ so that: $\hat{y}^{(i)}$ is close to $y^{(i)}$ for all $(x^{(i)},y^{(i)})$.
- Squared error cost function: $J(w,b)=\dfrac{1}{2m}\displaystyle\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})^2$
  - calculate the difference $\hat{y}^{(i)}-y^{(i)}$, also known as the error
  - this indicates "how close" the predication is to the target
  - do this for all training examples, calculate the average
### Linear regression: cost function intuition
- goal, find parameters $w,b$ with the lowest (possible) cost
  - $\displaystyle{minimize_{w,b}}J(w,b)$
- $J(w)$ becomes 0 if the cost function is a perfect match
### Linear regression: visualizing the cost function
- 2-dimensional visualization for single parameter $w$
- 3-dimensional visualization for two parameters $w,b$
- contour map includes horizontal slices of a 3d surface
  - you get all the points at the same height
- different points with same $J$, but different $w,b$
### Linear regression: visualization examples
- good parameters/low cost is close to the center circle of the contour map
## Train the model with gradient descent
### Gradient descent
- Can be used to minimize cost function
- Gradient descent cannot only be used for linear regression but for any function
- not limited in terms of number of variables, e.g. $J(w_1,w_2,\dots,w_n,b)$
- Outline
  - Start with some $w,b$
  - Keep changing $w,b$ to reduce $J(w,b)$
  - Until we settle at or near a minimum
    - We may have >1 minimum
- Direction of steepest descent guides to the most efficient way to get to the lowest point
- Repeating these steps is a gradient descent
- Possible to repeat the gradient descent process from another starting point
  - Possible to end up in a different valley aka minimum
### Implementing gradient descent
- General algorithm: $w=w-\alpha\dfrac{d}{dw}J(w,b)$ (equal sign here is an assignment)
  - Take the previous $w$, modify it slightly via $\alpha$ and recalculate J
  - $\alpha$ is also known as the learning rate
  - $\dfrac{d}{dw}J(w,b)$ derivative term indicates in which direction you want to take a baby step also simply known as derivative
- same applies for $b$
- repeat until algorithm converges which means reaching a local minimum where $w,b$ don't change much with each additional step that you take
- you want to update both parameters at the same time aka simultaneous update
### Implementing descent intuition
- learning rate $\alpha$ determines how big of a step you take when learning
- derivate term is $\dfrac{d}{dw}J(w)$
- derivative is the slope of the function $j$ at a point
- positive or negative slope indicates in which direction "to go" for advancing $w$
### Learning rate
- if the learning rate $\alpha$ is too small, gradient descent will work, but takes a long time
- if the learning rate $\alpha$ is too large, you may overshoot, never reach the minimum
  - may fail to converge, even diverge
- what happens if you are at a local minimum, but there is another, smaller local minimum elsewhere? ( we don't know yet... )
- working with a fixed learning rate
  - gradient descent automatically makes smaller steps as we reach a local minimum
### Gradient descent for linear regression
- repeat until convergence
  - $w=w-\alpha\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$
  - $b=b-\alpha\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})$
- when using a squared error cost function with linear regression, the cost function does not and will never have multiple local minima
- it has a single global minimum aka a convex function
- informally, a convex function is a bow shaped function with a single global minimum
### Running gradient descent
- Batch gradient descent refers to the fact that on every step of gradient descent we look at **all** training examples
  - other versions of gradient descent just look at subsets

# Week2
## Multiple linear regression
### Multiple features
- We write $x_{j} = j^{th}$ feature to denote multiple features such as $x_{1}, x_{2}, x_{3} \dots$
- $n$ = the number of features
- $\overrightarrow{x}^{(i)}$ = features of $i^{th}$ training example, e.g.:  $\overrightarrow{x}^{(2)} = [1,2,3,4]$
- Previously: $f_{w,b}(x)=wx+b$
- Now:
  - $f_{w,b}(x)=w_{1}x_{1}+w_{2}x_{2}+\dots+w_{n}x_{n}+b$
  - parameters of the model 
     - $\overrightarrow{w}=[w_{1},w_{2},w_{3} \dots w_{n}]$
     - $b$ is a number
  - resulting in $f_{\overrightarrow{w},b}(\overrightarrow{X})=\overrightarrow{w}\cdot\overrightarrow{X}+b$
### Vectorization
- in linear algebra vector counts start at 1
- NumPy in python is mostly used numbers library
- f = np.dot(w,x) + b in NumPy, vectorized implementation of dot multiply operation
- computer can multiply all vector values at once in one step
### Gradient descent for multiple linear regression
- cost function $J(\overrightarrow{w},b)$
- repeat until convergence
  - $w_{n}=w_{n}-\alpha\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})-y^{(i)})x_{n}^{(i)}$
  - $b=b-\alpha\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})-y^{(i)})$
- An alternative to gradient descent, called "Normal equation"
  - Only for linear regression
  - Solve for w,b without iterations
  - Doesn't generalize to other learning algorithms, slow when number of features is large
## Gradient descent in practice
### Feature scaling
- If range of values for a feature is large, it is more likely that a good model will choose a small parameter value
- If possible range of values is small, the value for the parameter will be rather large
- In order to avoid this, we rescale the value range for our features, so that their ranges become comparable
- This results in the contours of the visualization o the parameters to look more like circles which are preferable for gradient descent to properly work. It can find a more direct path to the minimum.
- Example: range of $300\le x_{1}\le 2000$ becomes $0.15\le x_{1,scaled}\le 1$
- Mean normalization aims at centering them around 0, example: range of $300\le x_{1}\le 2000$ becomes $-0.18\le x_{1}\le 0.82$
- Z-score normalization based on standard deviation, example: range of $300\le x_{1}\le 2000$ becomes $-0.67\le x_{1}\le 3.1$
- In general, aim for about $-1\le x_{j}\le 1$ for each feature $x_{j}$
- No need to rescale in roughly within the same range, but do rescale if range too large or too small
### Checking gradient descent for convergence
- General goal: find parameters close to the global minimum of the cost function or minimize cost function $J(\overrightarrow{w},b)$
- Idea: check that cost decreases with more iterations by plotting costs after a number of (e.g. 100) iterations
  - This is also called the "learning curve"
  - Good indicator when you can stop training your model
  - Also possible to do an "automatic convergence test" where we compare the cost improvement in one iteration against a predefined $\epsilon$ epsilon.
    - Challenge with this approach is to pick a fitting epsilon value, so simply "looking at" the learning curve might yield better results
### Choosing the learning rate
- Challenge: if cost rate goes up or down with more iterations, gradient descent is not working properly
  - Possible reasons: bug in the code or learning rate $\alpha$ is too big
  - Fix: use a smaller learning $\alpha$
  - With a small enough $\alpha$, cost function should decrease on every iteration
    - However, can then take very long to complete
  - Approach: pick different alphas such as $0.0001$, $0.01$, $0.1$, $1$ ... , run a few iterations, and observe resulting learning curve
### Feature engineering
- Using intuition to design new features by transforming or combining original features.
- Example: Combine frontage and depth of a property to area by multiplying them. Add area as a separate feature to the model with its own weight.
### Polynomial regression
- Allows you to fit curves (non-linear functions) to your data 
- Combine ideas of multiple linear regression and feature engineering
  - Example: $f_{\overrightarrow{w},b}(x)=w_{1}x+w_{2}x^2+b$
    - Here, parameter $x$ (size of house), is also added as $x^2$
- How to decide which features to use? -> Will be discussed later, for now just know that these choices exist.

# Week 3
## Classification with logistic regression
### Motivations
- Linear regression predicts a number
- Classification: output variable y can take on only one of a small handful of possible values
- Classification examples:
  - Is this email spam? -> no/yes
  - Is the transaction fraudulent? -> no/yes
  - Is the tumor malignant? -> no/yes
  - Answer $y$ can only be of two values aka "binary classification"
- In this context, class and category are used interchangeably
- false or 0 are called "negative class", true or 1 are called "positive class"
- Linear regression - even with a threshold - not really applicable as outlier examples change the fitted curve and hence move the decision boundary leading to misclassified data
### Logistic regression
- Probably the single most widely used classification algorithm
- Sigmoid function is a logistic function which creates outputs between $0$ and $1$
- $g(z)=\dfrac{1}{1+{e}^{-z}}$
  - If $z$ converges to a large number, $g(z)$ converges to 1
  - If $z$ converges to a very small, negative number, $g(z)$ converges to 0
- Building logistic regression formula
  - Linear regression: $f_{\overrightarrow{w},b}(\overrightarrow{x})=\overrightarrow{w}\cdot\overrightarrow{x}+b$
    - Which we assign to $z$, resulting in $z = \overrightarrow{w}\cdot\overrightarrow{x}+b$
  - We now take z and pass it into the Sigmoid function, resulting in: $f_{\overrightarrow{w},b}(\overrightarrow{x})=g(\overrightarrow{w}\cdot\overrightarrow{x}+b)=\dfrac{1}{1+{e}^{-(\overrightarrow{w}\cdot\overrightarrow{x}+b)}}$
- Interpretation of logistic regression output
  - Outputting the probability that class or the label y is 1, given a certain input x
  - Example: $f_{\overrightarrow{w},b}(\overrightarrow{x})=0.7$
    - $70$% chance that $y$ is $1$
    - Also implies that is has a $30$% chance of being $0$
### Decision boundary
  - Decision boundary: $z=\overrightarrow{w}\cdot\overrightarrow{x}+b=0$
  - That line is where you are almost neutral if the decision is one or zero
## Cost function for logistic regression
- Squared error cost function is not an ideal cost function for logistic regression
  - Will result in many local minima, but not in a global minimum
- Given a training set for logistic regression, how an you determine parameters $w, b$?
- Formula:
$$
\mathrm{L(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)},y^{(i)}))} = \begin{cases}
    -\log(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})) & \text{if } y^{(i)} = 1 \\
    -\log(1-f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})) & \text{if } y^{(i)} = 0
\end{cases}
$$
- The further prediction $f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})$ is from target $y^{(i)}$, the higher the loss.
- This makes the loss function convex, reliably leading to the global minimum
- Resulting cost function that includes all training examples: $J(\overrightarrow{w},b)=\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}L(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)}),y^{(i)})$
### Simplified Cost Function for Logistic Regression
- Possible to simplify if we assume $y$ can either be $1$ or $0$
- $J(\overrightarrow{w},b)=-\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}[y^{(i)}\log(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)}))+(1-y^{(i)})\log(1-f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)}))]$
- Based on the statistical principle of "maximum likelihood"
## Gradient Descent for Logistic Regression
- Find values $w, b$ to minimize cost function
- Simultaneous updates to apply updates to all parameters at the same time
- Main difference is that in logistic regression the defintion for $f(x)$ is Sigmoid function
## The problem of overfitting
- If a model does not fit the training set well, it has an "underfit" or a "high bias"
  - Can happen with too few features.
- If a model fits the training set pretty well, it achieves "generalization" aka "just right"
- If a model fits the training set extremely well (e.g. the cost function is zero), it can have an "overfit" or has an "overfitting problem" or it has "high variance"
  - Can happen with too many features.
### Addressing overfitting
- collect more training examples
- select features to include/exclude
  - exclude features for an overfitting problem
  - lot of features and insufficient data -> overfit
  - feature selection: picking the "just right" features by selecting just a subset of the available features (e.g. based on intuition)
    - disadvantage: useful features could be lost
- Reduce size of parameters aka "regularization"
  - reduce the size of parameters $w_{j}$ (instead of eliminating)
  - keep all features but prevents individual features from having an overly large effect
### Cost function with regularization
- if you have a lot of features, how to know which ones are most important?
- Regularization term: $\dfrac{\lambda}{2m}\displaystyle\sum_{j=1}^{n}w^{2}_{j}$
- Resulting in: $J(\overrightarrow{w},b) = \dfrac{1}{2m}\displaystyle\sum_{i=1}^{m}(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})-y^{(i)})^2+\dfrac{\lambda}{2m}\displaystyle\sum_{j=1}^{n}w^{2}_{j}$
- value of $\lambda$ balances between i) fitting the data and ii) keeping $w_j$ small
  - if $\lambda = 0$, no use of regularization term at all ( overfit )
  - if $\lambda$ is a very large number, the function reduces to $f(x)=b$ ( underfit )
### Regularized linear regression
- repeat until convergence
  - $w_{j}=w_{j}-\alpha[\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}[(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})-y^{(i)})x_{j}^{(i)}]+\dfrac{\lambda}{m}w_j]$
  - $b=b-\alpha\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)})-y^{(i)})$
### Regularized logistic regression
- Cost function: $J(\overrightarrow{w},b)=-\dfrac{1}{m}\displaystyle\sum_{i=1}^{m}[y^{(i)}\log(f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)}))+(1-y^{(i)})\log(1-f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)}))]+\dfrac{\lambda}{2m}\displaystyle\sum_{j=1}^{n}w^{2}_{j}$