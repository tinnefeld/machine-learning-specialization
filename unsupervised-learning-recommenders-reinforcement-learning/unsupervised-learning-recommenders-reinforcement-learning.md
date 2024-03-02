# Week 1 - Unsupervised learning
## Clustering
### What is clustering?
- Clustering looks at a number of data points and automatically finds data points that are related or similar.
- Supervised learning includes inputs x and (target) labels y
- Unsupervised learning has no (target) labels y
- Algorithm is asked to find some structure regarding the data
- Clustering - see if data can be grouped, is similar to each other
- Applications of clustering: e.g. grouping similar news, DNA analysis, market segmentation, astronomical data analysis
### K-means intuition
- Step 1
  - Takes a random guess where the centers of clusters are, for now start with 2 centers
  - Calculate for each point (input) how far they are from the centers
  - Assign (or associate) each point to the cluster centroids it is closer to
- Step 2
  - assign colors for the two groups of cluster ( e.g. red and blue )
  - look at all the red/blue points, take average of their location, move their centers accordingly
- Repeat, starting with step 1 for updated center locations
- Eventually, no more changes to the colors of the points or updates to the centers
  - -> the algorithm has converged
### K-means algorithm
- Randomly initialize $K$ cluster centroids $\mu_{1},\mu_{2},...,\mu_{K}$
  - in previous example $K=2$ (red and blue)
- Repeat
  - Assign points to cluster centroids
    - for $i=1$ to $m$
      - $c^{(i)}$ := index (from $1$ to $K$) of cluster centroid closet to $x^{(i)}$
  - Move cluster centroids
    - for $k=1$ to $K$
      - $\mu_{k}$ := average (mean) of points assigned to cluster $k$
- If one cluster has zero examples in it, eliminate this cluster
  - Alternatively just randomly re-initialize that cluster centroids
- K-means for clusters that are not well separated
  - Can still be used for continuous value ranges with useful results, e.g. t-shirt sizing
### K-means optimization objective
- $c^(i)$ = index of cluster (1,2,...K) to which example $x^(i)$ is currently assigned
- $\mu_k$ = cluster centroid $k$
- $\mu_{c^i}$ = cluster centroid of cluster to which example $x^(i)$ has been assigned
- $J(c^{(1)},...,c^{(m)},\mu_{1},...,\mu_{K}) = \frac{1}{m}\displaystyle\sum_{i=1}^{m}||x^{(i)}-\mu_{c^{(i)}}||^{2}$
- Cost function (distortion function) wants to minimize the "distance" from all the examples in a cluster to its centroid, by moving the centroid around
- Minimizing $J$ guarantees that the algorithm will converge eventually
### Initializing K-means
- Random initialization
  - Choose $K<m$ aka less centroids than training examples
  - Randomly pick $K$ training examples.
  - Set centroids equal to examples
    - Might result in clustering based on local optimizations
- Run K-means multiple times to find the best solution by comparing the outcome $J$
- Results often in lower cost $J$ compared to running it once
- Running it 50 - 1000 times with random initializations is a good value to pick
### Choosing the number of clusters / What is the right value of K?
- Clustering in the context of unsupervised learning has no "right" answer
- No recommended approach
  - Elbow method
    - Plot cost function $J$ as a function of the numbers of clusters
    - However, often there is "no clear elbow"
- Recommended approach
  - Often, you want to get clusters for some later downstream purpose
  - Evaluate K-means based on how well it performs on that later purpose
  - Example, do you want to end up with 3 or 5 different t-shirt sizes?
## Anomaly detection
### Finding unusual events
- Most common way to do anomaly detection is by doing 'density estimation'
- Probability of $x$ being seen in dataset
  - If $p(x_{test}) < \epsilon$
    - -> low probability of being in the dataset, most likely an anomaly
  - If $p(x_{test}) >= \epsilon$
    - -> high probability of being in the dataset, looks ok/normal
  - $\epsilon$ should be small number as threshold
- Anomaly detection example:
  - fraud detection based on users activities
  - manufacturing based on features of the product
  - monitor computers in data centers based on features of machine/computer
### Gaussian (normal) distribution
- Also called "normal distribution" or bell-shaped distribution
- Given a number $x$, probability of $x$ is determined by a Gaussian with mean $\mu$, variance $\sigma^{2}$
  - $p(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x-\mu)^{2}}{2\sigma^{2}}}$
- Area under the bell curve always has to be $1$
- Maximum likelihood for $\mu$, $\sigma$
  - $\mu=\frac{1}{m}\displaystyle\sum_{i=1}^{m}x^{(i)}$
  - $\sigma^{2}=\frac{1}{m}\displaystyle\sum_{i=1}^{m}(x^{(i)}-\mu)^2$
### Anomaly detection algorithm
- Each example $\overrightarrow{x}^{(i)}$ has $n$ features
- $p(\overrightarrow{x})$ is product of all $p(x)$ for all features with their own $\mu$, $\sigma$
- $\displaystyle\prod_{i=1}^{m}p(x_{j};\mu_{j},\sigma_{j}^{2})$
- How to build an anomaly detection algorithm?
1. Choose $n$ features $x_{i}$ that you think might be indicative of anomalous examples.
2. Fit parameters $\mu_{1,}...,\mu_{n},\sigma^{2}_{1},...,\sigma^{2}_{n}$ with "maximum likelihood" formulas from above
3. Given new example $x$, compute $p(x)$
4. See if $p(x) < \epsilon$
### Developing and evaluating an anomaly detection system
- real-number evaluation, is a number that tells you if a change made the algorithm better or worse
  - assumes we have some labeled data, of anomalous and non-anomalous examples
  - allows creation of cross validation set and test set
- allows you to quantify how many anomalous examples are recognized by the algorithm in the cross validation set
- it is still considered "unsupervised learning" since the examples have no labels
- Alternatively: don't create test set, but just use cross validation set
  - makes sense when number of anomalous examples is small
  - downsides:
    - not possible to tell how well it will do on real data in the future
    - higher risk of overfitting
- Algorithm evaluation
  - Fit model $p(x)$ on training set $x^{(1)},x^{(2)},...,x^{(m)}$
  - On cross validation/test example $x$, predict if anomaly or not based on comparison with $\epsilon$
  - Possible evaluation metrics:
    - True positive, false positive, false negative, true negative
    - Precision/Recall
    - $F_{1}$ score
  - Use cross validation set to choose parameter $\epsilon$
### Anomaly detection vs. supervised learning
- If you have anomalous and non-anomalous examples, why not use supervised learning?
- Anomaly detection, use if
  - Very small number of positive examples (0-20 is common)
  - Large number of negative examples
  - Many different "types" of anomalies. Hard for any algorithm to learn from positive examples, future anomalies may look very different
  - e.g. works for
    - (financial) fraud
    - manufacturing, unseen defects
    - monitoring machines in a data center
- Supervised learning, use if
  - Large number of positive and negative examples
  - Enough positive examples for algorithm to get a sense what positive examples are like, future positive examples likely to be similar to ones in training set.
  - e.g. works for
    - email spam classification
    - manufacturing, known+seen defects
    - weather prediction
    - diseases classification
### Choosing what features to use
- Transform non-gaussian features to have a "more gaussian form" e.g. by replacing $x$ with $log x$
- Error analysis for anomaly detection
  - If $p(x)$ is comparable for normal and anomalous examples, try to identify/add another feature for better distinction
- Try to choose features that might take on unusually large or small values in the event of an anomaly
- Try to combine multiple features into one which indicate some anomalous behavior
# Week 2 - Recommender systems
## Collaborative filtering
### Making recommendations
- Predicting movie ratings, 0 to 5 stars, users given ratings to items (movies in this context)
- $n_{u}$ = no. of users
- $n_{m}$ = no. of movies
- $r(i,j)$ = 1 if user $j$ has rated movie $i$
- $y^(i,j)$ = rating given by user $j$ to movie $i$, defined only if $r(i,j) = 1$
### Using per-item features
- $n$ = numbers of features
- $x^{(i)} = feature vector for movie $i$
- For user j: predict j's rating for movie $i$ as $w^{(j)} x^{(i)}+b^{(j)}$
  - just like linear regression
- Cost function
  - $m^{(j)}$ = no. of movies rated by user $j$
  - $J\begin{pmatrix} w^{(i)},...,w^{(n_{u})}\\ b^{(i)},...,b^{(n_{u})} \end{pmatrix} = \frac{1}{2} \displaystyle\sum_{j=1}^{n_{u}}\displaystyle\sum_{i:r(i,j)=1}(w^{(j)} x^{(i)}+b^{(j)}-y^{(i,j)})^{2} + \frac{\lambda}{2}\displaystyle\sum_{j=1}^{n_{u}}\displaystyle\sum_{k=1}^{n}(w_{k}^{(j)})^{2}$
  - Find parameters $w^{(j)},b^{(j)}$ for all users, so that the predication over all movies ($w^{(j)} x^{(i)}+b^{(j)}$) they have rated becomes as close as possible to the actual given rating $y^(i,j)$
### Collaborative filtering algorithm
- features are unknown, we derive them from multiple ratings (aka from different users) for the same item/movie
  - term "collaborative" refers to the fact that ratings from multiple users were given
- minimize resulting cost function via gradient descent where J becomes $J(w,b,x)$
  - $x$ now also becomes a parameter
### Binary labels
- How to go from discrete labels (5 star rating) to binary labels (like, not like)?
  - Similar approach compared to going from linear to logistic regression
- Binary label meanings in general
  - 1 - engaged after being shown item
  - 0 - did not engage after being shown item
  - ? - item not yet shown
## Recommender systems implementation detail
### Mean normalization
- Take all ratings and compute the average rating for all given ratings (e.g. condensing matrix to vector)
- Subtract average value from all ratings given which become the new values for $y^{(i,j)}$
  - implies that the initial guess how something will be rated is the average (instead of 0) e.g. when having a new user
- Makes algorithm faster, but also result in better predictions, especially for new users which have not given many ratings yet
### TensorFlow implementation of collaborative filtering
- tf.variables are the parameters we want to optimize
  - also called "auto diff" or "auto grad"
- tf.GradientTape records the sequence of steps needed to compute cost J
- use assign_add to modify value of tf.variable
### Finding related items
- The features $x^{(i)}$ of item $i$ are quite hard to interpret.
- To find other items related to to it, find item k with $x^{(k)} similar to x^{(i)}$
  - $\displaystyle\sum_{l=1}^{n}(x_{l}^{(k)}-x_{l}^{(i)})^2$, find smallest distance
- Limitations of Collaborative Filtering
  - Cold start problem, how to
    - rank new items that few users have rated?
    - show something reasonable to new users who have rated few items?
  - Use side information about items or users:
    - Item: genre, movie stars, studio
    - User: Demographics, expressed preferences
## Content-based filtering
### Collaborative filtering vs Content-based filtering
- Collaborative filtering: recommend items to you based on ratings of users who gave similar ratings as you
- Content-based filtering: recommend items to you based on features of user and item to find good match
  - $r(i,j)$ = 1 if user $j$ has rated movie $i$
  - $y^(i,j)$ = rating given by user $j$ to movie $i$, defined only if $r(i,j) = 1$
  - Make use of features of users AND item
    - user features: age, gender, country, movies watched etc
    - movie features: year, genre, reviews, average rating
- Learning to match
- Compute vectors $v_u$ for the users and $v_m$ for the items, take dot product between them: $v_u \cdot v_m$
### Deep learning for context-based filtering
- Two neural networks, compute $v_u$ and $v_m$ separately, then take dot product for the prediction
- Can be pre-computed
### Recommending from a large catalog
- How to efficiently make recommendation from a large catalog?
- Two steps
  - Retrieval
    - Generate large list of plausible item candidates
    - How many items to retrieve?
      - Retrieving more items results in better performance, but slower recommendations
      - Carry out offline experiments to analyze trade-off
  - Ranking
    - Combine retrieved items into list, removing duplicates and items already watched/purchased
# Week 3 Reinforcement learning
## Reinforcement learning introduction
### What is reinforcement learning ?
- Given a state $s$, find action $a$ to reach a goal
- Reward function tells if outcome of action is good or not
- Tell "what" to do instead of telling "how" to do it
### Mars rover example
- Each state is associated with its own reward
- Terminal state -> "end of run", no more rewards after that (e.g. for a given day)
- (s,a,R(s),s')
- state, action, reward, new state
### The return in reinforcement learning
- Concept of return captures the trade-off between reward and effort
- Full reward at first step, little bit lesser reward at second step etc
- Sum of the rewards that the system gets, weighted by the discount factor where rewards in the far future are weighted by the discount factor raised to a high power
- Negative rewards make the system trying to avoid those or postpone them as much as possible
### Making decisions: Policies in reinforcement learning
- A policy is a function $\pi(s) = a$ mapping from states to actions, that tells you what action a to take in a given state $s$.
- Goal of reinforcement learning: Find a policy $\pi$ that tells you what action $a = \pi(s)$ to take in every state (s) so as to maximize the return.
### Review of key concepts
- states
- actions
- rewards
- discount factor $\gamma$
- return
- policy $\pi$
- Markov Decision Process (MDP)
  - The future only is based on the current state and not on anything that happened previously
  - Agent ($\pi$) -> action $a$ -> Environment/World -> state $s$ and reward $R$
## State-action value function
### State-action value function definition
- $Q(s,a)$
  - Return if you
    - start in state $s$
    - take action $a$ (once)
    - then behave *optimally after that*
  - The goal is to get the best *total* return
- Also often called the (Q-function)
- The best possible return from state $s$ is $max_a Q(s,a)$
- The best possible action in state $s$ is the action $a$ that gives $max_a Q(s,a)$
  - e.g. if you are in state 4 and going left gives $Q(s,a) = 12.5$ and going right gives $Q(s,a) = 10$, you go left
- Sometimes $Q$ function is also denoted ad $Q^*$ or "optimal $Q$ function"
### Bellman Equation
- current state + action
  - $s$ : current state
  - $a$ : current action
- next state + action
  - $s'$ : state you get to after taking action $a$
  - $a'$ : action that you take in state $s'$
- $R(s)$ : rewards of current state
- $Q(s,a)=R(s)+ \gamma max_{a'} Q(s',a')$
- Bellman equation breaks down sequence of rewards into two components
  - Reward you get right away -> $R(s)$
  - Return from behaving optimally starting from state $s'$ -> $\gamma max_{a'} Q(s',a')$
### Random (stochastic) environment
- There might be a certain chance that action $a$ actually leads to being in state $s'$
  - might result in being a different state $s'$ than anticipated (e.g. rover is still in the same state or a completely different one)
- $E$ - Expected return
- Bellman equation with expected return: $Q(s,a)=R(s)+ \gamma E[max_{a'} Q(s',a')]$
## Continuous state spaces
- How to deal with a large number of states?
### Example of continuous state space application / lunar lander
- State is a vector which many variables which can take discrete values
- actions:
  - do nothing
  - left thruster
  - main thruster
  - right thruster
- $ s = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \\ \theta \\ \dot{\theta} \\ l \\ r\end{bmatrix}$
 - position $x$ and $y$, how far to the left or right
 - velocity $\dot{x}$ and $\dot{y}$, how fast is it moving in each direction
 - angel of the rover $\theta$ and $\dot{\theta}$ to the left or right
 - boolean $l$ and $r$ indicating if the left leg or right leg is grounded
- Reward function
  - Getting to landing pad: +100 up to +140
    - Additional reward to moving toward/away from the pad
  - Crash: -100
  - Soft landing: +100
  - Leg grounded: +10
  - Fire main engine: -0.3
  - Fire side thruster: -0.03
- Learn a policy $\pi$ that, given $s$ pick action $a = \pi(s)$ so as to maximize the return
- Use a large value for $\lambda = 0.985$
### Learning the state-value function / Deep Reinforcement Learning
- Basic approach: train neural network to approximate the state action value function $Q(s,a)$
- Combine 8 states and 4 possible inputs (each encoded as one-hot feature vector) to input vector with 12 inputs
- Output layer with 1 output $y$
- $Q(s,a)$ becomes the target value $y$, goal is to pick the action that maximizes Q(s,a) aka $y$
- reinforcement learning is different from supervised learning, however we are using a neural network inside the reinforcement learning algorithm
- How to train neural network to output $Q(s,a)$?
  - use Bellman's equations to create a training set with lots of examples $x$ and $y$
  - then apply supervised learning
- How to get training set for $x$ and $y$?
  - $Q(s,a)$ -> $s$ and $a$ are $x$
  - $R(s)+ \gamma max_{a'} Q(s',a')$ -> is $y$
  - "trying out" different actions gives us examples for different $y$
    - each one of them is one training example
- Initially we don't know the value of $Q(s',a')$
  - We take a guess initially
- Full Learning algorithm
  - Initialize neural network randomly as guess of $Q(s,a)$
  - Repeat
    - Take actions in the lunar lander. Get ($s,a,R(s),s'$)
    - Store 10000 most recent ($s,a,R(s),s'$) tuples.
      - aka the Replay Buffer
    - Train neural network:
      - Create training set of 10000 examples using
        - $x = (s,a)$ and $y = R(s)+ \gamma max_{a'} Q(s',a')$
      - Train $Q_{new}$ such that $Q_{new}(s,a) \approx y$
      - Set $Q = Q_{new*}$
- Also called DQN algorithm, Deep Q-Network because we use deep learning and neural network to train a model to learn the Q function
### Algorithm refinement: Improved neural network architecture
- With the current approach, we would have to run the NN four times, once for each possible action, and then pick the action resulting in the largest $Q$ value.
  - which is inefficient
- Better to have a single NN which outputs all four values at the same time
- NN has 8 inputs in the input layer (the state) and 4 units in the output layer (the different actions)
 - each output has the $Q(s,a)$ for one action, then pick action $a$ which maximizes $Q(s,a)$
### Algorithm refinement: $\epsilon$-greedy policy
- How to pick actions while still learning?
- Options:
  - 1. Pick the action $a$ that maximizes $Q(s,a)$
  - 2.
    - With probability 0.95, pick the action $a$ that maximizes $Q(s,a)$.
      - Sometimes called a "greedy" action or "exploitation step"
    - With probability 0.05, pick an action $a$ randomly.
      - Sometimes called "exploration" step
      - $\epsilon$-greedy policy ($\epsilon = 0.05$)
- Reinforcement learning algorithms are more finicky when it comes to setting parameters
### Algorithm refinement: mini-batch and soft updates
- Mini-batches
  - Problem: with very large dataset/example size (e.g. 100 million), every step of gradient descent becomes computationally expensive/slow
  - Idea with mini-batches:
     - don't take all 100 million examples for each step, but e.g. $m'$ with just 1000 examples
     - then run multiple iterations with one mini batch at a time
  - On average, mini-batch will eventually arrive at the global minimum. May take more iterations, but each iteration is much cheaper/quicker
  - That means for DQN/lunar lander, don't take all 10000 examples from the replay buffer, but just e.g. 1000
- Soft updates
  - Problem: setting $Q = Q_{new*}$ can be a very abrupt change, potentially replacing it with something worse
  - W = $0.01 W_{new} + 0.99 W$
    - Don't just replace $W$ but "blend in" a bit of $W_{new}$ with the existing $W$
### The state of reinforcement learning
- Limitations
  - Much easier to get to work in a simulation than in the real world / real robot.
  - Far fewer reinforcement learning applications than supervised and unsupervised learning.
  - Exciting research direction with potential for future applications
### Papers to read
-  Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).
-  Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. Continuous Control with Deep Reinforcement Learning. ICLR (2016).
-  Mnih, V., Kavukcuoglu, K., Silver, D. et al. Playing Atari with Deep Reinforcement Learning. arXiv e-prints. arXiv:1312.5602 (2013).