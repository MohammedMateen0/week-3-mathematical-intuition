# Week 3 — Day 2: Probability, Bayes & Statistical Intuition

## 🎯 Objective

Build deep intuition for probability concepts used in Data Science:

* Bayes Theorem
* Probability distributions (Bernoulli, Binomial, Normal, Poisson)
* Central Limit Theorem (CLT)
* Expectation & Variance
* Naive Bayes classification

---

## 📌 Why This Matters

In interviews, candidates are often tested on:

* Interpreting probabilities correctly
* Avoiding common statistical misconceptions
* Applying Bayes reasoning in real-world scenarios

---

## 🧠 1. Bayes Theorem (Medical Test)

We simulate a disease testing scenario:

* Disease prevalence = 1%
* Test sensitivity = 99%
* False positive rate = 5%

### Key Result

Even with a highly accurate test:

P(Disease | Positive) ≈ 0.17

### Insight

> Rare events lead to many false positives — base rate dominates.

---

## 📊 2. Probability Distributions

### Bernoulli Distribution

* Models binary outcomes
* Mean = p
* Variance = p(1 − p)

### Binomial Distribution

* Models number of successes in n trials
* Example: probability of 30 successes in 100 trials

### Normal Distribution

* Used for continuous variables
* Example: probability of rating > 4.5

### Poisson Distribution

* Models event counts in fixed intervals
* Example: number of orders per hour

⚠️ Important:
P(X > k) = 1 − CDF(k)

---

## 📈 3. Central Limit Theorem (CLT)

We simulate:

* Skewed exponential population
* Sampling repeatedly (n=30)

### Result

Sample means become normally distributed.

### Insight

> CLT explains why normal distribution appears everywhere in statistics.

---

## 📊 4. Expectation & Variance

We compute:

* Expected rating (weighted average)
* Variance and standard deviation

### Insight

> Expectation is the center of mass of a distribution.

---

## 🤖 5. Naive Bayes (Text Classification)

We implement a simple spam classifier using:

* Prior probabilities
* Conditional probabilities of words
* Log probabilities for numerical stability

### Insight

> Naive Bayes assumes conditional independence between features.

---

## ⚠️ Important Learnings

* p-value ≠ probability hypothesis is true
* Rare events distort intuition (base rate fallacy)
* Log probabilities prevent underflow
* Poisson tail probabilities require complement

---

## ▶️ Run Instructions

```bash
pip install -r requirements.txt
python bayes_medical_test.py
python distributions_demo.py
python clt_simulation.py
python naive_bayes_text.py
```

---



## 🚀 Extensions

* Add Laplace smoothing to Naive Bayes
* Visualize distributions with histograms
* Implement Bayesian updating over time

---
