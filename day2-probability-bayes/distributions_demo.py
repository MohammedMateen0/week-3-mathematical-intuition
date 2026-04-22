import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


bern=stats.bernoulli(p=0.3)
print(f'''
Bernoulli distribution for p=0.3:
mean={bern.mean():.4f}
varience={bern.var():.4f}    
''')

binom=stats.binom(n=100,p=0.3)
print(f'''
Binomial distribution for n=10,p=0.3:
mean={binom.mean():.4f}
varience={binom.var():.4f}
Exactly 30 successes in 100 trials: {binom.pmf(30):.4f}
Between 25 to 35 successes:{binom.cdf(35)-binom.cdf(24):.4f}
      ''')

norm=stats.norm(loc=4.0,scale=0.5)
print(f'''
P(rating > 4.5): {1-norm.cdf(4.5):.4f}
95th percentile : {norm.ppf(0.95):.3f}
''')

poisson=stats.poisson(mu=12)
print(f'''
P(exactly 15 orders in an hour)={poisson.pmf(15):.4f}
P(more than 20 orders)={poisson.cdf(20):.4f}
''')