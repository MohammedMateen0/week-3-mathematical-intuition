import numpy as np

np.random.seed(42)
population=np.random.exponential(scale=2,size=100_000)

sample_means=[np.random.choice(population,size=30).mean() for _ in range(3000) ]

print(f'''
population mean:{population.mean():.4f}
Sample mean -mean : {np.mean(sample_means):.3f}
std:{np.std(sample_means):.3f}
''')