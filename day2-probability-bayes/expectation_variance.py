import numpy as np

ratings=np.array([1,2,3,4,5])
probs=np.array([0.05,0.10,0.20,0.40,0.25])

expected_rating=np.dot(ratings,probs)
varience=np.dot((ratings-expected_rating)**2,probs)
std_dev=np.sqrt(varience)

print(f'''
Expected rating: {expected_rating:.2f}
Varience: {varience:.2f}
Standard deviation: {std_dev:.2f}
''')