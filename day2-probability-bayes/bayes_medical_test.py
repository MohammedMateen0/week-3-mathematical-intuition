import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

p_desease=0.01
p_no_desease=1-p_desease

p_positive_given_desease=0.99
p_positive_given_no_desease=0.05

p_positive=(p_desease*p_positive_given_desease)+(p_no_desease*p_positive_given_no_desease)

p_desease_given_pos=(p_positive_given_desease*p_desease)/p_positive

print(f'''
P(positive test) = {p_positive:.4f}
P(desease | positive) = {p_desease_given_pos:.4f}
      ''')