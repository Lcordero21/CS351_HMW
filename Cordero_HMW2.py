import time
import random
import numpy as np
import matplotlib.pyplot as plt

def time_algorithm(algo, arr):
    start = time.time()
    algo(arr.copy())
    return time.time() - start

def make_array(n):
    random.seed(42)
    final_array = random.sample(range(1,n+1),n)
    return(final_array)