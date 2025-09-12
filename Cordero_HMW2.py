import time
import random
import numpy as np
import matplotlib.pyplot as plt

def time_algorithm(algo, arr):
    start = time.time()
    algo(arr.copy())
    return time.time() - start