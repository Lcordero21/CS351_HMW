import time
import random
import numpy as np
import matplotlib.pyplot as plt

def time_algorithm(algo, arr):
    """
    Input: Algorithm, Array
    Output: Time
    """
    start = time.time()
    algo(arr.copy())
    return time.time() - start

def make_array(n):
    """
    Input: Integer
    Output: Array
    """
    random.seed(42)
    final_array = random.sample(range(1,n+1),n)
    return(final_array)

def three_sum(arr, target):
    """
    Input: Array and Target Value 
    Output: Boolean
    """
    
    n = len(arr)-2
    count = 0

    if len(arr) < 3:
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range (j+1, n):
                count += 1
                if arr[i] + arr[j] + arr[k] == target:
                    return count, True
    
    return count,False

