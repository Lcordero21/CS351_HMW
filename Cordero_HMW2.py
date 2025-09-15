import time
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt

def time_algorithm(algo, arr, target):
    """
    Input: Algorithm, Array
    Output: Time
    """
    start = time.time()
    algo(arr.copy(),target)
    return time.time() - start

def make_array(n):
    """
    Input: Integer
    Output: Array
    """
    final_array = []
    random.seed(random.randint(0,50))
    for num in range (n):
        final_array.append(random.randint(0,n))
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
                    print("Length of Array", n+2, count, True)
                    return count
    print("Length of Array", n+2,count, False)
    return count

def run(n):
    """
    Input: Array
    Output: Lots of stuff
    """
    three_sum_med = []
    for i in range (len(n)):
        for j in range (10):
            array = make_array(n[i])
            temp_array1 = []
            temp_array1.append(time_algorithm(three_sum,array,52))
        three_sum_med.append(statistics.median(temp_array1))
    print("Three Sum Median Time", three_sum_med)
    return(three_sum_med)

default_lengths = [50,100,200,400,800]
med_time= run(default_lengths)

#Ignore how impractical this is, but I really don't want to write more code. So I'm just inputting the count from one of my runs.
med_count = [statistics.median([12,64,46,202,126,292,46,202,126,292]),statistics.median([6219,6732,9596,10081,587,22360,9596,10081,587,22360]),statistics.median([76461,17493,327,58292,2133,415,39338,95983,58689,60131]),statistics.median([586400,157015,157015,157015,157015,157015,157015,157015,157015,157015]),statistics.median([1028452,3573990,649485,5593380,10066125,2209630,5946425,16297971,87323,688494])]

plt.plot(default_lengths, med_time, marker="*")
plt.xlabel("N Value")
plt.ylabel("Median Run Times in Seconds")
plt.title("Three-Sum: Linear Graph of N vs Run Times")
plt.show()

plt.plot(default_lengths, med_count, marker = "*")
plt.xlabel("N Value")
plt.ylabel ("Median Operation Counts")
plt.title("Three-Sum: Linear Graph of N vs Operation Count")
plt.show()




