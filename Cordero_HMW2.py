import time
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt

temp_array_med_count = []

def time_algorithm(algo, arr, target):
    """
    Input: Algorithm, Array, Target Value

    Output: Time

    Purpose:
    To time the time taken to execute the algorithm in seconds by taking the time of the device when the algorithm
    is started to the time of the device when the algorithm finishes. It returns the difference of the two to get the 
    execution time.
    """
    start = time.time()
    temp_array_med_count.append(algo(arr.copy(),target))
    return time.time() - start

def make_array(n):
    """
    Input: Integer

    Output: Array

    Purpose: 
    Makes an random, unsorted array of the inputted length. There are duplicates, I will talk more about why
    in the report.

    Error:
    TypeError if n is not an integer
    """
    if type(n) != int:
        raise TypeError ("Please input an integer")

    final_array = []
    random.seed(random.randint(0,50))
    for num in range (n):
        final_array.append(random.randint(0,n))
    return(final_array)

def three_sum(arr, target):
    """
    Input: Array and Target Value 

    Output: Boolean

    Purpose:
    Utilizing nested loops, this is a brute force method of seeing if any three integers in the given array adds
    up to the target. The function accounts for arrays

    Errors:
    I have included a couple of errors that would arise for various cases (such as a TypeError if someone tries to
    input anything but a list/array, a KeyError if the sum target is less than 1, and an IndexError if the length of
    the array is less than 3). 

    Limitations:
    While the algorithm records and prints the count of how many loop operations it does, and returns the count, I do 
    have to admit that I don't properly use the count in my overall analysis. Later in my code, I manually inputted 
    the count of one of my runs (after doing a bunch of test runs), I have chosen not to adjust my code to do this 
    for me, because I nearly ran out of time by the time I noticed...
    """

    n = len(arr)-2
    count = 0
    if type(arr) != list:
        raise TypeError("Please input an array.")
    if target < 1:
        raise KeyError("Target is too small, must be greater than -1.")
        return False
    if len(arr) < 3:
        raise IndexError("Length of array must be greater than 2.")
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range (j+1, n):
                count += 1
                if arr[i] + arr[j] + arr[k] == target:
                    print("Length of Array", n+2, count, True)
                    return count
    print("Length of Array", n+2,count, False)
    return count

def run(n,target):
    """
    Input: Array

    Output: Three-Sum Algorithm Median Times per n

    Purpose: 
    To run the program itself. This creates an array, starts the timer, and runs the algorithm a set 
    number of times.

    Errors:
    TypeError arises if the type of n is not an array, if the target is not an integer,
    and if the type of a specific index of n is not an integer.
    """
    if type(n) != list:
        raise TypeError
    
    if type(target) != int:
        raise TypeError
    count_med = []
    three_sum_med = []
    for i in range (len(n)):

        if type(i) != int:
            raise TypeError
        for j in range (10):
            array = make_array(n[i])
            temp_array1 = []
            temp_array1.append(time_algorithm(three_sum,array,target))
        count_med.append(statistics.median(temp_array_med_count))
        three_sum_med.append(statistics.median(temp_array1))
    print("Three Sum Median Time", three_sum_med)
    return(three_sum_med,count_med)

default_lengths = [50,100,200,400,800]
med_time,med_count= run(default_lengths,52)

print("Median Count", med_count)
def plot():
    """
    Input: None

    Output: None

    Purpose: 
    Makes the graphs I need for my report. I calculated the theoretical growth by 
    making the next value 8 times slower than the previous (with the first value being the actual 1st 
    measurement)
    """
    theory_time = []
    theory_time.append(med_time[0])

    theory_count = []
    theory_count.append(med_count[0])

    for i in range (len(default_lengths)-1):
        theory_time.append(theory_time[i]*8)
        theory_count.append(theory_count[i]*8)

    print ("Theory stuff:", theory_time, theory_count)

    plt.plot(default_lengths, med_time, label="Actual", marker="*")
    plt.plot(default_lengths, theory_time, label = "Theoretical", marker = "*")
    plt.legend()
    plt.xlabel("N Value")
    plt.ylabel("Median Run Times in Seconds")
    plt.title("Three-Sum: Linear Graph of N vs Run Times")
    plt.show()

    plt.plot(default_lengths, med_count, label = "Actual", marker = "*")
    plt.plot(default_lengths, theory_count, label = "Theoretical", marker = "*")
    plt.legend()
    plt.xlabel("N Value")
    plt.ylabel ("Median Operation Counts")
    plt.title("Three-Sum: Linear Graph of N vs Operation Count")
    plt.show()  

plot()



