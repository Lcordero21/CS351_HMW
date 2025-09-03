import time
import random
import matplotlib.pyplot as plt
import statistics



def time_algorithm(algo, arr):
    """"""
    start = time.time()
    algo(arr.copy())
    return time.time() - start


# Starter code
def selection_sort(arr):
    new_array = arr
    for i in range(len(new_array)):
        minimum =i
        for m in range (i+1,len(new_array)):
            if new_array[m] < new_array [minimum]:
                minimum = m
        new_array [i], new_array [minimum] = new_array [minimum], new_array [i]
    return new_array

def merge_sort(arr):
    A = split(arr)[0]
    B = split(arr)[1]

    C = A
    D = B

    if len(A)== 1 and len(B) == 1:
        return merge(A,B)
    if len(A) > 1:
       C = merge_sort (A)
    if len(B) > 1:
       D = merge_sort (B)
    return merge(C,D)
    
def split(arr):
    half_point = len(arr)//2
    return [arr[0:half_point],arr[half_point:len(arr)]]


def merge(C, D):
        i = j = 0
        B = []
        
        while i < len(C) and j < len(D):
            if C[i] < D[j]:
                B.append(C[i])
                i += 1
            else:
                B.append(D[j])
                j += 1
        
        # Add remaining elements
        B.extend(C[i:])
        B.extend(D[j:])
        return B

def make_array(n):
    random.seed(42)
    final_array = random.sample(range(1,n+1),n)
    return(final_array)

def run(n):
    selection_med = []
    merge_med = []
    for i in range (len(n)):
        array = make_array(n[i])
        
        for j in range (5):
            temp_array1 = []
            temp_array1.append(time_algorithm(selection_sort,array))
        selection_med.append(statistics.median(temp_array1))

        for k in range (5):
            temp_array2 = []
            temp_array2.append(time_algorithm(merge_sort, array))
        merge_med.append(statistics.median(temp_array2))

    print("Selection Sort Median Time", selection_med)
    print ("Merge Sort Median Time", merge_med)



run([100, 500, 1000, 5000])