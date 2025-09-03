import time
import random
#import matplotlib

def time_algorithm(algo, arr):
    start = time.time()
    algo(arr.copy())
    return time.time() - start


# Starter code
def selection_sort(arr):
    new_array = arr
    for i in range(len(array)):
        pass
    pass

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

def start(input):
    #loop each array and start timer

    pass
array = make_array(12)
print(array)
print(merge_sort(array))