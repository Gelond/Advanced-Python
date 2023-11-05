#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Python performance exercises</h1>

# ## Python best practices exercises

# ### Exercise 1
# 
# considering the following function for concatenating list strings with delimiter.

# In[18]:


import numpy as np


# In[19]:


def ft_concatenate(l_strings, d):
    """concatenate list of strings into one string separated by delimiter"""
    res = l_strings[0]
    for e in l_strings[1:]:
        res = res + d + e
    return res


# - profile the function and identify the bottlenecks.
# - improve speed up of the function
# *Hint: you may need to look to the string functions in python documentation*

# In[20]:


# write your code here

import random
import string
import timeit

def generate_strings():
    n = 10000
    length = 10
    random_strings = []

    for _ in range(n):
        random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
        random_strings.append(random_string)

    return random_strings
    
random_strings=generate_strings()
d = " "
get_ipython().run_line_magic('timeit', 'ft_concatenate(random_strings,d)')


# In[21]:


get_ipython().run_line_magic('prun', '-s cumulative ft_concatenate(random_strings,d)')


# In[26]:


import cProfile
cProfile.run("ft_concatenate(random_strings,d)", sort="cumulative")


# In[17]:


pip install line_profiler


# In[28]:


get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('lprun', '-f ft_concatenate ft_concatenate(random_strings,d)')


# In[43]:


pip install memory_profiler


# In[53]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[57]:


get_ipython().run_line_magic('lprun', '-f ft_concatenate ft_concatenate(random_strings, d)')


# In[65]:


from memory_profiler import profile


# In[66]:


@profile
def ft_concatenate(l_strings, d):
    """concatenate list of strings into one string separated by delimiter"""
    res = l_strings[0]
    for e in l_strings[1:]:
        res = res + d + e
    return res


# In[70]:


get_ipython().system('jupyter nbconvert --to script python_performance_assign-Copy1.ipynb')


# ### Exercise 2
# 
# In this exercise you will solve the following problem using two methods bruteforce method, and fast method.
# 
# **Problem:** You are given a list of n integers, and your task is to calculate the number of distinct values in the list.
# 
# **Example**
# - Input:
# 5
# 2 3 2 2 3
# 
# - Output:
# 2
# 
# **Implement the following methods:**
# 
# 1. **bruteforce method:** create an empty list and start adding items for the given list without adding the previous item add, at the end the result list will contain unique values, print lenght of the list and you are done. 
# 2. **fast method** think of using Set data structure.
# 
# - time the two methods, what do you think?

# In[ ]:


import time

# Brute Force Method
def distinct_count_bruteforce(arr):
    unique_values = []
    for item in arr:
        if item not in unique_values:
            unique_values.append(item)
    return len(unique_values)

# Fast Method using Set
def distinct_count_fast(arr):
    return len(set(arr))

# Input list
arr = [5, 2, 3, 2, 2, 3]

# Measure execution time for the brute force method
start_time = time.time()
result_bruteforce = distinct_count_bruteforce(arr)
end_time = time.time()
bruteforce_time = end_time - start_time

# Measure execution time for the fast method
start_time = time.time()
result_fast = distinct_count_fast(arr)
end_time = time.time()
fast_method_time = end_time - start_time

print("Distinct Values (Brute Force Method):", result_bruteforce)
print("Execution Time (Brute Force Method):", bruteforce_time)

print("Distinct Values (Fast Method):", result_fast)
print("Execution Time (Fast Method):", fast_method_time)


# In[14]:


# bruteforce method
def bruteforce_method(arr):
    unique_values = []
    for i in arr:
        if i not in unique_values:
            unique_values.append(i)
    return len(unique_values)


# In[15]:


# fast method
def fast_method(arr):
    return len(set(arr))


# In[16]:


import time
# Create a random list of numbers for testing
list = [2,3,2,2,3]
# time the two methods
start = time.time()
bruteforce_method(list)
end = time.time()

print(end-start)


# ## Cython exercises

# ### Exercise 1

# 1. load the cython extension.

# In[12]:


get_ipython().run_line_magic('cython', '')


# 2. Considering the following polynomial function:

# In[12]:


def poly(a,b):
    return 10.5 * a + 3 * (b**2)


# - Create an equivalent Cython function of `poly` with name `poly_cy`.

# In[ ]:





# 3. time the performance of Python and Cython version of the function, what is the factor of speed up between the two verions.

# In[13]:


# write your code here


# 4. Now let's work on another example using loop.
#     - rewrite the same function below fib that calculates the fibonacci sequence using cython, but now try to add type for the variables used inside it, add a prefix `_cy` to your new cython function.

# In[14]:


def fib(n):
    a, b = 1, 1
    for i in range(n):
        a, b = a + b, a

    return a


# In[15]:


# write your code here


# - time the two function for fibonacci series, with n = 20, what is the factor of speed now, What do you think?

# In[16]:


# write your code here


# 5. Recursive functions are functions that call themselves during their execution. Another interesting property of the Fibonacci sequence is that it can be written as a recursive function. That’s because each item depends on the values of other items (namely item n-1 and item n-2)
# 
# - Rewrite the fib function using recursion. Is it faster than the non-recursive version? Does Cythonizing it give even more of an advantage? 

# In[17]:


# write your code here


# ### Exercise 2
# 
# - Monte Carlo methods are a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results. 
# - One of the basic examples of getting started with the Monte Carlo algorithm is the estimation of Pi.
# 
# **Estimation of Pi**
# 
# - The idea is to simulate random (x, y) points in a 2-D plane with domain as a square of side 1 unit. 
# - Imagine a circle inside the same domain with same diameter and inscribed into the square. 
# - We then calculate the ratio of number points that lied inside the circle and total number of generated points. 
# - Refer to the image below:
# 
# ![demo](../data/MonteCarloPlot.png)
# 
# We know that area of the square is 1 unit sq while that of circle is $\pi \ast  (\frac{1}{2})^{2} = \frac{\pi}{4}$. Now for a very large number of generated points,
# 
# ![demo](../data/MonteCarloCalc.png)
# 
# 
# ## The Algorithm
# 
# 1. Initialize cile_points, square_points and interval to 0.
# 2. Generate random point x.
# 3. Generate random point y.
# 4. Calculate d = x*x + y*y.
# 5. If d <= 1, increment circle_points.
# 6. Increment square_points.
# 7. Increment interval.
# 8. If increment < NO_OF_ITERATIONS, repeat from 2.
# 9. Calculate pi = 4*(circle_points/square_points).
# 10. Terminate.

# **Your mission:** time the function `monte_carlo_pi`, identify the bottlenecks and create a new version using cython functionality to speed up monte carlo simulation for PI, use 100,000 points and compare the speed up factor between python and cython, considering the following optimizations:
# - add type for variables used.
# - add type for the function
# - use c rand function instead of python rand function.
#  
# *Hint: you can import function from C libraries using the following approach `from libc.<name of c library> cimport <library function name>`, replace the holders `<>` with the right identities for the current problem*

# In[18]:


import random
def monte_carlo_pi(nsamples):
    pi = 0.
   # Implement your code here
    return pi


# ## Numba exercises

# ### Exercise 1
# 
# Previously we considered how to approximateby Monte Carlo.
# 
# - Use the same idea here, but make the code efficient using Numba.
# - Compare speed with and without Numba when the sample size is large.

# In[ ]:


# Your code here


# ### Exercise 2
# 
# In the [Introduction to Quantitative Economics](https://python.quantecon.org/intro.html) with Python lecture series you can learn all about finite-state Markov chains.
# 
# For now, let's just concentrate on simulating a very simple example of such a chain.
# 
# Suppose that the volatility of returns on an asset can be in one of two regimes — high or low.
# 
# The transition probabilities across states are as follows ![markov](../data/markov.png)
# 
# For example, let the period length be one day, and suppose the current state is high.
# 
# We see from the graph that the state tomorrow will be
# 
# - high with probability 0.8
# 
# - low with probability 0.2
# 
# Your task is to simulate a sequence of daily volatility states according to this rule.
# 
# Set the length of the sequence to `n = 1_000_000` and start in the high state.
# 
# Implement a pure Python version and a Numba version, and compare speeds.
# 
# To test your code, evaluate the fraction of time that the chain spends in the low state.
# 
# If your code is correct, it should be about 2/3.
# 
# Hints:
# 
# - Represent the low state as 0 and the high state as 1.
# 
# - If you want to store integers in a NumPy array and then apply JIT compilation, use `x = np.empty(n, dtype=np.int_)`.
# 
