#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Python performance exercises</h1>

# ## Python best practices exercises

# ### Exercise 1
# 
# considering the following function for concatenating list strings with delimiter.

# In[364]:


import numpy as np


# In[365]:


def ft_concatenate(l_strings, d):
    """concatenate list of strings into one string separated by delimiter"""
    res = l_strings[0]
    for e in l_strings[1:]:
        res = res + d + e
    return res


# - profile the function and identify the bottlenecks.
# - improve speed up of the function
# *Hint: you may need to look to the string functions in python documentation*

# In[366]:


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


# In[367]:


get_ipython().run_line_magic('timeit', 'ft_concatenate(random_strings,d)')


# In[368]:


get_ipython().run_line_magic('prun', '-s cumulative ft_concatenate(random_strings,d)')


# In[369]:


import cProfile
cProfile.run("ft_concatenate(random_strings,d)", sort="cumulative")


# In[370]:


pip install line_profiler


# In[371]:


get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('lprun', '-f ft_concatenate ft_concatenate(random_strings,d)')


# In[372]:


pip install memory_profiler


# In[373]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[374]:


get_ipython().run_line_magic('lprun', '-f ft_concatenate ft_concatenate(random_strings, d)')


# In[375]:


from memory_profiler import profile


# In[376]:


@profile
def ft_concatenate(l_strings, d):
    """concatenate list of strings into one string separated by delimiter"""
    res = l_strings[0]
    for e in l_strings[1:]:
        res = res + d + e
    return res


# In[377]:


get_ipython().system('jupyter nbconvert --to script python_performance_assign-Copy1.ipynb')


# In[378]:


def ft_concatenate(l_strings, d):
    """concatenate list of strings into one string separated by delimiter"""
    res = d.join(l_strings)
    return res


# In[379]:


get_ipython().run_line_magic('timeit', 'ft_concatenate(random_strings,d)')


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

# In[380]:


import time


# In[381]:


# bruteforce method
def bruteforce_method(arr):
    unique_values = []
    for i in arr:
        if i not in unique_values:
            unique_values.append(i)
    return len(unique_values)


# In[382]:


list = [2,3,2,2,3]
bruteforce_method(list)


# In[383]:


# fast method
def fast_method(arr):
    return len(set(arr))


# In[384]:


fast_method(list)


# In[385]:


# Create a random list of numbers for testing
n = 1000
random_list = [random.randint(1,100) for i in range(n)]


# In[386]:


# time the two methods
get_ipython().run_line_magic('timeit', 'bruteforce_method(random_list)')


# In[387]:


get_ipython().run_line_magic('timeit', 'fast_method(random_list)')


# ## Cython exercises

# ### Exercise 1

# 1. load the cython extension.

# In[388]:


get_ipython().system('pip install cython')


# In[389]:


get_ipython().run_line_magic('load_ext', 'Cython')


# 2. Considering the following polynomial function:

# In[390]:


def poly(a,b):
    return 10.5 * a + 3 * (b**2)


# - Create an equivalent Cython function of `poly` with name `poly_cy`.

# In[393]:


get_ipython().run_cell_magic('cython', '', 'cdef float poly_cy(double a, double b):\n    return 10.5 * a + 3 * (b**2)\n')


# 3. time the performance of Python and Cython version of the function, what is the factor of speed up between the two verions.

# In[394]:


a=100
b=256


# In[395]:


# write your code here

get_ipython().run_line_magic('timeit', 'poly(a,b)')


# In[396]:


get_ipython().run_line_magic('timeit', 'poly_cy(a,b)')


# 4. Now let's work on another example using loop.
#     - rewrite the same function below fib that calculates the fibonacci sequence using cython, but now try to add type for the variables used inside it, add a prefix `_cy` to your new cython function.

# In[397]:


def fib(n):
    a, b = 1, 1
    for i in range(n):
        a, b = a + b, a

    return a


# In[398]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[399]:


get_ipython().run_cell_magic('cython', '', 'cdef int fib_cy(int n):\n    cdef int a, b, i\n    a = 0\n    b = 1\n    for i in range(n):\n        a, b = a + b, a\n\n    return a\n')


# - time the two function for fibonacci series, with n = 20, what is the factor of speed now, What do you think?

# In[400]:


n = 20


# In[401]:


# write your code here
get_ipython().run_line_magic('timeit', 'fib(n)')


# In[402]:


get_ipython().run_line_magic('timeit', 'fib_cy(n)')


# 5. Recursive functions are functions that call themselves during their execution. Another interesting property of the Fibonacci sequence is that it can be written as a recursive function. That’s because each item depends on the values of other items (namely item n-1 and item n-2)
# 
# - Rewrite the fib function using recursion. Is it faster than the non-recursive version? Does Cythonizing it give even more of an advantage? 

# In[403]:


# write your code here
def fib_rec(n):
    if n == 0 or n == 1:
        return n
    else: return fib_rec(n-1) + fib_rec(n-2)
fib_rec(7)


# In[404]:


get_ipython().run_cell_magic('cython', '', 'cdef int fib_rec_cy(int n):\n    if n==0 or n==1:\n        return n\n    else:\n        return fib_rec_cy(n-1) + fib_rec_cy(n-2)\n    \n')


# In[405]:


n = 20


# In[406]:


get_ipython().run_line_magic('timeit', 'fib_rec(n)')


# In[407]:


get_ipython().run_line_magic('timeit', 'fib_rec_cy(n)')


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

# In[408]:


import random
def monte_carlo_pi(nsamples):
    pi = 0.
   # Implement your code here
    
    cile_points = 0
    square_points = 0
    increment = 0
    while increment < nsamples:
        d = 0.
        x = random.uniform(-1.,1.)
        y = random.uniform(-1.,1.)
        d = x*x + y*y
        if d <= 1:
            cile_points+=1
        square_points+=1
        increment+=1
            
    pi = 4*(cile_points/square_points)
    return pi


# In[409]:


nsamples = 100000


# In[410]:


monte_carlo_pi(nsamples)


# In[411]:


get_ipython().run_cell_magic('cython', '', 'from libc.stdlib cimport rand\ncdef float monte_carlo_pi_cy(int nsamples):\n    cdef int cile_points, square_points, increment\n    cdef float d, pi, x, y\n    pi = 0.\n    cile_points = 0\n    square_points = 0\n    \n    for increment in range(nsamples):\n        d = 0.\n        x = rand()\n        y = rand()\n        d = x*x + y*y\n        if d <= 1: cile_points+=1\n        square_points+=1\n        \n    pi = 4*(cile_points/square_points)\n    return pi\n')


# In[412]:


get_ipython().run_line_magic('timeit', 'monte_carlo_pi(nsamples)')


# In[413]:


get_ipython().run_line_magic('timeit', 'monte_carlo_pi_cy(nsamples)')


# ## Numba exercises

# ### Exercise 1
# 
# Previously we considered how to approximateby Monte Carlo.
# 
# - Use the same idea here, but make the code efficient using Numba.
# - Compare speed with and without Numba when the sample size is large.

# In[416]:


from numba import njit


# In[418]:


# Your code here
@njit
def monte_carlo_pi(nsamples):
    pi = 0.
   # Implement your code here
    
    cile_points = 0
    square_points = 0
    increment = 0
    while increment < nsamples:
        d = 0.
        x = random.uniform(-1.,1.)
        y = random.uniform(-1.,1.)
        d = x*x + y*y
        if d <= 1:
            cile_points+=1
        square_points+=1
        increment+=1
            
    pi = 4*(cile_points/square_points)
    return pi


# In[419]:


get_ipython().run_line_magic('timeit', 'monte_carlo_pi(nsamples)')


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

# In[708]:


import numpy as np

def simulate_markov_chain(n):
    states = np.empty(n, dtype=np.int_)
    current_state = 1
    states[0] = current_state
    
    for i in range(n-1):
        if current_state == 1:
            next_state = 0 if random.random()<0.8 else 1
        else:
            next_state = 1 if random.random()<0.2 else 0
        
        current_state = next_state
        states[i+1] = current_state
    return states


# In[709]:


n = 1000000


# In[710]:


1-np.mean(simulate_markov_chain(n))


# In[711]:


@njit
def simulate_markov_chain_numba(n):
    states = np.empty(n, dtype=np.int_)
    current_state = 1
    states[0] = current_state
    
    for i in range(n-1):
        if current_state == 1:
            next_state = 0 if random.random()<0.8 else 1
        else:
            next_state = 1 if random.random()<0.2 else 0
        
        current_state = next_state
        states[i+1] = current_state
    return states


# In[712]:


get_ipython().run_line_magic('timeit', 'simulate_markov_chain(n)')


# In[713]:


get_ipython().run_line_magic('timeit', 'simulate_markov_chain_numba(n)')


# In[714]:


def simulate_markov_chain2(n):
    states = np.empty(n, dtype=np.int_)
    current_state = 1
    states[0] = current_state
    
    for i in range(n-1):
        if current_state == 1:
            next_state = np.random.choice([0,1], size=1, p=[0.8,0.2])
        else:
            next_state = np.random.choice([0,1], size=1, p=[0.9,0.1])
        
        current_state = next_state
        states[i+1] = current_state
    return states


# In[715]:


1-np.mean(simulate_markov_chain2(n))


# In[716]:


@njit
def simulate_markov_chain2_numba(n):
    states = np.empty(n, dtype=np.int_)
    current_state = 1
    states[0] = current_state
    
    for i in range(n-1):
        if current_state == 1:
            next_state = np.random.choice([0,1], size=1, p=[0.8,0.2])
        else:
            next_state = np.random.choice([0,1], size=1, p=[0.9,0.1])
        
        current_state = next_state
        states[i+1] = current_state
    return states


# In[706]:


get_ipython().run_line_magic('timeit', 'simulate_markov_chain2(n)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'simulate_markov_chain2_numba(n)')

