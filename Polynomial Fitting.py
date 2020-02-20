#!/usr/bin/env python
# coding: utf-8

# # Ashutosh Shinde

# # 1) Plotted the noisy data and the polynomial in the same figure. The values of m are selected from 2, 3, 4, 5, 6

# Here I practically implemented the polynomial fitting. I also visualized the effect of the magnitude of noise and number of samples on the accuracy of the plotted function by tuning the parameters. I have used the polyfit() function provided by the numpy package.
# The polynomial used is y = 5 * x + 20 * x * 2 + x * 3.
# I deployed the Matplotlib library package for plotting our findings using the pyplot function. 
# (Matplotlib is a python library that provides a MALTLAB like interface using a set of functions similar to MATLAB)
# Numpy will be used to carry out the calculation of the mathematical function.

# In[9]:


import numpy as np
import matplotlib.pyplot as plt


# In[10]:


def x_y_values(noise_scale, number_of_samples):
    x = 25*(np.random.rand(number_of_samples, 1) - 0.8)
    y = 5 * x + 20 * x**2 + 1 * x**3 + noise_scale*np.random.randn(number_of_samples, 1)
    return x, y
    


# In[11]:


def func_polynomial(noise_scale, number_of_samples, m):
    
    #x = 25*(np.random.rand(number_of_samples, 1) - 0.8)
    #y = 5 * x + 20 * x**2 + 1 * x**3 + noise_scale*np.random.randn(number_of_samples, 1)
    x, y = x_y_values(noise_scale, number_of_samples)
    
    plt.style.use('seaborn-whitegrid')
    plt.plot(x,y,'ro')
 
    print("type of x is:", type(x))
    print("type of y is:", type(y))
    
    x = x.flatten()
    y = y.flatten()
    
    coef = np.polyfit(x,y,m)
    print("Coefficients for the polynomial are:", coef)
    
    y_calculated = []
    
    fig = np.poly1d(coef)
    xl = np.linspace(-20,5,50)
    plt.plot(x,y,'ro',xl,fig(xl),'g')
    plt.show()
    
    for i in range(len(x)):
        y_new = fig(x[i])
        y_calculated.append(y_new)
    
    mean_sqr_error = np.square(np.subtract(y,y_calculated)).mean() 
    print("The mean square error is:", mean_sqr_error)
    return mean_sqr_error
    


# Here I have plotted the noisy data and also the polynomials for m = 1 to m = 8 

# In[12]:


mse1 = func_polynomial(100,50,1)


# In[13]:


mse2 = func_polynomial(100,50,2)


# In[14]:


#m=3
mse3 = func_polynomial(100,50,3)


# In[15]:


#m=4
mse4 = func_polynomial(100,50,4)


# In[16]:


#m=5
mse5 = func_polynomial(100,50,5)


# In[17]:


#m=6
mse6 = func_polynomial(100,50,6)


# In[18]:


mse7 = func_polynomial(100,50,7)


# In[19]:


mse8 = func_polynomial(100,50,8)


# # 2) Plotted MSE versus order m, for m = 1, 2, 3, 4, 5, 6, 7, 8. Identified the best choice of order m

# In[20]:


MSE = [mse1,mse2,mse3,mse4,mse5,mse6,mse7,mse8]
print("The MSE values for the different m values are:", MSE)


# In[21]:


print("The least MSE value is :", min(MSE))


# We see above that the least MSE value is when m = 6
# with the The mean square error of 5788.05923134297 
# 

# In[24]:


m_val = [1,2,3,4,5,6,7,8]
plt.plot(m_val,MSE,'-bo')
plt.xlabel("M values")
plt.ylabel("MSE Values")


# As we can see from the above line graph the value of m for which the MSE value is the minimum is for m =6

# # 3) Changed the variable noise_scale to 150, 200, 400, 600, 1000 respectively, re-ran the algorithm and plotted the polynomials with the m found in 2). Described the impact of noise scale to the accuracy of the returned parameters. 
# 

# In[ ]:


Here I have called the function "func_polynomial" by increasing the noise_scale from 150 to 1000. 


# In[25]:


func_polynomial(150,50,6)


# In[26]:


func_polynomial(200,50,6)


# In[27]:


func_polynomial(400,50,6)


# In[28]:


func_polynomial(600,50,6)


# In[29]:


func_polynomial(1000,50,6)


# # Here I observed that as the noise increased, the points dispersed more and the the polynomial function plotted lost its accuracy. Due to an increase in the oscillation of data, the polynomial function failed to identify the nature of the data provided. With an increase in <TT>noise_scale</TT>, the function first widened(with respect to the orignal) and then took another form to fit most of the data, however inaccurately.

# # 4) Here I changed the variable, number_of_samples to 40, 30, 20, 10 respectively, re-ran the algorithm and plotted the polynomials with the m found in 2). Described the impact of the number of samples to the accuracy of the returned parameters.

# In[30]:


func_polynomial(100,40,6)


# In[31]:


func_polynomial(100,30,6)


# In[32]:


func_polynomial(100,20,6)


# In[33]:


func_polynomial(100,10,6)


# # The polynomial function has less data to train and the polynomial function starts to lose its accuracy as the number of samples decrease. We observed that with a decrease in the number of samples there was also a decrease in the variety of data available that reflected the true nature of the data. It can be observed that when sample size started to decrease, the polynomial function varied from the original data because the function hasn't learnt enough.

# In[ ]:




