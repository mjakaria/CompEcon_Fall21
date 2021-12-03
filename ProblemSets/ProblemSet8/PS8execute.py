

import numpy as np
import matplotlib.pyplot as plt
import PS8functions


# create grid of h
e_lb = 0
e_ub = 22
size_e = 300
e_grid = np.linspace(e_lb, e_ub, size_e)

# define parameters
beta = 0.9
sigma=1
m=1000
n=2000
k=1500
params = [beta,k,m,n,sigma]
u = np.zeros((size_e,size_e))
c = np.zeros((size_e,size_e))

# call value function from functions.py
x = PS8functions.VFI(params, e_grid, u, c)

# create visualization of value function
plt.figure()
plt.plot(e_grid, x[0])
plt.xlabel('Size of Education')
plt.ylabel('Value Function')



# Policy function from function.py

#Plot cake to leave rule as a function of cake size
#plt.figure()
#fig, ax = plt.subplots()
#ax.plot(e_grid[1:], opte[1:], label='education')
#ax.plot(e_grid[1:], e_grid[1:], '--', label='45 degree line')
# Now add the legend with some customizations.
#legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
#for label in legend.get_texts():
    #label.set_fontsize('large')
#for label in legend.get_lines():
    #label.set_linewidth(1.5)  # the legend line width
#plt.xlabel('Size of education')
#plt.ylabel('Optimal education choosing')
#plt.title('Policy Function')
#plt.show()