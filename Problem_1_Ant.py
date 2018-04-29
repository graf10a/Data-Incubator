import numpy as np


def get_path(m, n):    
    
    """Computes a random path between the southwest-most 
    point (0,0) and northeast-most point (m,n). Returns 
    the x and y coordinates for the points of the path."""
    
    x = np.array([0], dtype='uint8')
    y = np.array([0], dtype='uint8')
    
    while ((x[-1] < m)|(y[-1] < n)): 
        
        x_step = np.random.randint(0, 2)
        y_step = 1 - x_step
        
        x_next = x[-1] + x_step
        
        if (x_next <= m):
            x = np.concatenate((x, np.array([x_next])))
        else: 
            x = np.concatenate((x, x[[-1]]))
             
        y_next = y[-1] + y_step
        
        if (y_next <= n):
            y = np.concatenate((y, np.array([y_next])))
        else: 
            y = np.concatenate((y, y[[-1]]))
        
    return x, y


def dev(x, y, m, n):
    
    """Computes the deviation for a given path.
    The path must be specified by the arrays of 
    the x and y coordinates of its points."""
    
    d = x/m - y/n
    return np.absolute(d).max()


def get_devs(m, n, n_iter):
    
    """Computed deviations for n_iter random paths."""
    
    d = np.array([])
    
    for i in range(0, n_iter):
        x, y = get_path(m, n)
        d = np.concatenate((d, np.array([dev(x, y, m, n)])))
    return d

def mean_std_prob(m, n, n_iter):
    
    """Computes the mean, standard deviation, and
    conditional probalities for n_iter random paths."""
    
    print("\nParameters: m = {}, n = {}, n_iter = {}.".format(m, n, n_iter))
    
    devs = get_devs(m, n, n_iter)
    
    m = devs.mean()
    print("The mean is {}.".format(m))
    
    sd = devs.std()
    print("The standard deviation is {}.".format(sd))
    
    sub_1 = devs[devs > 0.2]
    sub_2 = sub_1[sub_1 > 0.6]
    prob = len(sub_2)/len(sub_1)
    print("The conditional probability is {}.".format(prob))
    
mean_std_prob(11, 7, 100000)
mean_std_prob(23, 31, 100000)