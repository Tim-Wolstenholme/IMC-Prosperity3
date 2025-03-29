import numpy as np
from math import sqrt

def find_stationary(x_proc: np.array):
    # Performs a DF test at 1,5,10,25 significance
    z_proc = x_proc[:-1]
    #Find Least squares estimate
    phi_hat = np.matmul(np.matmul(np.invert(np.matmul(np.transpose(z_proc), z_proc)),
                                  np.transpose(z_proc)), x_proc[1:])
    #Find standard error
    variance_sum = 0
    for i in range(1, len(x_proc)):
        variance_sum += (x_proc[i] - phi_hat * x_proc[i - 1]) ** 2
    residual_variance = variance_sum/(len(x_proc)-2)
    error_sum = 0
    for i in range(1,len(x_proc)-1):
        error_sum += x_proc[i-1]**2
    std_error = sqrt(residual_variance/error_sum)
    #test statistic
    tau = (phi_hat-1)/std_error
    return {1:tau < -2.58,
            5:tau < -1.95,
            10:tau < -1.62,
            25: tau < -1.28}


def diffing(time_series):
    for i in range(1,len(time_series)):
        time_series[i] -= time_series[i-1]
    return time_series

