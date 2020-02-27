import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.optimize import curve_fit, root, fsolve
from scipy.signal import argrelextrema
from numpy import dot, outer
from os import listdir
from os.path import isfile, join

def main():
    
    # Set parameters
    n_minimax = 14                     # Number of minimax points
    R_minimax = int(146)              # Range of the minimax approximation
    n_x       = 12000                   # total number of points on the x-axis for optimization
    eps_diff  = 10**(-10)

    path_time = "~/11_minimax_python_time/1_other_ranges/alpha_beta_of_N_14"
    path_freq = "~/12_minimax_python_frequency/1_other_ranges/alpha_beta_of_N_14"

    desired_ranges_file = "./desired_ranges"

    with open(desired_ranges_file) as f:
      lines = [int(x) for x in f]

    ydata = np.zeros(n_x,dtype=np.float128)

    gammas = np.zeros( (n_minimax, n_minimax), dtype=np.float128 )

    count_desired_ranges = 0

    while True:

       R_minimax = lines[count_desired_ranges]
       count_desired_ranges += 1
       alphas_time = find_closest_R(n_minimax, R_minimax, path_time)
       alphas_freq = find_closest_R(n_minimax, R_minimax, path_freq)

       xdata = 10**(np.logspace(0,np.log10(np.log10(R_minimax)+1),n_x,dtype=np.float128))/10

       for omega in alphas_freq: 

          gammas = least_squares(xdata, alphas_time, omega)

          fig1, (axis1) = pl.subplots(1,1)
          axis1.set_xlim((0.8,R_minimax))
          axis1.semilogx( xdata, eta(xdata, gammas, alphas_time, omega) )
#          axis1.semilogx([0.8,R_minimax], [alphas[-1],alphas[-1]])
#          axis1.semilogx([0.8,R_minimax], [-alphas[-1],-alphas[-1]])
          pl.show()


          while (E/E_old < 1-eps_diff or E > E_old):
   
             E_old = E
  
             extrema_x = np.append(xdata[0], xdata[argrelextrema(eta(xdata,alphas[0:np.size(alphas)-1]), np.greater)[0]])
             if np.size(extrema_x) == n_minimax: 
                extrema_x = np.append(extrema_x, xdata[-1])
             extrema_x = np.append(extrema_x, xdata[argrelextrema(eta(xdata,alphas[0:np.size(alphas)-1]), np.less)[0]])
             num_extrema = np.size(extrema_x)
  
             E = np.average(np.abs(eta(extrema_x,alphas[0:np.size(alphas)-1])))
             i += 1
             print("iteration =", i, "E =",  E, "Range =", R_minimax)
     
             gammas = my_fsolve(extrema_x, gammas, alphas_time, alphas_freq)
  
       sort_indices = np.argsort(alphas[0:n_minimax])
       num_zeros = 13-len(str(R_minimax))
   
       np.savetxt("alpha_beta_of_N_"+str(n_minimax)+"_R_"+"0"*num_zeros+str(R_minimax)+"_E_"+\
               np.array2string(np.amax(np.abs(eta(extrema_x,alphas))), formatter={'float_kind':lambda x: "%.3E" % x}), \
               np.append(alphas[sort_indices],alphas[sort_indices+n_minimax]) )
   
#       fig1, (axis1) = pl.subplots(1,1)
#       axis1.set_xlim((0.8,R_minimax))
#       axis1.semilogx(xdata,eta(xdata,alphas))
#       axis1.semilogx([0.8,R_minimax], [alphas[-1],alphas[-1]])
#       axis1.semilogx([0.8,R_minimax], [-alphas[-1],-alphas[-1]])
#       pl.show()

       if(break_i):
           R_minimax += R_add
       else:
           R_minimax = int(R_minimax/1.5)


def least_squares(xdata, alphas_time, omega):

    n_minimax = np.size(alphas_time)

    n_x_points = np.size(xdata)

    mat_J = np.zeros((n_x_points, n_minimax),dtype=np.float128)

    for index_i in range(n_minimax):
        mat_J[:,index_i] = np.cos(omega*alpha_time[index_i])*np.exp(-xdata*alpha_time[index_i])

    vec_v = 2*xdata/(xdata**2+omega**2)

    vec_JTv = np.dot( np.transpose(mat_J), vec_v )

    mat_JTJ = np.dot( np.transpose(mat_J), mat_J )

    mat_for_Gauss = np.zeros((n_x_points, n_minimax+1),dtype=np.float128)
    mat_for_Gauss[:, 0:n_minimax+1] = mat_JTJ
    mat_for_Gauss[:, n_minimax+1] = vec_JTv

    gamma = gauss(mat_for_Gauss)

    return gamma


def find_closest_R(n_minimax, R_minimax, path):

    files = [f for f in listdir(path) if isfile(join(path, f))]

    for f in files:
        if(not f.startswith("alpha")): continue
        R_file = int(f[21:34])
        if( R_file == R_minimax ): 
            with open(path+"/"+f) as filetoread:
                alphas_betas_E = [np.float64(x) for x in filetoread]

    alphas = np.float128(alphas_betas_E[0:n_minimax])

    print("\nR_desired =", R_minimax, "R_file =", R_file, "alphas =", alphas)

    return alphas

def eta(x, gammas, alphas_time, omega):

#    params_1d = np.transpose(params)[:,0]

    n_x_points = np.size(x)

    eta = 2*x/(x**2+omega**2) 

    for index_i in range(n_minimax):
      eta -= gammas[index_i]*np.cos(omega*alphas_time[index_i])*np.exp(-x*alphas_time[index_i])

    return eta

def my_fsolve(extrema_x, alphas):
    size_problem = np.size(alphas)
    n_minimax = (size_problem-1)//2

    E = np.empty(size_problem, dtype=np.float128)
    E[0:size_problem//2+1] = alphas[-1]
    E[size_problem//2+1:] = -alphas[-1]

    vec_f = eta(extrema_x, alphas) - E

    mat_J = np.zeros((size_problem, size_problem+1),dtype=np.float128)

    for index_i in range(n_minimax):
        mat_J[:,index_i] = -extrema_x[0:size_problem]*alphas[index_i+n_minimax]*np.exp(-extrema_x[0:size_problem]*alphas[index_i])
        mat_J[:,index_i+n_minimax] = np.exp(-extrema_x[0:size_problem]*alphas[index_i])

    mat_J[:,-2] = np.sign(E[0:size_problem])
    mat_J[:,-1] = vec_f[0:size_problem]

    delta = gauss(mat_J)

    return alphas + delta

def gauss(A):
    n = len(A)

    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n+1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -A[k][i]/A[i][i]
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/A[i][i]
        for k in range(i-1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x

if __name__ == "__main__":
    main()

