import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

np.random.seed(0)
path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])))
print(path)

a = 0.3
b = 0.6
c = 0.3
T = 10
mu, sigma = 0, 1
X10 = 0
X20 = 0.01
dt_exact = 0.0001
N_exact = int(T/dt_exact)
print(N_exact)
dt_exact_list = np.arange(0, T, dt_exact)
print(len(dt_exact_list))

def Num_Scheme(X10, X20, N, dW_exact, case):
    dt = T/N
    #print(N)
    X1_list = np.zeros(N+1)
    X2_list = np.zeros(N+1)
    X1_old = X10
    X2_old = X20
    if N!=N_exact:
        incr = int(dt/dt_exact)
        dW = np.array([np.sum(dW_exact[i*incr:(i+1)*incr]) for i in range(N)])
    else:
        dW = dW_exact
    X1_list[0] = X10
    X2_list[0] = X20
    
    if case=="Euler-Ito":
        for i in range(N):
            X1 = X1_old + X2_old*dt
            X2 = X2_old + (-b*X2_old-np.sin(X1_old)+c*np.sin(2*X1_old))*dt + (-b*X2_old-a*np.sin(X1_old))*dW[i]
            X1_old = X1
            X2_old = X2
            X1_list[i+1] = X1
            X2_list[i+1] = X2

    if case=="Euler-Stratonovich":
        for i in range(N):
            X1 = X1_old + X2_old*dt
            X2 = X2_old + (-b*X2_old-np.sin(X1_old)+c*np.sin(2*X1_old) + 0.5*b**2*X2_old+0.5*a*b*np.sin(X1_old))*dt + (-b*X2_old-a*np.sin(X1_old))*dW[i]
            X1_old = X1
            X2_old = X2
            X1_list[i+1] = X1
            X2_list[i+1] = X2

    if case=="Milstein":
        for i in range(N):
            X1 = X1_old + X2_old*dt
            X2 = X2_old + (-b*X2_old-np.sin(X1_old)+c*np.sin(2*X1_old) + 0.5*b**2*X2_old+0.5*a*b*np.sin(X1_old))*dt + (-b*X2_old-a*np.sin(X1_old))*dW[i] + 0.5*(b**2*X2_old+a*b*np.sin(X1_old))*(dW[i]**2-dt)
            X1_old = X1
            X2_old = X2
            X1_list[i+1] = X1
            X2_list[i+1] = X2

    return X1_list, X2_list

dt_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
dW_realizations = 1000

err_strongX1_array_Ito = np.zeros((dW_realizations, len(dt_list)))
err_strongX2_array_Ito = np.zeros((dW_realizations, len(dt_list)))
err_weakX1_array_Ito = np.zeros((dW_realizations, len(dt_list)))
err_weakX2_array_Ito = np.zeros((dW_realizations, len(dt_list)))

err_strongX1_array_Stratonovitch = np.zeros((dW_realizations, len(dt_list)))
err_strongX2_array_Stratonovitch = np.zeros((dW_realizations, len(dt_list)))
err_weakX1_array_Stratonovitch = np.zeros((dW_realizations, len(dt_list)))
err_weakX2_array_Stratonovitch = np.zeros((dW_realizations, len(dt_list)))

err_strongX1_array_Milstein = np.zeros((dW_realizations, len(dt_list)))
err_strongX2_array_Milstein = np.zeros((dW_realizations, len(dt_list)))
err_weakX1_array_Milstein = np.zeros((dW_realizations, len(dt_list)))
err_weakX2_array_Milstein = np.zeros((dW_realizations, len(dt_list)))

start_time = time.time()
for i in range(dW_realizations):
    dW_exact_realization = np.random.normal(mu, np.sqrt(dt_exact), int(T/dt_exact))
    for j in range(len(dt_list)):
        dt = dt_list[j]
        print(f"dt: {dt}, realization: {i}")
        N = int(T/dt)
        X_exact = Num_Scheme(X10, X20, N_exact, dW_exact_realization, "Euler-Ito")
        X_approx = Num_Scheme(X10, X20, N, dW_exact_realization, "Euler-Ito")

        err_strongX1_array_Ito[i][j] = np.abs(X_exact[0][-1]-X_approx[0][-1])
        err_strongX2_array_Ito[i][j] = np.abs(X_exact[1][-1]-X_approx[1][-1])
        err_weakX1_array_Ito[i][j] = (X_exact[0][-1]-X_approx[0][-1])
        err_weakX2_array_Ito[i][j] = (X_exact[1][-1]-X_approx[1][-1])

        X_exact = Num_Scheme(X10, X20, N_exact, dW_exact_realization, "Euler-Stratonovich")
        X_approx = Num_Scheme(X10, X20, N, dW_exact_realization, "Euler-Stratonovich")

        err_strongX1_array_Stratonovitch[i][j] = np.abs(X_exact[0][-1]-X_approx[0][-1])
        err_strongX2_array_Stratonovitch[i][j] = np.abs(X_exact[1][-1]-X_approx[1][-1])
        err_weakX1_array_Stratonovitch[i][j] = (X_exact[0][-1]-X_approx[0][-1])
        err_weakX2_array_Stratonovitch[i][j] = (X_exact[1][-1]-X_approx[1][-1])

        X_exact = Num_Scheme(X10, X20, N_exact, dW_exact_realization, "Milstein")
        X_approx = Num_Scheme(X10, X20, N, dW_exact_realization, "Milstein")

        err_strongX1_array_Milstein[i][j] = np.abs(X_exact[0][-1]-X_approx[0][-1])
        err_strongX2_array_Milstein[i][j] = np.abs(X_exact[1][-1]-X_approx[1][-1])
        err_weakX1_array_Milstein[i][j] = (X_exact[0][-1]-X_approx[0][-1])
        err_weakX2_array_Milstein[i][j] = (X_exact[1][-1]-X_approx[1][-1])
    
err_strongX1_list_Ito = np.mean(err_strongX1_array_Ito, axis=0)
err_strongX2_list_Ito = np.mean(err_strongX2_array_Ito, axis=0)
err_weakX1_list_Ito = np.abs(np.mean(err_weakX1_array_Ito, axis=0))
err_weakX2_list_Ito = np.abs(np.mean(err_weakX2_array_Ito, axis=0))

err_strongX1_list_Stratonovitch = np.mean(err_strongX1_array_Stratonovitch, axis=0)
err_strongX2_list_Stratonovitch = np.mean(err_strongX2_array_Stratonovitch, axis=0)
err_weakX1_list_Stratonovitch = np.abs(np.mean(err_weakX1_array_Stratonovitch, axis=0))
err_weakX2_list_Stratonovitch = np.abs(np.mean(err_weakX2_array_Stratonovitch, axis=0))

err_strongX1_list_Milstein = np.mean(err_strongX1_array_Milstein, axis=0)
err_strongX2_list_Milstein = np.mean(err_strongX2_array_Milstein, axis=0)
err_weakX1_list_Milstein = np.abs(np.mean(err_weakX1_array_Milstein, axis=0))
err_weakX2_list_Milstein = np.abs(np.mean(err_weakX2_array_Milstein, axis=0))

end_time = time.time()
print(f"Time taken for {dW_realizations} realizations and exact dt={dt_exact}: {end_time-start_time} seconds")
    
plt.loglog(dt_list, err_strongX1_list_Ito)
plt.loglog(dt_list, err_strongX2_list_Ito)
plt.xlabel("dt")
plt.ylabel("Strong Error")
plt.title("Strong Error for X1 and X2 - Euler Scheme, Ito Interpretation")
plt.legend(["Strong Error X1", " Strong Error X2"])
plt.grid()
plt.savefig(path + "/Strong_Error_X1_X2_Euler_Ito.png")
plt.clf()

plt.loglog(dt_list, err_weakX1_list_Ito)
plt.loglog(dt_list, err_weakX2_list_Ito)
plt.xlabel("dt")
plt.ylabel("Weak Error")
plt.title("Weak Error for X1 and X2 - Euler Scheme, Ito Interpretation")
plt.legend(["Weak Error X1", " Weak Error X2"])
plt.grid()
plt.savefig(path + "/Weak_Error_X1_X2_Euler_Ito.png")
plt.clf()


plt.loglog(dt_list, err_strongX1_list_Stratonovitch)
plt.loglog(dt_list, err_strongX2_list_Stratonovitch)
plt.xlabel("dt")
plt.ylabel("Strong Error")
plt.title("Strong Error for X1 and X2 - Euler Scheme, Stratonovitch Interpretation")
plt.legend(["Strong Error X1", " Strong Error X2"])
plt.grid()
plt.savefig(path + "/Strong_Error_X1_X2_Euler_Stratonovitch.png")
plt.clf()

plt.loglog(dt_list, err_weakX1_list_Stratonovitch)
plt.loglog(dt_list, err_weakX2_list_Stratonovitch)
plt.xlabel("dt")
plt.ylabel("Weak Error")
plt.title("Weak Error for X1 and X2 - Euler Scheme, Stratonovitch Interpretation")
plt.legend(["Weak Error X1", " Weak Error X2"])
plt.grid()
plt.savefig(path + "/Weak_Error_X1_X2_Euler_Stratonovitch.png")
plt.clf()
    

plt.loglog(dt_list, err_strongX1_list_Milstein)
plt.loglog(dt_list, err_strongX2_list_Milstein)
plt.xlabel("dt")
plt.ylabel("Strong Error")
plt.title("Strong Error for X1 and X2 - Milstein Scheme, Stratonovitch Interpretation")
plt.legend(["Strong Error X1", " Strong Error X2"])
plt.grid()
plt.savefig(path + "/Strong_Error_X1_X2_Milstein_Stratonovitch.png")
plt.clf()

plt.loglog(dt_list, err_weakX1_list_Milstein)
plt.loglog(dt_list, err_weakX2_list_Milstein)
plt.xlabel("dt")
plt.ylabel("Weak Error")
plt.title("Weak Error for X1 and X2 - Milstein Scheme, Stratonovitch Interpretation")
plt.legend(["Weak Error X1", " Weak Error X2"])
plt.grid()
plt.savefig(path + "/Weak_Error_X1_X2_Milstein_Stratonovitch.png")
plt.clf()