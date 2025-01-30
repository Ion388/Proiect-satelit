import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

np.random.seed(0)
path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])))
print(path)
err_color = 'blue'
line_color = 'red'
sol_plots = True
err_study = True
ar = 'auto'
height = 16
width = 9
dpi = 100
title_size = 20
label_size = 18
tick_size = 16
legend_size = 16

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
dt_exact_list = np.arange(0, T+dt_exact, dt_exact)
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

if sol_plots==True:
    dW_exact = np.random.normal(mu, np.sqrt(dt_exact), int(T/dt_exact))
    for dt in [0.0005, 0.005, 0.05, 0.5]:
        N = int(T/dt)
        X_exact = Num_Scheme(X10, X20, N_exact, dW_exact, "Euler-Ito")
        X_approx = Num_Scheme(X10, X20, N, dW_exact, "Euler-Ito")
        print(len(np.arange(0, T, dt)))
        dt_list = np.arange(0, T+dt*0.5, dt)
    

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        ax.plot(dt_exact_list, X_exact[0], color=err_color, label="Exact Solution X1")
        ax.plot(dt_list, X_approx[0], color=line_color, label="Approximate Solution X1")
        ax.set_xlabel("Time", fontsize=label_size)
        ax.set_ylabel("X1", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.set_title(f"Exact Solution vs Approximate Solution for X1, dt={dt} - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
        ax.legend(fontsize=legend_size)
        ax.grid()
        ax.set_aspect(ar)
        plt.savefig(path + f"/Exact_vs_Approx_X1_{dt}.png")
        plt.clf()


        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        ax.plot(dt_exact_list, X_exact[1], color=err_color, label="Exact Solution X2")
        ax.plot(dt_list, X_approx[1], color=line_color, label="Approximate Solution X2")
        ax.set_xlabel("Time", fontsize=label_size)
        ax.set_ylabel("X2", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)    
        ax.set_title(f"Exact Solution vs Approximate Solution for X2, dt={dt} - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
        ax.legend(fontsize=legend_size)
        ax.grid()
        ax.set_aspect(ar)
        plt.savefig(path + f"/Exact_vs_Approx_X2_{dt}.png")
        plt.clf()

if err_study==True:
    dt_list = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
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
            # print(f"dt: {dt}, realization: {i}")
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
        

    #### strong error plots Euler, Ito ####
    slope_X1 = np.polyfit(np.log10(dt_list), np.log10(err_strongX1_list_Ito), 1)
    slope_X2 = np.polyfit(np.log10(dt_list), np.log10(err_strongX2_list_Ito), 1)
    print(f"Slope of Strong Error X1 (Euler-Ito): {slope_X1}")
    print(f"Slope of Strong Error X2 (Euler-Ito): {slope_X2}")

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_strongX1_list_Ito, color=err_color, label="Strong Error X1")
    ax.loglog(dt_list, dt_list**slope_X1[0]*10**slope_X1[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X1[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Strong Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Strong Error for X1 - Euler Scheme, Ito Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Strong_Error_X1_Euler_Ito.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_strongX2_list_Ito, color=err_color, label="Strong Error X2")
    ax.loglog(dt_list, dt_list**slope_X2[0]*10**slope_X2[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X2[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Strong Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Strong Error for X2 - Euler Scheme, Ito Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Strong_Error_X2_Euler_Ito.png")
    plt.clf()

    #### weak error plots Euler, Ito ####
    slope_X1 = np.polyfit(np.log10(dt_list), np.log10(err_weakX1_list_Ito), 1)
    slope_X2 = np.polyfit(np.log10(dt_list), np.log10(err_weakX2_list_Ito), 1)
    print(f"Slope of Weak Error X1 (Euler-Ito): {slope_X1}")
    print(f"Slope of Weak Error X2 (Euler-Ito): {slope_X2}")

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_weakX1_list_Ito, color=err_color, label="Weak Error X1")
    ax.loglog(dt_list, dt_list**slope_X1[0]*10**slope_X1[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X1[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Weak Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Weak Error for X1 - Euler Scheme, Ito Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Weak_Error_X1_Euler_Ito.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_weakX2_list_Ito, color=err_color, label="Weak Error X2")
    ax.loglog(dt_list, dt_list**slope_X2[0]*10**slope_X2[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X2[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Weak Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Weak Error for X2 - Euler Scheme, Ito Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)  
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Weak_Error_X2_Euler_Ito.png")
    plt.clf()

    #### strong error plots Euler, Stratonovitch ####
    slope_X1 = np.polyfit(np.log10(dt_list), np.log10(err_strongX1_list_Stratonovitch), 1)
    slope_X2 = np.polyfit(np.log10(dt_list), np.log10(err_strongX2_list_Stratonovitch), 1)
    print(f"Slope of Strong Error X1 (Euler-Stratonovitch): {slope_X1}")
    print(f"Slope of Strong Error X2 (Euler-Stratonovitch): {slope_X2}")

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_strongX1_list_Stratonovitch, color=err_color, label="Strong Error X1")
    ax.loglog(dt_list, dt_list**slope_X1[0]*10**slope_X1[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X1[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Strong Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Strong Error for X1 - Euler Scheme, Stratonovitch Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Strong_Error_X1_Euler_Stratonovitch.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_strongX2_list_Stratonovitch, color=err_color, label="Strong Error X2")
    ax.loglog(dt_list, dt_list**slope_X2[0]*10**slope_X2[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X2[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Strong Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Strong Error for X2 - Euler Scheme, Stratonovitch Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Strong_Error_X2_Euler_Stratonovitch.png")
    plt.clf()

    #### weak error plots Euler, Stratonovitch ####
    slope_X1 = np.polyfit(np.log10(dt_list), np.log10(err_weakX1_list_Stratonovitch), 1)
    slope_X2 = np.polyfit(np.log10(dt_list), np.log10(err_weakX2_list_Stratonovitch), 1)
    print(f"Slope of Weak Error X1 (Euler-Stratonovitch): {slope_X1}")
    print(f"Slope of Weak Error X2 (Euler-Stratonovitch): {slope_X2}")

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_weakX1_list_Stratonovitch, color=err_color, label="Weak Error X1")
    ax.loglog(dt_list, dt_list**slope_X1[0]*10**slope_X1[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X1[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Weak Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Weak Error for X1 - Euler Scheme, Stratonovitch Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Weak_Error_X1_Euler_Stratonovitch.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_weakX2_list_Stratonovitch, color=err_color, label="Weak Error X2")
    ax.loglog(dt_list, dt_list**slope_X2[0]*10**slope_X2[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X2[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Weak Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Weak Error for X2 - Euler Scheme, Stratonovitch Interpretation - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Weak_Error_X2_Euler_Stratonovitch.png")
    plt.clf()

    #### strong error plots Milstein ####
    slope_X1 = np.polyfit(np.log10(dt_list), np.log10(err_strongX1_list_Milstein), 1)
    slope_X2 = np.polyfit(np.log10(dt_list), np.log10(err_strongX2_list_Milstein), 1)
    print(f"Slope of Strong Error X1 (Milstein): {slope_X1}")
    print(f"Slope of Strong Error X2 (Milstein): {slope_X2}")

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_strongX1_list_Milstein, color=err_color, label="Strong Error X1")
    ax.loglog(dt_list, dt_list**slope_X1[0]*10**slope_X1[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X1[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Strong Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Strong Error for X1 - Milstein Scheme - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Strong_Error_X1_Milstein.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_strongX2_list_Milstein, color=err_color, label="Strong Error X2")
    ax.loglog(dt_list, dt_list**slope_X2[0]*10**slope_X2[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X2[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Strong Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Strong Error for X2 - Milstein Scheme - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Strong_Error_X2_Milstein.png")
    plt.clf()

    #### weak error plots Milstein ####
    slope_X1 = np.polyfit(np.log10(dt_list), np.log10(err_weakX1_list_Milstein), 1)
    slope_X2 = np.polyfit(np.log10(dt_list), np.log10(err_weakX2_list_Milstein), 1)
    print(f"Slope of Weak Error X1 (Milstein): {slope_X1}")
    print(f"Slope of Weak Error X2 (Milstein): {slope_X2}")

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_weakX1_list_Milstein, color=err_color, label="Weak Error X1")
    ax.loglog(dt_list, dt_list**slope_X1[0]*10**slope_X1[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X1[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Weak Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Weak Error for X1 - Milstein Scheme - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Weak_Error_X1_Milstein.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.loglog(dt_list, err_weakX2_list_Milstein, color=err_color, label="Weak Error X2")
    ax.loglog(dt_list, dt_list**slope_X2[0]*10**slope_X2[1], linestyle='dashed', color=line_color, label=f"Slope: {slope_X2[0]:.2f}")
    ax.set_xlabel("dt", fontsize=label_size)
    ax.set_ylabel("Weak Error", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(f"Weak Error for X2 - Milstein Scheme - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid()
    ax.set_aspect(ar)
    plt.savefig(path + "/Weak_Error_X2_Milstein.png")
    plt.clf()