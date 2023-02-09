# In [1]:

import qutip, sys, pickle
import numpy as np 
import matplotlib.pyplot as plt 

# In [2}:

def visz_H_tensor_evs(Htensor):
    if (type(Htensor) == qutip.Qobj):
        Htensor_local = np.array(Htensor)
    x = sorted(np.array(qutip.Qobj(Htensor_local).eigenenergies().real))
    y = sorted(np.array(qutip.Qobj(Htensor_local).eigenenergies().imag))
    z = np.arange(len(x))
    fig1, ax1 = plt.subplots()
    ax1.plot(z,x, label = "Real Part evs")
    ax1.plot(z,y, label = "Imag part evs")
    ax1.legend(loc=0)
    ax1.set_title("H-tensor's eigenvalues' real and imag part")

def plot_exact_v_proj_ev_avgs(obs, labels, timespan, Result_proj_ev, 
                                                     Result_exact,
                                                     visualize_diff_expt_vals = False):
    Tot = len(obs); Cols = 3
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)
    z = timespan[:-1]
    fig = plt.figure(figsize=(18, 14))
    for k in range(Tot):
        ax = fig.add_subplot(Rows,Cols,Position[k])
        if visualize_diff_expt_vals:
            ax.plot(z, Result_exact.expect[k][:-1] - Result_proj_ev[k], label = "Exact diff Proj.ev")
        else:
            ax.plot(z, Result_exact.expect[k][:-1], label = "Exact")
            ax.plot(z, Result_proj_ev[k], label = "Manifold proj")        
        ax.legend(loc=0)
        ax.set_title("Expected values: Proj-ev. v. Exact for " + labels[k])
    plt.show()

def plot_exact_v_proj_ev_metrics(ts, metrics, label_metric):
    Tot = len(label_metric); Cols = 3
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)
    z = ts[:-1]
    fig = plt.figure(figsize=(10, 5))
    for k in range(Tot):
        ax = fig.add_subplot(Rows,Cols,Position[k])
        ax.plot(z, metrics[k], label = "Exact v. Proj ev: " + label_metric[k])
        ax.legend(loc=0)
        ax.set_title("Matrix metrics")
    plt.show()    
    
# In [3]:

def plot_exact_v_proj_ev_avgs_multiple(labels, timespan, no_cols_desired,
                                                    multiple_evolutions,
                                                    range_of_temps_or_dims,
                                                    plot_var_lengths = False,
                                                    plot_var_HierarchBases_dim = False,
                                                    plot_var_temps = False):
    
    if (plot_var_lengths== False) and (plot_var_HierarchBases_dim == False) and (plot_var_temps == False):
            print("No visualization choice taken")
            
    title = "avgs_ev_proj_ev"
    if plot_var_lengths:
        title += "_var_lengths"
    if plot_var_HierarchBases_dim:
        title += "_var_HierarchBases"
    if plot_var_temps: 
        title += "_var_temps"
    
    Tot = len(labels); Cols = no_cols_desired
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)
    z = timespan[:-1]
    fig = plt.figure(figsize=(25, 16))
    range_temps_labels = [i for i in range(len(range_of_temps_or_dims))]
    for k in range(Tot):
        ax = fig.add_subplot(Rows, Cols, Position[k])
        
        if (plot_var_lengths== False) and (plot_var_HierarchBases_dim == False) and (plot_var_temps == False):
            print("No visualization choice taken")
        
        if plot_var_lengths and all_exact_dynamics_processed: 
            range_dims = range_of_temps_or_dims
            for dim in range_dims: 
                ax.plot(z, multiple_evolutions["dict_res_proj_ev_all"]["dict_res_proj_ev_N" + str(range_dims.index(dim))]["Avgs"][k],
                        label = "Proj_ev. N=" + str(dim))
                ax.plot(z, multiple_evolutions["res_exact_all"]["res_exact_N" + str(range_dims.index(dim))].expect[k][:-1],
                        label = "Ex_ev. N=" + str(dim))
                
        if plot_var_HierarchBases_dim:
            range_HB_dims = range_of_temps_or_dims
            for dim in range_HB_dims: 
                ax.plot(z, multiple_evolutions["dict_res_proj_ev_all"]["dict_res_proj_ev_HierarchBases" + str(range_HB_dims.index(dim))]["Avgs"][k], label = "Proj_ev. ℓ=" + str(dim))
                var_loc = multiple_evolutions["res_exact_all"]["res_exact_HierarchBases" + str(range_HB_dims.index(dim))]
                if var_loc is None:
                    pass
                else: 
                    ax.plot(z, var_loc.expect[k][:-1], label = "Exact Evolution")
                var_loc = None
            
        if plot_var_temps and all_exact_dynamics_processed: 
            range_temps = range_of_temps_or_dims
            for T in range_temps:
                ax.plot(z, multiple_evolutions["dict_res_proj_ev_all"]["dict_res_proj_ev_T" + str(range_temps.index(T))]["Avgs"][k],
                        label = "Proj_ev. T=" + str(T))
                ax.plot(z, multiple_evolutions["res_exact_all"]["res_exact_T" + str(range_temps.index(T))].expect[k][:-1],
                        label = "Ex_ev. T=" + str(T))
        
        ax.legend(loc=0)
        ax.set_title("Expected values: Proj-ev. v. Exact for " + labels[k])   
    
    save_results_to = 'C:/Users/tomas/PhD Physics/1st Year/Spin Chains/results_figs/'
    plt.savefig(save_results_to + title + f"_processed={len(range_of_temps_or_dims)}_items.svg")
    plt.show()
    
def plot_exact_v_proj_ev_metrics_multiple(timespan, range_of_temps_or_dims, metric_local,
                                          plot_var_lengths = False,
                                          plot_var_HierarchBases_dim = False,
                                          plot_var_temps = False):
    
    if (plot_var_lengths== False) and (plot_var_HierarchBases_dim == False) and (plot_var_temps == False):
            print("No visualization choice taken")
            
    title = "metrics_proj_ev"
    if plot_var_lengths:
        title += "_var_lengths"
    if plot_var_HierarchBases_dim:
        title += "_var_HierarchBases"
    if plot_var_temps: 
        title += "_var_temps"
    
    label_metric = ["Bures Exact v. Proj ev", "S(exact || proj_ev)", "S(proj_ev || exact)"]
    Tot = len(label_metric); Cols = 3
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)
    z = timespan[:-1]
    fig = plt.figure(figsize=(20, 10))
    
    for k in range(Tot):
        ax = fig.add_subplot(Rows, Cols, Position[k])
       
        if plot_var_lengths:
            range_dims = range_of_temps_or_dims      
            for dim in range_dims:
                ax.plot(z, metric_local[k]["N"+str(range_dims.index(dim))], label = label_metric[k] + " N=" + str(dim))
                ax.legend(loc=0)
            ax.set_title("Matrix metrics")
            
        if plot_var_HierarchBases_dim:
            range_HB_dims = range_of_temps_or_dims      
            for dim in range_HB_dims:
                ax.plot(z, metric_local[k]["HierarchBases"+str(range_HB_dims.index(dim))], label = label_metric[k] + " ℓ=" + str(dim))
                ax.legend(loc=0)
            ax.set_title("Matrix metrics")  
            
        if plot_var_temps:
            range_temps = range_of_temps_or_dims      
            for T in range_temps:
                ax.plot(z, metric_local[k]["T"+str(range_temps.index(T))], label = label_metric[k] + " T=" + str(T))
                ax.legend(loc=0)
            ax.set_title("Matrix metrics")
    
    save_results_to = 'C:/Users/tomas/PhD Physics/1st Year/Spin Chains/results_figs/'
    plt.savefig(save_results_to + title + f"_processed={len(range_of_temps_or_dims)}_items.svg")
    plt.show()    
