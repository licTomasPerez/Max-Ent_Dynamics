# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import matplotlib.pyplot as plt
import time as time
import scipy.linalg as linalg

import matrix_analysis_lib as mat_ansys
import spin_representations as su2
import evs_visualization_tools as evs_plot
import projected_evolution_dynamics as ProjEv

# In [2]:

def HierarchBasis_vardim_proj_evs(Hamiltonian, fixed_ops_list, rho_ref, range_derived_series_orders, 
                                     temp_ref, temp_rho,
                                     generating_operator,
                                     init_coeff_list,
                                     timespan, label_ops,
                                     observables,
                                     rho_ref_equal_rho0 = False):
    
    """
    This module performs multiple projected and exact 
    evolutions for different reference hierarchical basis cardinality, 
    at fixed spin chain length and fixed reference temperature, 
    by adding one element at a time in said basis. This module takes as input:
    
        *♥*♥* 1. depth_and_seed_ops: 
                    a chosen set of operators/endomorphisms.
                    In our context, this set will be constructed
                    by taking iterated commutators with the
                    system's Hamiltonian, yielding a Lie algebra. 
                    
        *♥*♥* 2. observables: 
                    a chosen set of observables, for which their 
                    expected value's time evolution is desired.
                    
        *♥*♥* 3. label_ops: 
                    a list with strings values, containing the 
                    observables' physical name. These string 
                    literals are used for plotting the results.
                    
        *♥*♥* 4. ref_operator: 
                    a hermitian operator, from which a gaussian 
                    reference state may be constructed, using some 
                    user-input reference temperatures (see point 8). 
                    This state is constructed as:
                         
                         rho_ref = (-1/temp_ref * ref_operator).expm().
                    
        *♥*♥* 5. Hamiltonian:
                    the system's Hamiltonian. 
                    
        *♥*♥* 6. temp_rho: 
                    a fixed temperature for the system's initial 
                    state. 
        
        *♥*♥* 7. init_coeff_list:
                    Given an orthonormal basis of operators, this 
                    list of real numbers provides the initial 
                    state's decomposition in terms of said basis. 
                 
        *♥*♥* 8. timespan:
                    an array of times for which the time evolution
                    of the observables is to be calculated.
                     
        *♥*♥* 9. range_temps: 
                     a list of reference temperatures for which the 
                     different reference operators are constructed, as
                     thermal-like states, ie.
                     
                         rho_ref = (-1/temp_ref * ref_operator).expm()
                        
                    For each one of these refrence operators, both the
                    exact and projected evolutions are computed and returned. 
                    
        ====> Returns: multiple_evolutions: a dictionary storing all the 
                                            evolutions' data. 
                                            
             
        How the algorithm works: For a fixed-length spin chain and fixed
                                 set of operators, "depth_and_seed_ops" 
                                -constructed from iterated commutators-, 
        
            (i)*. a user-input set of reference temperatures, "range_temps",
                  is processed. For each one of these temperatures 
                  
                  .a: a thermal-like state, 'custom_rho_ref', is constructed 
                          using the user-input parameter "reference_operator".
                          A check is performed to confirm whether or not it is
                          a valid density operator. 
                          
                  .b: the projected and exact evolutions are computed, by calling
                      the d_depth_proj_ev routine, which takes as input (amongst others)
                      the previous reference state, the system's Hamiltonian, 
                      the list of observables. 
                  
                  .c: these results are then stored in four dictionaries 
                      (see the graph for a pictoral
                          representation of the data's structure). 
                      
            (ii)*. These steps are made for every desired reference temperature. 
                   When all of these are processed and their evolutions concluded,
                   a single dictionary, with the previous four nested dictionaries, 
                   is returned. 
       
       Warnings: (A). This module assumes all user input operator-valued parameters 
                      are hermitian. 
                 (B). This module assumes all operators have compatible dimensions 
                      as well. 
                 (C). All reference operators undergo a test for determining if these
                      are density operators of not. Should one of these tests fail,
                      the module will crash and all date will be lost.                
                 (D). Numerical stability has been checked upto temperatures of e-2
                      order. Lower temperatures may have numerical instabilities 
                      crashing the calculations.          
    """
    
    multiple_evolutions = {}
    multiple_init_configs = {}; multiple_evs_data = {}; multiple_dict_res_proj_ev = {}; multiple_res_exact = {}
    choose_process_exact_dynamics = [False for a in range(len(range_derived_series_orders) - 1)] + [True]
    
    for deg_solva in range_derived_series_orders:
        print("Processing step: ", range_derived_series_orders.index(deg_solva)+1, " and hierarchical basis of l= ", deg_solva)
        
        depth_and_seed_ops = [(1, op) for op in fixed_ops_list] + [(deg_solva+1, generating_operator)]
        
        #depth_and_seed_ops = [(1, id_op), 
        #                      (1, Hamiltonian), 
        #                      (deg_solva+1, generating_operator)]
        
        init_configs_MFT_state, evs_data, dict_res_proj_ev, res_exact = ProjEv.d_depth_proj_ev(
                    temp_ref = temp_ref,  temp_rho = temp_rho, 
                    timespan = timespan, 
                    Hamiltonian = Hamiltonian, lagrange_op = None,
                    depth_and_seed_ops = depth_and_seed_ops, observables = observables, label_ops = label_ops, 
                    coeff_list = init_coeff_list[range_derived_series_orders.index(deg_solva)], 
                    custom_ref_state = rho_ref, 
                    rho_ref_thermal_state = False, rho_ref_equal_rho0 = rho_ref_equal_rho0, visualize_H_evs = False, 
                    compute_exact_dynamics = choose_process_exact_dynamics[range_derived_series_orders.index(deg_solva)],
                    visualization_nonherm = False, visualize_expt_vals = False, visualize_diff_expt_vals = False
                    )
    
        multiple_init_configs["init_configs_HierarchBases" + str(range_derived_series_orders.index(deg_solva))] = init_configs_MFT_state
        multiple_evs_data["evs_data_HierarchBases" + str(range_derived_series_orders.index(deg_solva))] = evs_data
        multiple_dict_res_proj_ev["dict_res_proj_ev_HierarchBases" + str(range_derived_series_orders.index(deg_solva))] = dict_res_proj_ev
        multiple_res_exact["res_exact_HierarchBases" + str(range_derived_series_orders.index(deg_solva))] = res_exact
                    
    multiple_evolutions["init_configs_all"] = multiple_init_configs
    multiple_evolutions["evs_data_all"] = multiple_evs_data
    multiple_evolutions["dict_res_proj_ev_all"] = multiple_dict_res_proj_ev
    multiple_evolutions["res_exact_all"] = multiple_res_exact
    
    return multiple_evolutions
