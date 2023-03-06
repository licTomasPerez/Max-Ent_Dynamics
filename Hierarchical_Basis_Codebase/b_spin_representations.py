# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

import a_matrix_analysis_lib as mat_ansys

# In [2]:

def one_body_spin_ops(size):
    """
    Given an N-site spin chain, there are then 3N different, 
    non-trivial, operators acting on the full Hilbert space.
    N sigma-x operators, N sigma-y operators, 
    N sigma-z operators, and a global identity operator. 
    All these 3N+1-operators are constructed with a tensor
    product so that they all act on the full Hilbert space. 
    All but the global identity operator, act 
    non-trivially only on one Hilbert subspace. 
    """
    ### Basic, one-site spin operators are constructed.
    
    loc_sx_list = []; loc_sy_list = []; loc_sz_list = []; loc_globalid_list = []
    id2 = qutip.qeye(2)
    sx = .5*qutip.sigmax()
    sy = .5*qutip.sigmay()
    sz = .5*qutip.sigmaz()
    
    ### The global identity operator is constructed 
    loc_global_id = [qutip.tensor([qutip.qeye(2) for k in range(size)])]
    
    ### Lists of one-body operators are constructed, so that they all act on the full Hilbert space. 
    ### This is done via taking tensor products on lists of operators. 
    
    for n in range(size):
        operator_list = []
        for m in range(size):
            operator_list.append(id2)
        loc_globalid_list.append(loc_global_id)
        operator_list[n] = sx
        loc_sx_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sy
        loc_sy_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sz
        loc_sz_list.append(qutip.tensor(operator_list))        
    return loc_global_id, loc_sx_list, loc_sy_list, loc_sz_list

# In [2]: 

def all_upto_two_body_spin_ops(op_list, size):
    """
    This module constructs all pair-wise combinations
    (ie. correlators) of trivial and non-trivial 
    one-body operators (ie. sx, sy, sz operators only)
    Note that there are N(N+1)/2 different correlators  
    in an N-site spin chain.
    """
    loc_global_id_list, sx_list, sy_list, sz_list = op_list

    pauli_four_vec = [loc_global_id_list, sx_list, sy_list, sz_list];
        
    sxsa_list = []; sysa_list = []; szsa_list = []; two_body_s = [];
    
    sxsa_list = [sx_list[n] * pauli_four_vec[a][b] for n in range(size)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    
    sysa_list = [sy_list[n] * pauli_four_vec[a][b] for n in range(size)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    
    szsa_list = [sz_list[n] * pauli_four_vec[a][b] for n in range(size)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    # Notice that since you have added the identity, this contains also 0-body (the global id)
    # and 1-body operators...
    two_body_s = [sxsa_list, sysa_list, szsa_list]
    return two_body_s

# In [3]: 

def Heisenberg_Hamiltonian(op_list, chain_type, size, Hamiltonian_paras, closed_bcs = True, visualization = False):
    """
    This module constructs different types of nearest-neighbour
    Heisenberg-like spin chains, using a list of local [see Warnings and Remarks] spin operators.
    This module takes as input the following parameters:
    
        *♥*♥* 1. op_list: a list of hermitian local [see Warnings and Remarks further below] spin operators.
        *♥*♥* 2. chain_type: a string literal indicating which type of spin chain Hamiltonian
                             is to be constructed. 
        *♥*♥* 3. Size: the total number of lattice sites in the spin chain.
        *♥*♥* 4. Hamiltonian_paras: a list of four real-valued numbers Jx, Jy, Jz and h -in that order-,
                                    indicating the weights associated with each term in the chosen spin chain,
                                    where h is an external classical magnetic field.
        *♥*♥* 5. closed_bcs: boolean parameter, default value = True. 
                             If a spin chain has more than two lattice sites, the spin chain can be made
                             translationally invariant by adding an extra term corresponding to the N-th site's interaction
                             with the first site. 
        *♥*♥* 6. visualization: boolean parameter, default value = False.
                                Toggling this option will output a Hinton diagram of the Hamiltonian 
                                
        ====> Returns: the system's Hamiltonian.
        Warnings and Remarks: a. by "local spin operators", it is understood that these spin operators act on the global
                                 Hilbert space, having already been written in terms of tensor products. These spin operators
                                 act non-trivially only on a single lattice site, and act as an identity operator otherwise.
                              b. If a spin chain is chosen to be, eg., an XX chain, only the 
                                 first and fourth parameters will be use, the others will be ignored.
        
    """
    spin_chain_type = ["XX", "XY", "XYZ", "XXZ", "XXX", "Anderson"]
    loc_globalid_list, sx_list, sy_list, sz_list = op_list       
          
    H = 0; N = size
    Jx = Hamiltonian_paras[0] * 2 * np.pi #* np.ones(N)
    h =  Hamiltonian_paras[3] * 2 * np.pi #* np.ones(N)
    H += sum(-.5* h * sz_list[n] for n in range(N-1)) # Zeeman interaction 
    
    if (chain_type in spin_chain_type): 
        if (chain_type == "XX"):
            H += sum(-.5* Jx *(sx_list[n]*sx_list[n+1] 
                                 + sy_list[n]*sy_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H += .5* Jx *(sx_list[N-1]*sx_list[0] + sy_list[N-1]*sy_list[0])
        
        if (chain_type == "XY"):
            Jy = Hamiltonian_paras[1] * 2 * np.pi #* np.ones(N)
            H += sum(-.5* Jx* sx_list[n]*sx_list[n+1] 
                       -.5* Jy * sy_list[n]*sy_list[n+1] for n in range(N-1))
            
            if closed_bcs and N>2:
                H += -.5* Jx* sx_list[N-1]*sx_list[0] -.5* Jy * sy_list[N-1]*sy_list[0]
            
        elif (chain_type == "XXX"):
            H += sum(-.5* Jx * (sx_list[n]*sx_list[n+1] 
                                 + sy_list[n]*sy_list[n+1]
                                 + sz_list[n]*sz_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H += -.5* Jx * (sx_list[N-1]*sx_list[0] 
                                 + sy_list[N-1]*sy_list[0]
                                 + sz_list[N-1]*sz_list[0])
        
        elif (chain_type == "XXZ"):
            Jz =  Hamiltonian_paras[2] * 2 * np.pi #* np.ones(N)
            H += sum(-.5 * Jx * (sx_list[n] * sx_list[n+1] + sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H += -.5 * Jx * (sx_list[N-1] * sx_list[0] +
                                 sy_list[N-1] * sy_list[0]) 
                -.5 * Jz * (sz_list[N-1] * sz_list[0])
        
        elif (chain_type == "XYZ"):
            Jy = Hamiltonian_paras[1] * 2 * np.pi 
            Jz = Hamiltonian_paras[2] * 2 * np.pi 
            H += sum(-.5 * Jx * (sx_list[n] * sx_list[n+1]) 
                     -.5 * Jy * (sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H += -.5 * Jx * (sx_list[N-1] * sx_list[0])
                -.5 * Jy * (sy_list[N-1] * sy_list[0]) 
                -.5 * Jz * (sz_list[N-1] * sz_list[0])
                
        elif (chain_type == "Anderson"):
            pass
    else:
        sys.exit("Currently not supported chain type")
              
    if visualization:
        qutip.hinton(H)
    assert mat_ansys.non_hermitianess_measure(H) < 1e-10, "Non-hermitian Hamiltonian obtained" 
    return H

# In [4]:
        
def classical_ops(Hamiltonian, size, op_list, centered_x_op = False):
    
    identity_op = op_list[0][0]; sz_list = op_list[3]    
    labels = ["identity_op", "x_op", "p_op", "n_oc_op", "comm_xp", "corr_xp", "p_dot", "n_oc_disp"]
    
    cl_ops = {"identity_op": identity_op}
    if centered_x_op:
        cl_ops["x_op"] = sum((.5 + sz_list[k])*(k+1) for k in range(len(sz_list)))
    else:
        cl_ops["x_op"] = sum((k-size/2)*(sz_list[k] + .5 * identity_op) for k in range(len(sz_list)))  # el -1 no va, no?
        
    cl_ops["p_op"] = 1j * mat_ansys.commutator(cl_ops["x_op"], Hamiltonian)
    cl_ops["n_oc_op"] = sum([sz_list[k] + .5 * identity_op for k in range(len(sz_list))]) # el -1 no va, no?
    cl_ops["comm_xp"] = .5 * mat_ansys.anticommutator(cl_ops["x_op"], cl_ops["p_op"])
    cl_ops["corr_xp"] = -1j * mat_ansys.commutator(cl_ops["x_op"], cl_ops["p_op"])
    cl_ops["p_dot"] = 1j * mat_ansys.commutator(Hamiltonian, cl_ops["p_op"])
    cl_ops["n_oc_disp"] = (cl_ops["n_oc_op"]-1.)**2
    
    for i in range(len(labels)):
        if qutip.isherm(cl_ops[labels[i]]):
            pass
        else:
            print(labels[i], "not hermitian")
    return cl_ops, labels
