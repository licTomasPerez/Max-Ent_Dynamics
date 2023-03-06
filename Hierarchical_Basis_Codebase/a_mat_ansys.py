# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

# In [2]:

def ev_checks(rho, check_positive_definite = False, tol = 1e-5):
    """
    This module checks if a qutip.Qobj, a matrix, is either positive-definite or positive semi-definite, 
    i.e. whether or not all of its eigenvalues are strictly positive or non-negative, respectively. 
    This module takes as input the following parameters:
    
        *♥*♥* 1. rho: a qutip.Qobj.
        *♥*♥* 2. check_positive_definite: an optional boolean parameter 
                                            If it is toggled on, positive-definiteness 
                                            will be checked via a Cholesky decomposition.
                                          Otherwise, only positive semi-definiteness 
                                            will be analyzed, by explicitly checking the 
                                            matrix's eigenvalues. 
        *♥*♥* 3. tol: an optional boolean parameter, used only when positive semi-definiteness
                       is being analyzed, by establishing an upper bound on the smallest eigenvalue. 
        
        ====> Returns: a boolean, its truth value
                                  being whether or not 
                                  rho is a positive-definite
                                  matrix. 
        Warnings: None.
    """
    if isinstance(rho, qutip.Qobj):
        pass
    else:
        rho = qutip.Qobj(rho)
    
    if check_positive_definite: 
        try:
            rho = rho.full()
            np.linalg.cholesky(rho)
        except:
            return False
        return True
    else:
        ev_list = rho.eigenenergies()
        ev_list = sorted(rho.eigenenergies())
        min_ev = min(ev_list); ev_list = None
        if min_ev < 0:
            min_ev = abs(min_ev)
        if (min_ev < tol):
            return True
        else:
            return False 
        return True

def is_density_op(rho, verbose=False, critical=False, tol = 1e-5):
    """
    This module checks if the user-input QuTip.Qobj,
    rho, is a density operator or not. This is done 
    by checking if it is a hermitian, positive definite,  
    and trace-one, matrix. 
    This module takes as input the following parameters:
    
        *♥*♥* 1. rho: a qutip.Qobj,
        *♥*♥* 2. verbose: an optional boolean parameter 
                          for printing out logs,
                          stating which tests rho hasn't 
                          passed,
        *♥*♥* 3. critical: an optional boolean parameter
                           ???,
        *♥*♥* 4. tol: an optional boolean parameter for 
                      establishing a maximum tolerance 
                      for numerical errors, when computing
                      rho's trace. See Warnings further
                      below.
       
        ====> Returns: a boolean, its truth value
                                  being whether or not 
                                  rho is a density matrix.
                                  
        Warnings: Due to numerical instabilities, 
                  it may be possible for the trace 
                  to not be exactly one, even though 
                  it is supposed to be. Therefore, a 
                  cut-off is implemented to check for 
    """
    if not qutip.isherm(rho):
        if verbose:
            print("rho is not hermitian")
        assert not critical
        return True #False
    if abs(1 - rho.tr()) > tol:
        if verbose:
            print("Tr rho != 1, Tr rho = ", rho.tr())
        assert not critical
        return True #False
    if not ev_checks(rho):
        if verbose:
            print("rho is not positive")
        assert not critical
        return True #False
    return True

def non_hermitianess_measure(rho):
    """
    Returns a measure of the non-hermitian character
    of a user-input matrix, rho, by calculating the Frobenius 
    norm of the difference of rho and its adjoint.
    This module takes as input the following parameters:
    
        *♥*♥* 1. rho: a square matrix, not needing to be 
                      a QuTip.Qobj.
        ====> Returns: the norm ||rho - rho.dag()||_F
        
        Warnings: None
    """
    return linalg.norm(rho - rho.dag())

def null_matrix_check(rho, tol = 1e-5):
    return (linalg.norm(rho) < tol)

# In [3]:

def commutator(A, B):
    """
    Given two square matrices, this module computes 
    its commutator. 
    This module takes as input the following parameters:
        *♥*♥* 1. A: a complex-valued matrix,
        *♥*♥* 2. B: another complex valued matrix.
        ====> Returns: A*B - B*A
        
        Warnings: This module first checks the compatibi-
                  lity of the matrix dimensions. The
                  matrices need not to be QuTip.Qobj.
    """
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    return A*B - B*A

def anticommutator(A, B):
    """
    Given two square matrices, this module computes 
    its commutator. 
    This module takes as input the following parameters:
        *♥*♥* 1. A: a complex-valued matrix,
        *♥*♥* 2. B: another complex valued matrix.
        ====> Returns: A*B + B*A
        
        Warnings: This module first checks the compatibi-
                  lity of the matrix dimensions. The
                  matrices need not to be QuTip.Qobj.
    """
    result = 0
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    result += A*B+B*A
    return result

def basis_hermitian_check(basis):
    """
    Given a list or a dictionary of square Qutip.Qobj operators
    this module computes the non-hermitian character of said 
    operators by calculating the Frobenius norm of an operator
    and its adjoint. 
    
    This module takes as input the following parameters:
        *♥*♥* 1. basis: a length-M list of square Qutip.Qobj 
                        operators,
        ====> Returns: a length-M list of real-valued numbers,
                       where the i-th number is the following
                       Frobenius norm
                       
                       || basis[i] - basis[i].dag() ||_F
        
        Warnings: (a). This module assumes that the basis contains
                       square-matrices only. 
                  (b). If a dictionary is received, this module will 
                       extract its values, without changing the 
                       original basis' type.
    """
    if type(basis) is dict:
        basis_loc = basis.values()
    return [null_matrix_check(op - op.dag()) for op in basis_loc]

def basis_not_equal(basis1, basis2):
    """
    Given two sets of square matrices, 
    
    TODO
    """
    
    basis_elmts_dist = [("basis1 op:" + i, "basis2 op" + j, 
                         linalg.norm(basis1[i] - basis2[j])) for i in range(len(basis1))
                                                             for j in range(len(basis2))]
    return (basis_elmts_dist)

# In [3]:

def HS_inner_prod_t(op1, op2, rho0 = None): ### previous name: HS_inner_prod(A, B, rho0 = None):
    """
    Formally, this is the correct Hilbert-Schmidt 
    inner product. It is a complex valued inner 
    product on the space of all endomorphisms 
    acting on the N-partite Hilbert space 
    """
    if (op1.dims[0][0]==op2.dims[0][0]):   
        pass                                
    else:                                   
        raise Exception("Incompatible Qobj dimensions")
    
    if rho0 is None:
        rho0 = qutip.qeye(op1.dims[0])
        rho0 = rho0/rho0.tr()        
    else:
        if (is_density_op(rho0)):
            pass
        else:
            sys.exit("rho0 is not a density op")
    return (rho0 * (op1.dag() * op2)).tr()

def HS_inner_prod_r(op1, op2, rho0 = None): ### This inner product is real valued, provided both op1 and op2 are hermitian
    """
    Real-valued Hilbert-Schmidt inner product.
    Easier to compute for spin chains, as the 
    operator themselves can be written as 
    tensor products of local 1-body operatos.
    A global trace is then a product of traces
    over local Hilbert spaces. 
    """
    assert (op1.dims[0][0]==op2.dims[0][0]), "Incompatible Qobj dimensions"
    if rho0 is None:
        rho0 = qutip.qeye(op1.dims[0])
        rho0 = rho0/rho0.tr()
    else:
        assert is_density_op(rho0), "rho0 is not a density op" 
    return .5 * (rho0 * anticommutator(op1.dag(), op2)).tr()

def HS_inner_norm(op, rho0, sc_prod): ### previous name: mod_HS_inner_norm
    return sc_prod(op, op, rho0)

def HS_normalize_op(op, rho0, sc_prod):
    return op/sc_prod(op, op, rho0)

def HS_distance(rho, sigma, rho0, sc_prod):
    assert rho.dims[0][0]==sigma.dims[0][0], "Incompatible Qobj dimensions"
    return sc_prod(rho-sigma, rho-sigma, rho0)

def base_orth(ops, rho0, sc_prod, visualization = False, reinforce_reality=False):
    """
    A Gram-Schmidt procedure is implemented for 
    orthonormalizing a given basis. It takes as input:
    *. a list or dictionary of N-body operators,
    *. a reference state rho0,
    *. a Hilbert-Schmidt-type inner product, be it
         its real valued version or the default
         complex-valued one,
    *. an optional boolean parameter for visualizing 
        the i-th operator's inner products with 
        all operators in the basis,
    *. an optional boolean parameter for taking only
        real results, dumping complex-valued numbers. 
    """
    if isinstance(ops, dict):
        ops = [ops[key] for key in ops]
    if isinstance(ops[0], list):
        ops = [op for op1l in ops for op in op1l]
    basis = []
    for i, op in enumerate(ops): 
        op_norm = np.sqrt(sc_prod(op, op, rho0))
        op = op/op_norm
        alpha = np.array([sc_prod(op2, op, rho0) for op2 in basis])
        if reinforce_reality:
            alpha = alpha.real
        if visualization:
            print(alpha)
        op_mod = op - sum([c*op2 for c, op2, in zip(alpha, basis)])
        op_norm = np.sqrt(sc_prod(op_mod,op_mod,rho0))
        if visualization:
                print("*****************norm", op_norm)
        if op_norm > 1.e-5:
            op_mod = op_mod/(op_norm)
            basis.append(op_mod)
    return basis

# In [4]: 

def logM(rho, svd = True):
    """
    Evaluates the logarithm of a positive matrix rho.
    """
    assert ev_checks(rho), "Non positive-defined input matrix"
    if isinstance(rho, qutip.Qobj):
        qutip_form = True
        dims = rho.dims
    else:
        qutip_form = False        

    if svd:            
        if qutip_form:
            rho = rho.full()
        U, Sigma, Vdag = linalg.svd(rho, full_matrices = False)
        matrix_log = U @ np.diag(np.log(Sigma)) @ U.conj().transpose() 
    else: 
        if qutip_form:
            eigvals, eigvecs = rho.eigenstates()
            matrix_log = sum([np.log(vl)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)]) 
        else:
            rho = rho.full()
            eigvals, eigvecs = linalg.eigh(rho)
            return evecs @ np.array([np.log(ev)*np.array(f) for ev, f in zip(evals, evecs.transpose().conj())])
    
    if qutip_form:
        matrix_log = qutip.Qobj(matrix_log, dims)
    return matrix_log

def sqrtM(rho, svd = True):
    """
    Evaluates the square root of a positive matrix rho.
    """
    assert ev_checks(rho), "Non positive-defined input matrix"
    if isinstance(rho, qutip.Qobj):
        qutip_form = True
        dims = rho.dims
    else:
        qutip_form = False

    if svd:
        if qutip_form:
            rho = rho.full()
        U, Sigma, Vdag = linalg.svd(rho, full_matrices = False)
        matrix_log = U @ np.diag((Sigma)**.5) @ U.conj().transpose() 
    else: 
        if qutip_form:
            eigvals, eigvecs = rho.eigenstates()
            matrix_log = sum([(vl**.5)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)]) 
        else:
            rho = rho.full()
            eigvals, eigvecs = linalg.eigh(rho)
            return evecs @ np.array([(ev**.5)*np.array(f) for ev, f in zip(evals, evecs.transpose().conj())])
    
    if qutip_form:
        matrix_log = qutip.Qobj(matrix_log, dims)
    return matrix_log

def bures(rho, sigma, svd = True):
    """
    Evaluates the Bures metric between two density states. 
    """
    assert is_density_op(rho), "rho is not a density operator"
    assert is_density_op(sigma), "sigma is not a density operator"
    
    sqrt_sigma = sqrtM(sigma.full(), svd=svd)
    fidelity = sqrtM((sqrt_sigma @ rho.full()  @sqrt_sigma),svd=True).trace().real

    assert abs(fidelity.imag)<1.e-10, f"complex fidelity? fidelity={fidelity}"
    fidelity = fidelity.real
    assert 0 <= fidelity, f"negative fidelity? fidelity={fidelity}"
    if fidelity>1.05:
        assert (fidelity-1)<1.e-8, f"error in fidelity too large fidelity={fidelity}"
        return 0.
    return  np.arccos(fidelity)/np.pi

# In [5]:

def proj_op(K, basis, rho0, sc_prod):
    return sum([sc_prod(b, K,rho0) * b for b in basis])

def rel_entropy(rho, sigma, svd = True):
    if svd:
        val = (rho*(logM(rho, True) - logM(sigma, True))).tr()
    else:
        assert ((ev_checks(rho) and ev_checks(sigma))), "Either rho or sigma non positive"
        val = (rho*(logM(rho, False)-logM(sigma, False))).tr()
        if (abs(val.imag - 0)>1.e-10):
            val = None
            raise Exception("Either rho or sigma not positive")
    return val.real

def bures_vectorized(rhot_list, sigmat_list):
    assert len(rhot_list) == len(sigmat_list), "Lists not of the same length"
    bures_list_vcz = [bures(rhot_list[t], sigmat_list[t]) for t in range(len(rhot_list))]
    return bures_list_vcz

def relative_entropies_vectorized(rhot_list, sigmat_list):
    assert len(rhot_list) == len(sigmat_list), "Lists not of the same length"
    rel_rho_v_sigma = [rel_entropy(rhot_list[t], sigmat_list[t]) for t in range(len(rhot_list))]
    rel_sigma_v_rho = [rel_entropy(sigmat_list[t], rhot_list[t]) for t in range(len(rhot_list))]
    return rel_rho_v_sigma, rel_sigma_v_rho

# In [6]:

def recursive_basis(depth, seed_op, Hamiltonian, rho0): 
    """
    Build a basis of the form 
    [c_0(seed_op), c_1(seed_op), ... c_depth(seed_op)]
    with c_0(op)=op-<op>
    
    c_1(op) = [Hamiltonian, op]-<[Hamiltonian, op]> id_M
    
    c_{n+1}(op) = c_1(c_{n}(op))
    
    As a result, this is a zero-average basis of hermitian operators
    that expands the order `depth` Dyson's series for
    seed_op(t).
    """
    basis = [seed_op]; 
    if isinstance(depth, int) and depth > 0: 
        for i in range(1, depth):
            loc_op = qutip.Qobj(-1j * commutator(Hamiltonian, basis[i-1]))
            if (linalg.norm(loc_op) < 1e-10):
                loc_op = None
                break
                print("Log report: Hierarchichal Basis terminated due to null commutator.")
            loc_op = (loc_op * rho0).tr() - loc_op
            basis.append(loc_op)
    elif (depth == 0):
        pass
    return basis

def vectorized_recursive_basis(depth_and_ops, Hamiltonian, rho0):        
    basis_rec = []
    for depth, op in depth_and_ops: 
        basis_rec += recursive_basis(depth, op, Hamiltonian, rho0)
    return basis_rec

# In [7]:

def exact_v_proj_ev_matrix_metrics_multiple(timespan, range_of_temps_or_dims, multiple_evolutions,
                                              plot_var_HierarchBases_dim = False):
    
    z = timespan[:-1]
    bures_Ex_v_Proj_all = {}
    relEntropy_Ex_v_Proj_all = {}
    relEntropy_Proj_v_Ex_all = {}
    
    if (plot_var_HierarchBases_dim == False):
            print("No visualization choice taken")
    
    if plot_var_HierarchBases_dim:
        range_HB_dims = range_of_temps_or_dims
        ### The list is searched twice. First, to identify and extract the data from the exact evolution via sigmat_list.
        ### And then to actually compute the metrics of a particular ProjEv with the exact evolution data. 
        
        for dim in range_HB_dims: 
            res_exact_loc = multiple_evolutions["res_exact"]['res_exact_MaxEnt' + str(range_HB_dims.index(dim))]
            if res_exact_loc is None:
                pass
            else: 
                sigmat_list = res_exact_loc.states[:-1]
        
        for dim in range_HB_dims: 
            rhot_list = multiple_evolutions["all_max_ent_evs"]["res_evs_MaxEnt" + str(range_HB_dims.index(dim))]["State_ev"]
            sigmat_list = res_exact_loc.states[:-1] ### acá hay un bug raro, si no pongo esta línea nuevamente, se borra sigmat_list y
                                                    ### queda en None. Así funciona. 
            
            bures_Ex_v_Proj_all["Max-Ent" + str(range_HB_dims.index(dim))] = bures_vectorized(rhot_list = rhot_list,
                                                                                      sigmat_list = sigmat_list)
            local = relative_entropies_vectorized (rhot_list = rhot_list, sigmat_list = sigmat_list)
            relEntropy_Proj_v_Ex_all["Max-Ent" + str(range_HB_dims.index(dim))] = local[0]
            relEntropy_Ex_v_Proj_all["Max-Ent" + str(range_HB_dims.index(dim))] = local[1]
            rhot_list = None; sigmat_list = None; 
        
    return bures_Ex_v_Proj_all, relEntropy_Ex_v_Proj_all, relEntropy_Proj_v_Ex_all
