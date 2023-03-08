# Gaussian-like Restricted Max-Ent Dynamics for Spin Systems; 

[![Build Status](https://img.shields.io/badge/build-v1.0.0-brightgreen)](https://github.com/licTomasPerez) 
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 
[![GitHub license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/your/your-project/blob/master/LICENSE)

<br />
<p align="center">
  <h2> A Systematic Algorithm for the simulation of approximate closed Quantum Dynamics for Interacting Spin Systems </h2>
  <p align="center">
  </p>
</p>

Codebase and Documentation for the upcoming paper titled "Gaussian-like Max-Ent restricted Dynamics for Spin Systems"

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>

  </ol>
</details>

## Overview:

For a large many-body environment-non-interacting quantum system, the Schr\"odinger equation describes the coupled evolution of all of the system's possible correlations. The exact solution is, in all but a few cases, an analytically intractable and numerically expensive problem to solve, since the number of correlations present in an $N$-body quantum system, grows as ${\cal O}(\exp N)$.
Hence, the system's configuration space -the set of density states- is a very intricate and complicated mathematical object. The exact solution of the Schr\"odinger equation is equivalent to searching through this configuration space, at each step, looking for the correct state, a computationally very expensive problem. 

We circumvent this dramatic overhead in computational resources by noting that not all correlations equally weigh in on the system's dynamics, allowing us to develop a systematic approach for simulating approximate closed dynamics. In particular, we develop the notion of families (manifolds) ${\cal V}^{\ell}$ of **Generalized Gaussian States*, which are **Max-Ent** states with respect to a small set of observables, a **Hierarchical Basis** ${\cal B}^{\ell}$, whose elements are given in terms of iterated commutators with respect to a given -physically relevant- *seed operator*. 

This notion of *Gaussianization* combined with the *Max-Ent principle*, thus, allows for an exact mapping of the Schr\"odinger equation on the density states $\rho$, which are ${\cal O}(\exp 2N)$ dimensional objects, to an equation of motion on the correlations themselves, which are simply continous functions. 
These **restricted Max-Ent evolutions*, then, are much less computationally expensive than exact calculation of the system's dynamics.

Moreover, even though the mapping is exact, we develop a systematic method of constructing approximations, which should in principle yield better and better results as the Hierarchical Bases augments. 

<p align="center">
<img src="figs_readme/Comparing_Exact_vs_RME.jpg" style="width:300px;height:300px;">
</p>                                                                    
                                                         
These ideas and methods will be used, in this repository, to simulate the dynamics of the Heisenberg-like XX and XY models, which will be treated in the <a href= https://github.com/licTomasPerez/Max-Ent_Dynamics/tree/main/Hierarchical_Basis_Codebase/Tutorials>tutorials</a> section. 
We include a short pseudo-code explanation of the main steps required for a restricted Max-Ent evolution. 


<p align="center">
<img src="figs_readme/Restricted_ME_algo.jpg" style="width:300px;height:300px;">
</p>    

## Built with:

This implementation has been written in Python 3, using several external and user-defined libraries, namely

* [Quantum Toolbox in Python](https://qutip.org/qutip-tutorials/)
* [NumPy](https://numpy.org/doc/)
* [SciPy](https://docs.scipy.org/doc/scipy/index.html)

## Prerrequisites for use:

Have Python 3 installed

1. Clone the repo
   ```sh 
   git clone https://github.com/licTomasPerez/Max-Ent_Dynamics.git
   ```
2. Optional, but highly recommended. Create a virtual environment to avoid conflict with other dependencies
  ```sh
  python3 -m virtualenv {NameOfVirtualEnv}
  ```
  And activate the virtual environment
  ```sh
  source {NameOfVirtualEnv}/bin/activate.sh
  ```
3. Install libraries
   ```sh
   (NameOfVirtualEnv) pip3 install -r requirements.txt
   ```
4. Now you are ready to run custom restricted Max-Ent evolutions!
  ```sh
  (NameOfVirtualEnv) python3 meta_main.py
  ```
  
## Usage:

First and foremost, one must characterize the spin system. The code comes with a series of pre-defined Hamiltonians (e.g. XX, XY, XXX, etc.), with or without closed boundary conditions, etc., for which one must only specify the relevant parameters. 

Then, one must specify the desired $\ell$ dimensions, for which the restricted Max-Ent evolution on ${\cal V}^{\ell}$ is desired, alongside the chosen observables of interest, for which their time evolved expectation values are desired. 

Optionally, one can also compute the exact evolution by invoking QuTiP's *mesolve* function, for easy comparison of the approximate and exact results. 
Moreover, a series of <a href= https://github.com/licTomasPerez/Max-Ent_Dynamics/tree/main/Hierarchical_Basis_Codebase/Tutorials>tutorials</a> have been written in order to demonstrate how to carry out restricted Max-Ent evolutions for some simples quantum mechanical spin systems.

## Contributing

Contributions are open! Feel free to contact me via my socials. 

## Specifications
