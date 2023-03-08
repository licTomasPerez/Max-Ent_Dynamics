# Restricted Max-Ent Dynamics &middot; [![Build Status](https://img.shields.io/travis/npm/npm/latest.svg?style=flat-square)](https://github.com/licTomasPerez) [![npm](https://img.shields.io/npm/v/npm.svg?style=flat-square)](https://www.npmjs.com/package/npm) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/your/your-project/blob/master/LICENSE)

<br />
<p align="center">
  
  <h2> A Systematic Method for constructing Approximate Closed Quantum Dynamics for Interacting Spin Systems </h2>
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

For a large many-body environment-non-interacting quantum system, the Schr\"odinger equation describes the coupled evolution of all of the system's possible correlations. The exact solution is, in all but a few cases, an analytically intractable and numerically expensive problem to solve, since the number of correlations present in an $N$-body quantum system, grows as ${\cal O}(\exp N)$. Hence, this coupled evolution implies that the configuration space of density states is a very intricate and complicated mathematical object. 

We circumvent this dramatic overhead in computational resources by noting that not all correlations equally weigh in on the system's dynamics, allowing us to develop a systematic approach for simulating approximate closed dynamics, combining the notions of **Gaussianizations** and **Max-Ent** and the idea of **orthogonal projection**. In particular, we consider closed approximate dynamics of **generalized Gaussian states**, defined via the Max-Ent principle.
These generalized Max-Ent states, then, contain some of the most relevant correlations present in the system. 

These ideas and methods will be used, in this repository, to simulate the dynamics of the Heisenberg-like XX and XY models. 

<p align="center">
<img src="figs_readme/Restricted_ME_algo.jpg" style="width:500px;height:500px;">
</p>

<p align="center">
<img src="figs_readme/Comparing_Exact_vs_RME.jpg" style="width:500px;height:500px;">
</p>

## Built with:

## Prerrequisites for use:

## Usage:

## Contributing

## Specifications
