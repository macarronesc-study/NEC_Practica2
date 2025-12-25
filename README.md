# JSSP Optimization with Genetic Algorithms

**Course:** Neural and Evolutionary Computation (NEC) - Assignment 2
**Author:** Daniel Alejandro Coll Tejeda
**Date:** December 2025

## Description
This project implements a **Genetic Algorithm (GA)** to solve the Job Shop Scheduling Problem (JSSP). The solver minimizes the *makespan* (total execution time) by exploring various combinations of genetic operators.

For comparison purposes, a **Simulated Annealing (SA)** algorithm has also been implemented as a second optimization method.

## Features
*   **Representation:** Operation-Based Representation (Permutation with Repetition).
*   **GA Configurations:**
    *   **Selection:** Tournament, Roulette Wheel.
    *   **Crossover:** Job-Based Order Crossover (JOX), Precedence Operation Crossover (POX).
    *   **Mutation:** Swap, Insert.
*   **Benchmarks:** Optimized for instances `ft06`, `la01`, `la29`, and `la40`.
*   **Visualization:** Automatic generation of Gantt charts and convergence plots.
*   **Robustness:** Datasets are embedded directly in the code, allowing the script to run standalone without external file dependencies.

## Prerequisites
The project requires **Python 3.x** and the following libraries:

```bash
pip install numpy matplotlib
```

## How to Run
Execute the main script from the terminal:

```bash
python jssp_solver-final.py
```

*Note: The script checks for `jobshop1.txt`. If not found, it automatically uses the embedded fallback data for the required instances.*

## Output
Upon execution, the script will:
1.  Print a comparison table of 6 different GA configurations and the SA results to the console.
2.  Generate the following image files in the working directory:
    *   `[instance]_convergence.png`: Graph comparing GA and SA evolution.
    *   `[instance]_gantt.png`: Gantt chart of the best schedule found.