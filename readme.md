# EquationSolver

A simple tool for automatic balancing of chemical equations.
It parses formulas with indices and parentheses, builds a system of equations for atoms, and solves it using Gaussian elimination.

## Features
- Supports complex formulas with groups, e.g. `(NH4)2SO4`
- Console output with ANSI color highlighting
- Interactive CLI mode

## Usage
Run:
```bash
python main.py
```

Example:
```
?> (NH4)2Cr2O7 -> Cr2O3 + N2 + H2O
!> (NH4)2Cr2O7 -> Cr2O3 + N2 + 4H2O
```
