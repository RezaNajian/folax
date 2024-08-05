<p align=center><img height="32.125%" width="32.125%" src="https://github.com/RezaNajian/FOL/assets/62375973/0e1ca4e0-0658-4f5d-aad9-1ae7c9f67574"></p>


# FOL: Efficient Solution and Optimization of PDEs
Finite Operator Learning (FOL) combines neural operators, physics-informed machine learning, and numerical methods to solve partial differential equations without data, providing accurate sensitivities and enabling efficient gradient-based optimization. It uses a feed-forward neural network to map the design space to the solution space while ensuring compliance with physical laws.

# Installation Guide
To install FOL, follow these steps:
   ```sh
   git clone https://github.com/RezaNajian/FOL.git
   cd FOL
   python3 setup.py sdist bdist_wheel
   pip install -e .
   ```
To run the tests:
   ```sh
   pytest -s tests/
   ```
To run the examples:
   ```sh
   cd examples
   python3 examples_runner.py
   ```
# How to cite FOL?
Please, use the following references when citing FOL in your work.
- [Shahed Rezaei, Reza Najian Asl, Kianoosh Taghikhani, Ahmad Moeineddin, Michael Kaliske, and Markus Apel. "Finite Operator Learning: Bridging Neural Operators and Numerical Methods for Efficient Parametric Solution and Optimization of PDEs." arXiv preprint arXiv:2407.04157 (2024).](https://arxiv.org/pdf/2407.04157)
- [Shahed Rezaei, Reza Najian Asl, Shirko Faroughi, Mahdi Asgharzadeh, Ali Harandi, Rasoul Najafi Koopas, Gottfried Laschet, Stefanie Reese, Markus Apel. "A finite operator learning technique for mapping the elastic properties of microstructures to their mechanical deformations." arXiv preprint arXiv:2404.00074 (2024)](https://arxiv.org/pdf/2404.00074)
