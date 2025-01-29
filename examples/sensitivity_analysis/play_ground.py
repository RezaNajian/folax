import jax
import jax.numpy as jnp
from sympy import symbols, Matrix
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify

# Define symbolic variables for the vector components
Ux, Uy, Uz = symbols('Ux Uy Uz')

# Define the vector U
U = Matrix([Ux, Uy, Uz])

# Function 1: Scalar output using inner product, directly using U
# Define the symbolic expression as a string
scalar_formula_string = "U.dot(2*U - Matrix([Ux, Uy, Uz]))"  # Inner product with combination of U
scalar_formula_string = "U.dot(2*U)-Ux"  # Inner product with combination of U
# Parse the string into a symbolic expression
scalar_formula_expr = parse_expr(scalar_formula_string, local_dict={"U": U, "Matrix": Matrix})

# Convert the scalar expression to a JAX-compatible function
jax_scalar_function = lambdify((Ux, Uy, Uz), scalar_formula_expr, 'jax')
jit_scalar_function = jax.jit(lambda U: jax_scalar_function(U[0], U[1], U[2]))

# Function 2: Vector output
# Define the symbolic vector expression as a string
vector_formula_string = "[Ux**2, Uy**2 + Uz, Ux * Uy - Uz]"
vector_formula_expr = parse_expr(vector_formula_string)  # Parse the string into a symbolic expression

# Convert the vector expression to a JAX-compatible function
jax_vector_function = lambdify((Ux, Uy, Uz), vector_formula_expr, 'jax')
jit_vector_function = jax.jit(lambda U: jax_vector_function(U[0], U[1], U[2]))

# Evaluate the functions
input_vector = jnp.array([2.0, 3.0, 4.0])  # Example input: Ux=2, Uy=3, Uz=4

# Scalar output
scalar_result = jit_scalar_function(input_vector)
print("Scalar Output:", scalar_result)  # Example output: scalar value

grad_scalar_result = jax.grad(jit_scalar_function)(input_vector)

print(grad_scalar_result)

# # Vector output
# vector_result = jit_vector_function(input_vector)
# print("Vector Output:", vector_result)  # Example output: [Ux^2, Uy^2 + Uz, Ux * Uy - Uz]
