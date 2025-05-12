import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, lambdify, sympify, diff, integrate, simplify, pprint
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Numerical nth derivative
def nth_derivative(f, x, n=1, dx=1e-6):
    if n == 1:
        return (f(x + dx) - f(x - dx)) / (2 * dx)
    elif n == 2:
        return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx ** 2)
    elif n == 3:
        return (f(x + 2*dx) - 2*f(x + dx) + 2*f(x - dx) - f(x - 2*dx)) / (2 * dx ** 3)
    else:
        raise ValueError("Only 1st, 2nd, and 3rd derivatives supported.")

# Central difference for plotting first derivative
def numerical_derivative(f, x_vals, h=1e-5):
    return (f(x_vals + h) - f(x_vals - h)) / (2 * h)

# Numerical integral
def numerical_integral(f, x_vals):
    integral_vals = []
    for x in x_vals:
        result, _ = quad(f, x_vals[0], x)
        integral_vals.append(result)
    return np.array(integral_vals)

# Plot function, first derivative, and integral
def plot_all(x_vals, f_vals, d_vals, i_vals):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, f_vals, label='Original Function', linewidth=2)
    plt.plot(x_vals, d_vals, label='First Derivative', linestyle='--')
    plt.plot(x_vals, i_vals, label='Integral', linestyle=':')
    plt.title('Function, Derivative, and Integral')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot higher-order derivatives
def plot_higher_order_derivatives(f, x_vals):
    f1 = lambda x: nth_derivative(f, x, n=1)
    f2 = lambda x: nth_derivative(f, x, n=2)
    f3 = lambda x: nth_derivative(f, x, n=3)

    y_vals = f(x_vals)
    y1_vals = np.array([f1(xi) for xi in x_vals])
    y2_vals = np.array([f2(xi) for xi in x_vals])
    y3_vals = np.array([f3(xi) for xi in x_vals])

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='Original Function', linewidth=2)
    plt.plot(x_vals, y1_vals, label='1st Derivative', linestyle='--')
    plt.plot(x_vals, y2_vals, label='2nd Derivative', linestyle='-.')
    plt.plot(x_vals, y3_vals, label='3rd Derivative', linestyle=':')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function and Higher-Order Derivatives')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Get user-defined function
def get_function():
    x = symbols('x')
    user_input = input("Enter a function in terms of x (e.g., x**2 + 3*x + 5): ")
    try:
        expr = sympify(user_input)
        f = lambdify(x, expr, modules=['numpy'])
        return expr, f
    except Exception as e:
        print(f"Error parsing function: {e}")
        return None, None

# Main execution
def main():
    expr, f = get_function()
    if not f:
        return

    x = symbols('x')

    # Print symbolic expressions
    print("\nSymbolic Outputs:")
    print("Original Function:")
    pprint(expr)

    print("\nFirst Derivative:")
    derivative_expr = simplify(diff(expr, x))
    pprint(derivative_expr)

    print("\nIndefinite Integral:")
    integral_expr = simplify(integrate(expr, x))
    pprint(integral_expr)

    try:
        a = float(input("\nEnter start of range (e.g., -10): "))
        b = float(input("Enter end of range (e.g., 10): "))
    except ValueError:
        print("Invalid range.")
        return

    x_vals = np.linspace(a, b, 400)

    try:
        f_vals = f(x_vals)
        d_vals = numerical_derivative(f, x_vals)
        i_vals = numerical_integral(f, x_vals)

        plot_all(x_vals, f_vals, d_vals, i_vals)
        plot_higher_order_derivatives(f, x_vals)

    except Exception as e:
        print(f"Computation error: {e}")

if __name__ == "__main__":
    main()

#CreatedbyRSE
