import math
from scipy.stats import norm, binom
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


def plot_de_moivre_laplace(n, p):
    """
    Plot the binomial distribution and its normal approximation using
    the De Moivre-Laplace theorem.

    Parameters:
    - n: Number of trials in the binomial distribution.
    - p: Probability of success in each trial.
    """

    # Create an array of possible outcomes for the binomial distribution
    x = np.arange(0, n + 1)

    # Calculate the binomial PMF (Probability Mass Function) for each outcome
    binom_pmf = binom.pmf(x, n, p)

    # Mean and standard deviation of the normal distribution approximation
    mean = n * p
    std_dev = np.sqrt(n * p * (1 - p))

    # Calculate the normal PDF (Probability Density Function) for the same outcomes
    normal_pdf = norm.pdf(x, mean, std_dev)

    # Plot the binomial distribution as bars
    plt.bar(x, binom_pmf, width=0.5, color='blue', alpha=0.8, label="Binomial PMF")

    # Plot the normal distribution as a line
    plt.plot(x, normal_pdf, color='red', lw=2, label="Normal Approximation (PDF)")

    # Add labels and title
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distribution vs. Normal Approximation\nn = {n}, p = {p}')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()


def plot_de_moivre_laplace_cdf(n, p):
    """
    Plot the binomial CDF and its normal approximation (De Moivre-Laplace CDF).

    Parameters:
    - n: Number of trials in the binomial distribution.
    - p: Probability of success in each trial.
    """

    # Create an array of possible outcomes for the binomial distribution
    x = np.arange(0, n + 1)

    # Mean and standard deviation for the normal approximation
    mean = n * p
    std_dev = np.sqrt(n * p * (1 - p))

    # Calculate the normal CDF for the same outcomes
    normal_cdf = norm.cdf(x, mean, std_dev)

    # Plot the normal CDF as a smooth line
    plt.plot(x, normal_cdf, color='red', lw=2, label="Normal Approximation (CDF)")

    # Add labels and title
    plt.xlabel('Number of Successes')
    plt.ylabel('Cumulative Probability')
    plt.title(f'Normal Approximation (CDF)\nn = {n}, p = {p}')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


def exact_probability(n, p, k):
    return (math.factorial(n) / (math.factorial(k) * math.factorial(n - k))) * (p ** k) * ((1 - p) ** (n - k))


def exact_probability_cdf(n, p, k1, k2):
    exact_prob = 0
    for k in range(k1, k2 + 1):
        exact_prob += exact_probability(n, p, k)
    return exact_prob


# De Moivre Laplace Theorem
def de_moivre_laplace_approximation(n, p, k):
    """
    Approximate the probability P(K)
    using the De Moivre-Laplace theorem.

    Parameters:
    - n: Number of trials in binomial distribution
    - p: Probability of success in each trial
    - k: Number of desired successful trials

    Returns:
    - Approximation of P(K) using a normal distribution.
    """

    # Mean and standard deviation of the binomial distribution
    mean = n * p
    std_dev = math.sqrt(n * p * (1 - p))

    z_k = (k - mean) / std_dev

    # Use the normal CDF to approximate the binomial probability
    probability = normal_density(z_k) * (1 / std_dev)

    return probability


# De Moivre Laplace Integral Theorem
def normal_density(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-(x ** 2) / 2)


def de_moivre_laplace_integral(n, p, k1, k2):
    mean = n * p
    std_dev = math.sqrt(n * p * (1 - p))

    z_k1 = (k1 - mean) / std_dev
    z_k2 = (k2 - mean) / std_dev

    # Use scipy.integrate.quad to compute the integral of the normal density function
    integral_value1, _ = quad(normal_density, -math.inf, z_k1)
    integral_value2, _ = quad(normal_density, -math.inf, z_k2)
    return integral_value2 - integral_value1


# Example usage
n = int(input("Enter the Number of trials (n): "))  # Number of trials
p = float(input("Enter the Probability of success in each trial (p): "))  # Probability of success in each trial
k = int(input("Enter the Number of desired successful trials (k): "))  # Number of desired successful trials

approx_prob = de_moivre_laplace_approximation(n, p, k)
exact_prob = exact_probability(n, p, k)
print(f"Approximate probability P({k}): {approx_prob}")
print(f"Exact probability P({k}): {exact_prob}")
print(f"The Error percentage is: {(abs(exact_prob-approx_prob)/exact_prob)*100}%")
plot_de_moivre_laplace(n, p)

print(f"Your k1 and k2 should be between {n*p - 3*math.sqrt(n*p*(1-p))} and {n*p + 3*math.sqrt(n*p*(1-p))} in order to get the most accurate results for the approximation.")
# Example usage with integral
k1 = int(input("Enter the Lower bound (k1): "))  # Lower bound
k2 = int(input("Enter the Upper bound (k2): "))  # Upper bound
approx_prob = de_moivre_laplace_integral(n, p, k1, k2)
exact_prob = exact_probability_cdf(n, p, k1, k2)
print(f"Approximate probability using integral P({k1} <= X_n <= {k2}): {approx_prob}")
print(f"Exact probability using integral P({k1} <= X_n <= {k2}): {exact_prob}")
print(f"The Error percentage is: {(abs(exact_prob-approx_prob)/exact_prob)*100}%")

plot_de_moivre_laplace_cdf(n, p)
