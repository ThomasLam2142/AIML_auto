import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Fetch historical stock price data for ticker
ticker = "AMD"
data = yf.download(ticker, start="2020-01-01", end="2024-12-18")

# Calculate daily returns
data['Return'] = data['Close'].pct_change().dropna()

# Model definition
def bayesian_model(returns):
    alpha = numpyro.sample("alpha", dist.Normal(0,1 )) # Intercept
    beta = numpyro.sample("beta", dist.Normal(0,1)) # Coefficient
    sigma = numpyro.sample("sigma", dist.Exponential(1)) # Noise
    
    mean = alpha + beta * returns[:1]
    numpyro.sample("obs", dist.Normal(mean, sigma), obs=returns[1:])
    
# Prepare data for training
returns = data['Return'].dropna().values
returns = jnp.array(returns)

# Set up MCMC
rng_key = random.PRNGKey(0)
kernel = NUTS(bayesian_model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=1)

# Run MCMC to estimate parameters
mcmc.run(rng_key, returns)
samples = mcmc.get_samples()

# Print parameter estimates
print("Parameter Estimates:")
print(f"Alpha: {samples['alpha'].mean()}")
print(f"Beta: {samples['beta'].mean()}")
print(f"Sigma: {samples['sigma'].mean()}")

# Predict future returns using posterior samples
alpha_post = samples['alpha']
beta_post = samples['beta']
sigma_post = samples['sigma']

# Simulate future returns
future_returns = []
n_future_days = 30  # Number of days to forecast
last_return = returns[-1]

for _ in range(n_future_days):
    mean = alpha_post + beta_post * last_return
    future_return = np.random.normal(mean.mean(), sigma_post.mean())
    future_returns.append(future_return)
    last_return = future_return

# Plot predictions with uncertainty
plt.figure(figsize=(10, 6))
plt.plot(future_returns, label="Predicted Returns")
plt.fill_between(range(n_future_days), 
                 jnp.array(future_returns) - 2 * sigma_post.mean(),
                 jnp.array(future_returns) + 2 * sigma_post.mean(),
                 color="lightblue", alpha=0.5, label="Uncertainty (2σ)")
plt.legend()
plt.title("Predicted Returns with Uncertainty")
plt.savefig("predictions.png")
plt.close()