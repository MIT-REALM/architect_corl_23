import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Define a toy potential
    def U(x):
        return (x - 2) * (x + 2) * (x - 1.1) * (x + 1.1) * (x - 0.4) * (x + 0.4)

    # And a toy prior
    def logprior(x):
        return -(x**2)

    # Define the tempered logprob
    def logprob(x, t):
        return logprior(x) - t * U(x)

    # Make a function for normalizing the potentials
    def normalize(logp):
        return jnp.exp(logp) / jnp.sum(jnp.exp(logp))

    x = jnp.linspace(-2, 2, 800)
    logp = logprior(x)
    p = normalize(logp)
    plt.plot(x, p)
    plt.savefig("test1.png")
    plt.clf()

    logp = logprior(x) - U(x)
    p = normalize(logp)
    plt.plot(x, p)
    plt.savefig("test2.png")
    plt.clf()

    # Plot
    x = jnp.linspace(-2, 2, 800)
    t = jnp.linspace(0, 1, 1200)
    # t = 1 - jnp.exp(-3 * t)
    logp = jax.vmap(logprob, in_axes=(None, 0))(x, t)
    p = jax.vmap(normalize)(logp)
    plt.imshow(p.T)
    plt.axis("off")

    # Sample
    stepsize = 0.05

    def sample_scan(x_current, input):
        key, t = input
        key, subkey = jax.random.split(key)
        x_proposal = x_current + stepsize * jax.random.normal(
            subkey, shape=x_current.shape
        )
        logp_proposal = logprob(x_proposal, t)
        logp_current = logprob(x_current, t)
        logp_accept = logp_proposal - logp_current
        accept = jax.random.uniform(key) < jnp.exp(logp_accept)
        x_next = jax.lax.cond(accept, lambda: x_proposal, lambda: x_current)
        return x_next, x_next

    def sample_x(key, x_init, t):
        # Run the scan
        keys = jax.random.split(key, t.shape[0])
        x_sampled = jax.lax.scan(sample_scan, x_init, (keys, t))[1]
        return x_sampled

    key = jax.random.PRNGKey(0)
    n_chains = 3
    x_init = jnp.linspace(-0.5, 0.5, n_chains)
    x_sampled = jax.vmap(sample_x, in_axes=(0, 0, None))(
        jax.random.split(key, n_chains), x_init, t
    ).T
    w, h = p.shape
    plt.plot(t * w, -x_sampled * h / 4 + h / 2, "r-", linewidth=0.75)

    plt.savefig("test.png")
