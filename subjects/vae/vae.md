# Variational Autoencoders (VAE)'s

## TL:DR

A combination of two neural networks: The **encoder** which models latent variables based on input data $P(z|x)$ and the **decoder** which then models desired output based on these latent variables $P(x|z)$.

The latent variables also has a prior which you can control $P(z)$. Training is done by maximizing the $evidence lower-bound$ (ELBO). Which is a combination of the mode's ability to create meaningful data from the latent variables and the model's ability to accurately reflect the latent variable prior given the data (to prevent overfitting to the data)

## Pseudocode
<!-- Find way to write math in pseudocode, for now we must make python pseudocode -->
```python
def VAE_train(batch_size):
    p_z = prior()
    p_x_z = decoder()
    p_z_x = encoder()
    

    # Find log probability of data given latent var # (reconstruction term)

    for x in dataset:
        mu, sigma = encoder(x) # Get params for p(z|x)
        noise = Normal(0,1).sample(batch_size) # Sample from standard normal
        zs = mu + noise * sigma # Get z samples using reparameterization trick
        p_x_zs = decoder(zs)
        log_p_x_zs = p_x_zs.log_prob(x) # Eval decoder distribution data x 

    # Calculate KL-div. between agg. post. and prior (regularization term)
    
    # With Gaussian prior, KL has closed-form
    if prior is gaussian:
        KL = kl(Normal(mu, sigma), p_z)

    # Otherwise, use MC-estimate
    else:
        # Choose K as large enough to get accurate estimate of KL-div
        K = K
        noise_kl = Normal(0,1).sample(K)
        zs_kl = mu + noise_kl * sigma
        log_p_z_x_kl = p_z.log_prob(zs_kl)
        log_p_z = p_z.log_prob(zs_kl)

        KL = mean(log_p_z_x_kl - log_p_z)
    
    # Return mean difference as loss
    loss = mean(log_p_x_zs - KL)
    loss.backwards()
    ...
```

### Pseudocode notes

The VAE training consists of two parts: One passing the data through the encoder to generate latent variables, and passing these latent variables through the decoder to an output distribution. Then it checks the log proability of the data given this output distribution. This creates our **reconstruction error** which measures how well the VAE can represent the data given latent variables. 

Next, we calculate the KL-divergence between the aggregated posterior (our latent variables given the data) and the prior on the latent variables. This is the **regularization term** and exists partly to ensure the model takes data into account when generating latent variables. 

A side note is that if our prior on latent variables is gaussian, the KL divergence has a closed-form solution, otherwise we need to approximate it using monte carlo sampling.

## Detailed description