# Diffusion models

## Todo
- Add Notes about what courses and material this is taken from

"Creating data from noise is generative modelling"
- Yang Song et. al

## TL:DR

State-of-the-art generative method. Iteratively adds noise to images while training (akin to a markov process). Then learns reverse process to generate images from random noise.

### Algorithm Pseudocode

Goes here, find out how to add pseudocode to markdown

## Detailed description

Diffusion models, also called Denoising Diffusion Probabilistic Models (DDPM), works by iteratively adding gaussian noise to images. This is done in the forward process. The noise added is given by

$$
z_t = \sqrt{1 -\beta_t}  z_{t-1} + \sqrt{\beta_t}\mathbf{\epsilon} ~~~~ \text{where} ~~~~ \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{1})
$$

Here, $\beta_1, \dots, \beta_T \in (0,1)$ is called the noise schedule. And is determined beforehand depending on choices. The reason for having $\sqrt{1 - \beta_t}$ and $\sqrt{\beta_t}$ is to ensure that the mean gets closer to 0 and the variance gets closer to 1 the futher along our noise scheduler we get. \\

It should be fairly easy to see, that this just results in a gaussian with mean $\sqrt{1 - \beta_t}z_{t-1}$ and variance $\sqrt{\beta_t}^2 = \beta_t$:

$$
p(z_t|z_{t-1}) = \mathcal{N}(z_t|\sqrt{1 - \beta_t}z_{t-1}, \beta_t \mathbf{I})
$$

An example of the forward process can look like this:

<!-- Image showing forward process goes here, remember to add credit -->

On top of more noise being added each step, a larger amount of noise is added each subsequent step from t = 0. The reason for this is due to the aforementioned noise schedule. This is simply a list of $\beta_0, \beta_1, \dots, \beta_T$ with $T$ being a hyperparameter denoting the maximum amount of noise we can add. Some papers use a linear noise scheduler, meaning $\beta_t$ goes linearly from some minimum to some maximum value. One such example could be $\beta_0 = 10^{-4}, \beta_T = 0.02$. Some other papers use a cosine scheduler, given below.

$$
% Missing write it here, although I have some issues witwh beta_0 given from alpha
$$

