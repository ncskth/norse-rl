# Neuromorphic Robotics Olympics

In this workshop we are going to use *spiking neural networks* to control 2 simulated physical systems:
a cartpole and a Braitenberg vehicle "mouse".
Everything will be run online in [Jupyter Notebooks](https://jupyter.org/), so not installation or setup is required.

<p align="center">
<img src="https://github.com/ncskth/norse-rl/raw/master/book/images/EnvAgentBrain.png"/>
</p>

## Spiking neural networks and control problems

Spiking neural networks are nonlinear systems that **respond to input by integrating signals over time**. 
Put differently, they react to signals with some *delay*.
As an example, the [leaky integrate-and-fire](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) (LIF) neuron model integrates some incoming current until a threshold is reached where it fires a signal (`1`). Until that point in time, the neuron was silent (`0`).

:::{figure-md} markdown-fig
<img src="images/spikes.gif" alt="no-current" class="bg-primary mb-1" >

Three examples of how the LIF neuron model responds to three different, but constant, input currents: 0.0, 0.1, and 0.3. At 0.3, we see that the neuron fires a series of spikes, followed by a membrane "reset".
Note that the neuron parameters are non-biological and that the memebrane voltage threshold is 1.
:::

For control problems, this is actually useful. 
We can interpret this firing/non-firing as a means to turn on and off a signal. On when the neuron is firing (`1`). Off when the neuron is silent (`0`).

In this workshop, we will be exploiting these nonlinear neurondynamics to *intelligently* process sensory signals to achieve specific outcomes.

```{note}
More information about how to understand and operate spiking neural networks can be found in [Getting started with neural controls](getting-started).
```


## Tasks

We will be working with two different settings, resulting in three different neurorobotics tasks.
Your job in this workshop will be to 

|     Task      |    Description     |      Link      |
| ------------- | ------------------ | -------------- | 
| 1. **Cartpole**      |  Balance a wobbly cartpole | [**Click to start the experiment** ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncskth/norse-rl/HEAD?filepath=book%2Fenv_cartpole.ipynb) |
| 2. **Mice and cheese** |  Help a mouse find cheese | [**Click to start the experiment** ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncskth/norse-rl/HEAD?filepath=book%2Fenv_grid.ipynb) |
| 3. **Mice and mazes** |  Help a mouse find cheese, with obstacles! | [**Click to start the experiment** ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncskth/norse-rl/HEAD?filepath=book%2Fenv_grid.ipynb) |
| 4. ? |  To be disclosed | - |

# About Us

Located in Stockholm and affiliated to Kungliga Tekniska Hogskolan (KTH), we are the Neurocomputing Systems Lab.

<p align="center">
<img src="https://github.com/ncskth/norse-rl/raw/master/book/images/ncs.png" width="500px"/>
</p>

