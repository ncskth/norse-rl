# Neuromorphic Robotics Olympics

In this workshop we are going to use *spiking neural networks* to control 2 simulated physical systems:
a cartpole and a Braitenberg vehicle "mouse".
Everything will be run online in [Jupyter Notebooks](https://jupyter.org/), so not installation or setup is required.



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

````{margin}
```{note}
More information about how to understand and operate spiking neural networks can be found in [Getting started with neural controls](getting-started).
```
````

In this workshop, we will be exploiting these nonlinear neurondynamics to *intelligently* process sensory signals to achieve specific outcomes.

## Tasks

We will be working with two different settings, resulting in three different tasks you can solve.

### 1. Cartpole

A cartpole is a pendulum with a center of gravity above its pivot point. It’s unstable, but can be controlled by moving the pivot point under the center of mass. The goal is to keep the cartpole balanced by applying appropriate forces to a pivot point.

<p align="center">
<img src="https://github.com/ncskth/norse-rl/raw/master/book/images/CartPole.png"/>
</p>

## 2. Braitenberg vehicle

A Braitenberg vehicle is an agent that can autonomously move around based on its sensor inputs. It has primitive sensors that measure some stimulus at a point, and actuators or effectors. In the simplest configuration, a sensor is directly connected to an effector, so that a sensed signal immediately produces a movement of the wheel.

Depending on how sensors and actuators are connected, the vehicle exhibits different behaviors (which can be goal-oriented). 

The connections between sensors and actuators for the simplest vehicles can be ipsilateral or contralateral, and excitatory or inhibitory, producing four combinations with different behaviours named fear, aggression, liking, and love. These correspond to biological positive and negative stimuli present in many animals species.

<p align="center">
<img src="https://github.com/ncskth/norse-rl/raw/master/book/images/EnvAgentBrain.png"/>
</p>


### Maze (Task 3)


# About Us

Located in Stockholm and affiliated to Kungliga Tekniska Hogskolan (KTH), we are the Neurocomputing Systems Lab.

<p align="center">
<img src="https://github.com/ncskth/norse-rl/raw/master/book/images/ncs.png" width="500px"/>
</p>