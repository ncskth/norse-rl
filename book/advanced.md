(page-advanced)=
# Advanced topics

## Neuromorphic hardware

```{image} images/spinnaker.jpg
:alt: SpiNNaker
:width: 300px
:align: center
```

[Neuromorphic hardware](https://en.wikipedia.org/wiki/Neuromorphic_engineering) is an exciting new type of hardware that makes it possible to run spiking neural networks really, really fast.
They consume dramatically less energy, which makes them ideal for closed-loop neurorobotic systems.

Read more on the [Human Brain Project's website](https://www.humanbrainproject.eu/en/silicon-brains/).

## Deep learning with spiking neural networks

```{image} https://raw.githubusercontent.com/norse/norse/master/logo.png
:alt: Norse
:width: 500px
:align: center
```

The neuron simulator we used in this workshop is called [Norse](https://github.com/norse/norse) and is [one of many](https://github.com/norse/norse/#4-similar-work) simulators for neuron dynamics *similar* to biology.

Norse is a library for **deep learning with spiking neural networks** which, roughly, means that they provide biologically realistic neurons that can be optimized with *both* biologically plausible learning rules *and* deep learning optimizations.

Many more examples are available in their [documentation](https://norse.github.io/norse) and their [example notebooks](https://github.com/norse/notebooks).

## Event-based sensing

```{image} images/dvs.gif
:alt: Pushbot
:width: 300px
:align: center
```

This recording from the [IBM gesture dataset](https://research.ibm.com/interactive/dvsgesture/) neatly illustrates event-based sensing: a different type of vision from the frame-based cameras most people are used to.
Similar to the human sensory system, the event-based cameras used on/off events (spikes) to describe stimuli.
The signal is, as you can see, much different from RGB cameras, but it is much sparser and more energy efficient, which makes it ideal for spiking neural networks.

See this [curated list of event-based resources](https://github.com/uzh-rpg/event-based_vision_resources) for more information.