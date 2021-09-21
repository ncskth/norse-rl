import gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import norse.torch as norse
import IPython.display as display
import norse_rl  # Init environment


def draw_network(ax, layer_sizes, weights):
    # Thanks to https://stackoverflow.com/a/67289898/999865
    top = 0.9
    bottom = 0.1
    left = 0.2
    right = 0.9
    layer_sizes = [len(x) for x in layer_sizes]
    v_spacing = 1 / max(layer_sizes)
    h_spacing = 1 / len(layer_sizes)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0

        for m in range(layer_size):
            center = (n * h_spacing + left, layer_top - m * v_spacing)
            radius = v_spacing / 4.0
            circle = plt.Circle(center, radius, color="w", ec="k", zorder=4)
            ax.add_artist(circle)

        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
            layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    weight = weights[n][o][m]
                    width = abs(weight) * 10  # Scale so it looks bigger
                    color = "b" if weight < 0 else "r"
                    line = plt.Line2D(
                        [n * h_spacing + left, (n + 1) * h_spacing + left],
                        [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                        c=color,
                        lw=width,
                    )
                    ax.add_artist(line)


def draw_network_update(ax, spikes):
    is_circle = lambda c: isinstance(c, matplotlib.patches.Circle)
    artists = list(filter(is_circle, ax.artists))
    index = 0
    for layer in spikes:
        for neuron in layer:
            color = "b" if neuron else "w"
            artists[index].set_facecolor(color)
            index += 1


def ask_network(model, observation, state=None):
    observation = torch.tensor(observation, dtype=torch.float32)
    action, state = model(observation, state)
    return action.detach().numpy(), state


def weights_from_network(model):
    weights = []
    children = list(model.children())
    for l in children:
        if isinstance(l, torch.nn.Linear):
            weights.append(l.weight)
    assert len(weights) > 0, "We require at least one linear layer"
    assert (
        len(weights) <= len(children) // 2
    ), "We require at least every second layer to be a linear layer"
    return weights


class Simulation:
    def __init__(self, env):
        self.env = env

    def run(self, model):
        # Initialize environment and network
        observation = self.env.reset()
        state = None

        # Setup spike activity hooks
        activities = []

        def forward_state_hook(mod, inp, out):
            activities.append(out[0].detach())

        try:
            model.remove_forward_state_hooks()
            model.forward_state_hooks.clear()
            model.register_forward_state_hooks(forward_state_hook)
        except:
            pass  # Ignore if model already has registered hooks

        # Initialize plotting
        # Thanks to https://matplotlib.org/stable/tutorials/advanced/blitting.html
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.axis("off")
        ax2.axis("off")
        # Show the plot to start caching
        plt.show(block=False)
        bg = f.canvas.copy_from_bbox(f.bbox)
        # Draw initial background
        img = ax1.imshow(
            self.env.render(mode="rgb_array"), animated=True
        )  # only call this once
        ax1.add_artist(img)
        action, state = ask_network(model, observation, state)
        draw_network(ax2, activities, weights_from_network(model))

        # Loop until environment is done or user quits
        is_done = False
        try:
            while not is_done:
                # f.canvas.restore_region(bg)
                display.clear_output(wait=True)

                activities.clear()
                action, state = ask_network(model, observation, state)
                observation, _, is_done, _ = self.env.step(action)

                # Set visual changes
                img.set_data(self.env.render(mode="rgb_array"))  # just update the data
                draw_network_update(ax2, activities)

                # Render graphics
                f.canvas.blit(f.bbox)
                f.canvas.flush_events()
                display.display(f)
        except KeyboardInterrupt:
            pass
