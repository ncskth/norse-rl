import matplotlib
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import time
import math

import IPython.display as display


def draw_network(ax, activities, weights, input_labels=[], output_labels=[]):
    # Thanks to https://stackoverflow.com/a/67289898/999865
    top = 0.9
    bottom = 0.1
    left = 0.23
    right = 0.81
    layer_sizes = [len(x) for x in activities]
    v_spacing = 1 / max(layer_sizes)
    h_spacing = 1 / (len(layer_sizes) + 1.5)

    # Draw input labels
    layer_top = v_spacing * (len(input_labels) - 1) / 2.0 + (top + bottom) / 2.0
    for i, label in enumerate(input_labels):
        text = plt.Text(
            0.06,
            layer_top - i * v_spacing - 0.01,
            label + "  ➤",
            zorder=5,
            horizontalalignment="center",
            fontsize=16,
        )
        ax.add_artist(text)

    # Draw output labels
    for i, label in enumerate(output_labels):
        text = plt.Text(
            0.95,
            layer_top - i * v_spacing - 0.01,
            " ➤ " + label,
            zorder=5,
            horizontalalignment="center",
            fontsize=16,
        )
        ax.add_artist(text)

    # Nodes
    x_coo = torch.linspace(left, right, len(layer_sizes))
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0

        for m in range(layer_size):
            center = (x_coo[n], layer_top - m * v_spacing)
            radius = (v_spacing + h_spacing) / 8.0
            circle = plt.Circle(center, radius, ec="k", zorder=4)
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
                    width = abs(weight) * 6  # Scale so it looks bigger
                    color = "b" if weight < 0 else "r"
                    line = plt.Line2D(
                        [x_coo[n], x_coo[n + 1]],
                        [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                        c=color,
                        lw=width,
                    )
                    ax.add_artist(line)


def draw_network_update(ax, states):
    is_circle = lambda c: isinstance(c, matplotlib.patches.Circle)
    artists = list(filter(is_circle, ax.artists))
    index = 0
    for state in states:
        for v, i in zip(state.v, state.i):
            # Ensure voltage is displayed as spike when i >= 1
            v = 1.0 if i >= 1 else (v.clip(-1, 1).item() + 1) / 2
            color = cm.coolwarm(v, bytes=False)
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

def clearEnv(axis):

    axis.cla()
    axis.set(xlim=(0, 500), ylim=(0, 500))
    axis.axes.xaxis.set_visible(False)
    axis.axes.yaxis.set_visible(False)

class Simulation:
    def __init__(self, env):
        self.env = env

    def run(self, model):
        # Initialize environment and network
        observation = self.env.reset()
        state = None

        # Setup spike activity hooks
        # activities = []

        # def forward_state_hook(mod, inp, out):
        #     activities.append(out[0].detach())

        # try:
        #     model.remove_forward_state_hooks()
        #     model.forward_state_hooks.clear()
        #     model.register_forward_state_hooks(forward_state_hook)
        # except:
        #     pass  # Ignore if model already has registered hooks

        # Initialize plotting
        # Thanks to https://matplotlib.org/stable/tutorials/advanced/blitting.html
        f = plt.figure(tight_layout=True, figsize=(18, 8))
        g = gridspec.GridSpec(1, 5)
        ax1 = f.add_subplot(g[0, :2])
        ax2 = f.add_subplot(g[0, 2:])

        ax2.axis("off")
        plt.show(block=False)  # Show the plot to start caching

        # Draw initial environment
        img = ax1.imshow(self.env.render(mode="rgb_array"), animated=True)

        img_mouse = mpimg.imread('norse_rl/images/Mouse_60px.png')
        s_m_2 = int(img_mouse.shape[0]/2)

        ax1.add_artist(img)

        # Draw initial network
        action, state = ask_network(model, observation, state)
        in_labels = self.env.observation_labels
        out_labels = self.env.action_labels
        activities = [x for x in state if x is not None]
        draw_network(
            ax2, activities, weights_from_network(model), in_labels, out_labels
        )

        # Loop until environment is done or user quits
        is_done = False
        a = 0
        try:
            while not is_done:
                display.clear_output(wait=True)

                # activities.clear()
                action, state = ask_network(model, observation, state)
                activities = [x for x in state if x is not None]
                observation, _, is_done, _ = self.env.step(action)

                # Set visual changes
                clearEnv(ax1)
                
                # Draw Environment + Cheese + 'Mouse Tile'
                ax1.imshow(self.env.render(mode="rgb_array"))  # just update the data
                
                # Draw Mouse
                imgRot = ndimage.rotate((img_mouse*255).astype('uint8'), self.env.state[2]*180/math.pi, reshape=False)
                ax1.imshow(imgRot, extent=(self.env.state[0]-s_m_2, self.env.state[0]+s_m_2, self.env.state[1]-s_m_2, self.env.state[1]+s_m_2))
                
                draw_network_update(ax2, activities)

                # Render graphics
                f.canvas.blit(f.bbox)
                f.canvas.flush_events()
                display.display(f)
        except KeyboardInterrupt:
            pass
