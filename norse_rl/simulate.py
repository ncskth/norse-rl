import matplotlib
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import time

import torch

import IPython.display as display


def draw_network(ax, activities, weights, input_labels=[], output_labels=[]):
    # Thanks to https://stackoverflow.com/a/67289898/999865
    top = 0.99
    bottom = 0.0
    left = 0.22
    right = 0.84
    layer_sizes = [len(x.v) for x in activities]
    v_spacing = 1 / max(layer_sizes)
    h_spacing = 1 / (len(layer_sizes) + 1.5)

    # Draw input labels
    layer_top = v_spacing * (len(input_labels) - 1) / 2.0 + (top + bottom) / 2.0
    for i, label in enumerate(input_labels):
        text = plt.Text(
            0.1,
            layer_top - i * v_spacing - 0.01,
            label + "\n\n  ➤",
            zorder=5,
            horizontalalignment="center",
            fontsize=16,
        )
        ax.add_artist(text)

    # Draw output labels
    for i, label in enumerate(output_labels):
        text = plt.Text(
            0.93,
            layer_top - i * v_spacing - 0.01,
            label + "\n\n ➤",
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
            radius = (v_spacing + h_spacing) / 12.0
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
                    width = abs(weight) * 5  # Scale so it looks bigger
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


def draw_fps(ax, fps):
    text = plt.Text(
        0.98,
        0.96,
        f"{fps}fps",
        zorder=5,
        horizontalalignment="right",
        fontsize=12,
    )
    ax.add_artist(text)
    return text


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
    def __init__(self, env, show_fps: bool = False):
        self.env = env
        self.show_fps = show_fps

    def run(self, model):
        # Initialize environment and network
        observation = self.env.reset()
        state = None

        # Initialize plotting
        # Thanks to https://matplotlib.org/stable/tutorials/advanced/blitting.html
        f = plt.figure(tight_layout=True, figsize=(10, 4))
        # ax1, ax2 = f.subplots(1, 2)
        g = gridspec.GridSpec(1, 6)
        ax1 = f.add_subplot(g[0, :2])
        ax2 = f.add_subplot(g[0, 2:])
        ax1.axis("off")
        ax2.axis("off")
        plt.show(block=False)  # Show the plot to start caching

        # Draw initial environment
        img = ax1.imshow(self.env.render(mode="rgb_array"), animated=True)
        ax1.add_artist(img)

        # Draw initial network
        action, state = ask_network(model, observation, state)
        in_labels = self.env.observation_labels
        out_labels = self.env.action_labels
        activities = [x for x in state if x is not None]
        draw_network(
            ax2, activities, weights_from_network(model), in_labels, out_labels
        )

        # Draw fps
        if self.show_fps:
            fps_artist = draw_fps(ax2, "0")

        # Loop until environment is done or user quits
        is_done = False
        frames = 0
        start_time = time.time()
        frame_start = time.time()
        try:
            while not is_done:
                frame_diff = time.time() - frame_start
                frame_start = time.time()
                #display.clear_output(wait=True)

                network_time = time.time()
                action, state = ask_network(model, observation, state)
                activities = [x for x in state if x is not None]
                network_time = time.time() - network_time
                env_time = time.time()
                observation, _, is_done, _ = self.env.step(action)
                env_time = time.time() - env_time

                # Set visual changes
                visual_time = time.time()
                img.set_data(self.env.render(mode="rgb_array"))  # just update the data
                draw_network_update(ax2, activities)
                visual_time = time.time() - visual_time

                # Update fps and redraw every second
                frames += 1
                if self.show_fps and (time.time() - start_time) > 1:
                    fps_artist.set_text(
                        f"{frames / (time.time() - start_time):.1f}fps - {frame_diff:.2f}frame {network_time:.2f}net {env_time:.2f}env {visual_time:.2f}vis"
                    )
                    frames = 0
                    start_time = time.time()

                # Render graphics
                f.canvas.blit(f.bbox)
                f.canvas.flush_events()
                display.display(f, clear=True)
        except KeyboardInterrupt:
            pass
