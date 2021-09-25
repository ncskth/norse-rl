import matplotlib
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
from ipycanvas import Canvas, hold_canvas
import time
import torch
import IPython.display as display
from ipywidgets import Image
import math


def draw_network(
    canvas,
    states,
    weights,
    input_labels=[],
    input_values=[],
    output_labels=[],
    output_values=[],
    height=400,
    width=500,
    offsetx=400,
):
    # Thanks to https://stackoverflow.com/a/67289898/999865
    top = 0.99
    bottom = 0.0
    left = 0.29
    right = 0.78
    layer_sizes = [len(x.v) for x in states]
    v_spacing = 1 / max(layer_sizes)
    h_spacing = 1 / (len(layer_sizes) + 1.5)

    # Draw input labels
    font_size = 16
    line_height = font_size + 2
    canvas.fill_style = "black"
    canvas.text_align = "center"
    layer_top = v_spacing * (len(input_labels) - 1) / 2.0 + (top + bottom) / 2.0
    for i, label in enumerate(input_labels):
        x = offsetx + width * 0.15
        y = height - height * layer_top + i * height * v_spacing
        canvas.font = f"{font_size - 2}px sans-serif"
        canvas.fill_text(f"IN: {input_values[i]:.2f}", x, y - line_height)
        canvas.font = f"{font_size}px sans-serif"
        for i, line in enumerate(label.split("\n")):
            canvas.fill_text(line, x, y + i * line_height)

    # Draw output labels
    canvas.text_align = "left"
    layer_top = v_spacing * (len(output_labels) - 1) / 2.0 + (top + bottom) / 2.0
    for i, label in enumerate(output_labels):
        x = offsetx + width * 0.86
        y = height - height * layer_top + i * height * v_spacing
        canvas.font = f"{font_size - 2}px sans-serif"
        canvas.fill_text(f"OUT: {output_values[i]:.2f}", x, y - line_height)
        canvas.font = f"{font_size}px sans-serif"
        for i, line in enumerate(label.split("\n")):
            canvas.fill_text(line, x, y + i * line_height)

    # Draw edges
    x_coo = torch.linspace(left, right, len(layer_sizes))
    for n, (layer_size_a, layer_size_b) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                weight = weights[n][o][m]
                weight_magnitude = max(
                    0.001, min(8, abs(weight.item())) * 5
                )  # Scale so it looks bigger
                v = 1.0 if weight.item() >= 1 else (weight.clip(-1, 1).item() + 1) / 2
                color = cm.coolwarm(v, bytes=True)
                x1 = offsetx + width * x_coo[n].item()
                x2 = offsetx + width * x_coo[n + 1].item()
                y1 = height - height * layer_top_a + height * m * v_spacing - 5
                y2 = height - height * layer_top_b + height * o * v_spacing - 5
                canvas.stroke_style = f"rgb({color[0]}, {color[1]}, {color[2]})"
                canvas.line_width = weight_magnitude
                canvas.stroke_line(x1, y1, x2, y2)

    # Draw Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_bot = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        state_n = states[n]
        for m in range(layer_size):
            cx = offsetx + width * x_coo[n].item()
            cy = height - height * layer_bot + height * m * v_spacing - 5
            radius = (v_spacing + h_spacing) / 12.0 * width
            # Draw background
            # Ensure voltage is displayed as spike when i >= 1
            v, i = state_n.v[m], state_n.i[m]
            v = 1.0 if i.item() >= 1 else (v.clip(-1, 1).item() + 1) / 2
            color = cm.coolwarm(v, bytes=True)
            canvas.fill_style = f"rgb({color[0]},{color[1]},{color[2]})"
            canvas.fill_circle(cx, cy, radius)
            # Draw stroke
            canvas.line_width = 1
            canvas.stroke_style = "black"
            canvas.stroke_circle(cx, cy, radius)


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
    def __init__(self, env, show_fps: bool = False, fps_cap: int = 50):
        self.env = env
        self.show_fps = show_fps
        self.fps_sleep = 1 / fps_cap

    def run(self, model):
        # Initialize environment and network
        observation = self.env.reset()
        state = None
        iterations = 0

        try:
            max_it = 3000 * (1 + self.env.level * 2)
        except:
            max_it = 0

        canvas = Canvas(width=900, height=400)
        display.display(canvas)
        canvas.font = "12px serif"

        # Draw initial network
        action, state = ask_network(model, observation, state)
        in_labels = self.env.observation_labels
        out_labels = self.env.action_labels
        fps_text = ""

        # Loop until environment is done or user quits
        is_done = False
        frames = 0
        start_time = time.time()
        frame_start = time.time()
        try:
            while not is_done:
                with hold_canvas(canvas):
                    canvas.clear()

                    frame_diff = time.time() - frame_start
                    frame_start = time.time()

                    # Run and draw environment
                    network_time = time.time()
                    action, state = ask_network(model, observation, state)
                    old_obs = observation
                    activities = [x for x in state if x is not None]
                    network_time = time.time() - network_time
                    env_time = time.time()
                    observation, _, is_done, _ = self.env.step(action)
                    env_time = time.time() - env_time

                    # Draw network
                    draw_network(
                        canvas,
                        activities,
                        weights_from_network(model),
                        in_labels,
                        old_obs,
                        out_labels,
                        action,
                    )

                    # Set visual changes
                    visual_time = time.time()
                    self.env.render(canvas)
                    visual_time = time.time() - visual_time

                    # Update fps and redraw every second
                    frames += 1
                    if self.show_fps and (time.time() - start_time) > 1:
                        fps_text = f"{frames / (time.time() - start_time):.1f}fps - {frame_diff:.2f}frame {network_time:.2f}net {env_time:.2f}env {visual_time:.2f}vis"
                        frames = 0
                        start_time = time.time()

                    canvas.fill_style = "white"
                    canvas.fill_text(fps_text, 10, 20)
                time.sleep(max(0, self.fps_sleep - (time.time() - frame_start)))

                iterations += 1
                if max_it > 0 and iterations > max_it:
                    break

        except KeyboardInterrupt:
            pass

        try:
            score = self.env.food_items - len(self.env.food) * max_it / iterations
            canvas.fill_style = "rgb(50, 50, 50)"
            canvas.fill_rect(20, 160, 360, 80)
            canvas.font = "80px Courier New bolder"
            canvas.fill_style = "red"
            canvas.fill_text("Score: " + str(max(0, round(score))), 50, 225)
        except:
            score = 0
        finally:
            print("End of Task ... ")
