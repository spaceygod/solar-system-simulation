import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.widgets import Slider, Button

def visualize_simulation_with_controls(trajectories, radii_trajectory, downsample_factor=10, dynamic_axis_limits=False, save_as_gif=False, gif_filename="simulation.gif"):
    """
    Visualize the N-body simulation interactively with a slider and play controls.
    
    Parameters:
    - trajectories: List of positions of particles at each time step.
    - downsample_factor: Factor to downsample the trajectories for visualization.
    - dynamic_axis_limits: Whether to set axis limits dynamically based on the data.
    """
    # If raddi_trajectory is none just set it to 1 with same shape as trajectories
    if radii_trajectory is None:
        radii_trajectory = [np.ones_like(traj[:, 0]) for traj in trajectories]

    # Downsample trajectories for visualization
    trajectories = trajectories[::downsample_factor]
    radii_trajectory = radii_trajectory[::downsample_factor]

    # Prepare data
    steps = len(trajectories)

    # Check if trajectories are 2D or 3D
    is_3d = trajectories[0].shape[1] == 3

    # Initialize the plot
    fig = plt.figure()
    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter([], [], [], s=[])
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter([], [], s=[])

    # Use initial positions to set fixed axis limits
    initial_positions = trajectories[0]
    x_min, x_max = initial_positions[:, 0].min(), initial_positions[:, 0].max()
    y_min, y_max = initial_positions[:, 1].min(), initial_positions[:, 1].max()
    if is_3d:
        z_min, z_max = initial_positions[:, 2].min(), initial_positions[:, 2].max()

    # Set fixed axis limits with some padding
    ax.set_xlim(x_min*2, x_max*2)
    ax.set_ylim(y_min*2, y_max*2)
    if is_3d:
        ax.set_zlim(z_min*2, z_max*2)
    ax.set_title("Interactive Visualization")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")

    # Update function for animation and slider
    current_frame = [0]

    def update(frame):
        positions = trajectories[frame]
        radii = radii_trajectory[frame]
        sizes = [r**2 * 1000 for r in radii]  # Scale marker size with squared radius

        # Remove NaN or Inf positions
        valid_positions = positions[~np.isnan(positions).any(axis=1) & ~np.isinf(positions).any(axis=1)]

        if valid_positions.size == 0:
            print(f"Frame {frame}: No valid positions after filtering.")
            return

        if dynamic_axis_limits:
            # Calculate 10th and 90th percentiles for dynamic axis limits
            try:
                x_min, x_max = np.percentile(valid_positions[:, 0], [10, 90])
                y_min, y_max = np.percentile(valid_positions[:, 1], [10, 90])
                if is_3d:
                    z_min, z_max = np.percentile(valid_positions[:, 2], [10, 90])

                # Add padding to avoid tight limits
                padding = 0.25 * (x_max - x_min)
                x_min, x_max = x_min - padding, x_max + padding
                y_min, y_max = y_min - padding, y_max + padding
                if is_3d:
                    z_min, z_max = z_min - padding, z_max + padding

                # Update axis limits
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                if is_3d:
                    ax.set_zlim(z_min, z_max)
            except Exception as e:
                print(f"Error setting axis limits at frame {frame}: {e}")
                return

        if is_3d:
            scatter._offsets3d = (
                valid_positions[:, 0],  # x-coordinates
                valid_positions[:, 1],  # y-coordinates
                valid_positions[:, 2],  # z-coordinates
            )
            scatter._sizes = sizes  # Update sizes for 3D plot
        else:
            scatter.set_offsets(valid_positions)
            scatter.set_sizes(sizes)  # Update sizes for 2D plot

        fig.canvas.draw_idle()


    # Create slider for frame selection
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, steps - 1, valinit=0, valstep=1)

    # Update plot based on slider
    def slider_update(val):
        frame = int(slider.val)
        current_frame[0] = frame
        update(frame)

    slider.on_changed(slider_update)

    # Add play/pause button
    ax_button = plt.axes([0.8, 0.9, 0.1, 0.05], facecolor='lightgoldenrodyellow')
    play_button = Button(ax_button, 'Play')

    # Play/pause functionality
    playing = [False]

    def toggle_play(event):
        playing[0] = not playing[0]
        play_button.label.set_text('Pause' if playing[0] else 'Play')

    play_button.on_clicked(toggle_play)

    # Animation loop using Timer
    def update_timer(_):
        if playing[0]:
            current_frame[0] = (current_frame[0] + 1) % steps
            update(current_frame[0])
            slider.set_val(current_frame[0])  # Synchronize slider with animation

    ani = FuncAnimation(fig, update_timer, interval=50)

    # # Save as GIF if requested
    # if save_as_gif:
    #     print(f"Saving animation as {gif_filename}...")
    #     try: 
    #         # Save as GIF
    #         ani.save(gif_filename, writer="pillow", fps=20)
    #         print(f"Animation saved as {gif_filename}")
    #     except Exception as e:
    #         print("Error saving animation as gif:", e)

    plt.show()

def visualize_simulation_matplotlib(trajectories, downsample_factor=10):
    """
    Visualize the N-body simulation using Matplotlib's FuncAnimation.
    
    Parameters:
    - trajectories: List of positions of particles at each time step.
    - downsample_factor: Factor to downsample the trajectories for visualization.
    """
    # Downsample trajectories for visualization
    trajectories = trajectories[::downsample_factor]

    # Prepare data
    steps = len(trajectories)
    n_particles = trajectories[0].shape[0]

    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter([], [], [], s=50)

    # Set axis limits dynamically
    all_positions = np.vstack([traj for traj in trajectories])  # Combine all positions
    ax.set_xlim(all_positions[:, 0].min(), all_positions[:, 0].max())
    ax.set_ylim(all_positions[:, 1].min(), all_positions[:, 1].max())
    ax.set_zlim(all_positions[:, 2].min(), all_positions[:, 2].max())

    def update(frame):
        scatter._offsets3d = (
            trajectories[frame][:, 0],
            trajectories[frame][:, 1],
            trajectories[frame][:, 2],
        )
        return scatter,

    ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    plt.show()


def visualize_simulation_plotly(trajectories, downsample_factor=10):
    """
    Visualize the N-body simulation using Plotly.
    
    Parameters:
    - trajectories: List of positions of particles at each time step.
    """
    # Downsample trajectories for visualization
    trajectories = trajectories[::downsample_factor]
    
    steps = len(trajectories)
    
    # Convert trajectories from CuPy to NumPy
    x_data = [step[:, 0] for step in trajectories]
    y_data = [step[:, 1] for step in trajectories]
    z_data = [step[:, 2] for step in trajectories]

    # Compute dynamic axis ranges
    x_min, x_max = min([x.min() for x in x_data]), max([x.max() for x in x_data])
    y_min, y_max = min([y.min() for y in y_data]), max([y.max() for y in y_data])
    z_min, z_max = min([z.min() for z in z_data]), max([z.max() for z in z_data])

    # Create the initial 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_data[0], y=y_data[0], z=z_data[0],
                mode="markers",
                marker=dict(size=4, opacity=0.8)
            )
        ],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[y_min, y_max]),
                zaxis=dict(range=[z_min, z_max])
            ),
            title="N-body Simulation",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                    ]
                )
            ]
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=x_data[i], y=y_data[i], z=z_data[i],
                        mode="markers",
                        marker=dict(size=4, opacity=0.8)
                    )
                ],
                name=f"frame{i}"
            )
            for i in range(steps)
        ]
    )

    # Add sliders to navigate through frames
    fig.update_layout(
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f"frame{i}"], dict(mode="immediate", frame=dict(duration=50, redraw=True))],
                        label=f"{i}"
                    ) for i in range(steps)
                ],
                transition=dict(duration=0),
                x=0.1,
                len=0.9,
                currentvalue=dict(font=dict(size=15), prefix="Frame: "),
                pad=dict(t=50)
            )
        ]
    )

    # Show the figure
    fig.show()
