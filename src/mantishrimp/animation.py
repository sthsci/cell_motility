"""Killer-target animation for SimulationResult and legacy notebook outputs."""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from .results import SimulationResult


def animate(result: SimulationResult, *, plot_margin=0.06, **plot_options):
    """Display a simulation with the styling/export options of ``animate_KT``.

    Geometry comes from ``result.config``. ``frame_step`` and ``traj_steps``
    count recorded snapshots; ``interval`` is playback milliseconds per frame.
    Set ``render_html=False`` outside a notebook. Returns a FuncAnimation.
    Set ``transparent_background=True`` to hide the figure and axes backgrounds.
    """
    if not np.isfinite(plot_margin) or plot_margin < 0:
        raise ValueError("plot_margin must be finite and non-negative")
    config = result.config
    history = result.snapshots.rename(columns={"alive": "alive_status"}).sort_values(
        ["step", "cell_type", "cell_id"]
    )
    killers = history.loc[history.cell_type.eq("killer"), ["x", "y"]].to_numpy()
    targets = history.loc[history.cell_type.eq("target"), ["x", "y"]].to_numpy()
    killers = killers.reshape(-1, config.n_killers, 2)
    targets = targets.reshape(-1, config.n_targets, 2)
    d = config.domain
    mx, my = plot_margin * d.width, plot_margin * d.height
    options = dict(
        killer_radius=config.killer_radius, target_radius=config.target_radius,
        dead_target_radius=config.target_radius,
        box=(d.x_min, d.x_max, d.y_min, d.y_max),
        ax_box=(d.x_min - mx, d.x_max + mx, d.y_min - my, d.y_max + my),
        boundary_condition=d.boundary,
        killer_colour="#92C4E9", live_target_colour="#96C36E",
        dead_target_colour="#EA9A9D", traj_steps=250, traj_linewidth=2,
        traj_alpha=0.45, cir_linewidth=1.5, figsize=(6, 7), fig_dpi=100,
        interval=40, frame_step=10,
    )
    options.update(plot_options)
    return animate_KT(history, killers, targets, **options)


def draw_box(ax, box, **kwargs):
    """Draw rectangle boundaries (dashed box) on ax."""
    xmin, xmax, ymin, ymax = box
    ax.plot([xmin, xmax], [ymin, ymin], **kwargs)
    ax.plot([xmin, xmax], [ymax, ymax], **kwargs)
    ax.plot([xmin, xmin], [ymin, ymax], **kwargs)
    ax.plot([xmax, xmax], [ymin, ymax], **kwargs)

def periodic_traj_segment(traj, box):
    """
    For a 2D trajectory (N,2), returns a list of line segments
    that do NOT cross the periodic boundary visually.
    """
    xmin, xmax, ymin, ymax = box
    dx = xmax - xmin
    dy = ymax - ymin
    pts = np.asarray(traj)
    if len(pts) < 2:
        return [pts]
    segments = []
    seg = [pts[0]]
    for i in range(1, len(pts)):
        prev = pts[i-1]
        curr = pts[i]
        # If a jump is large (crossed box), break the segment
        if np.abs(curr[0]-prev[0]) > 0.5*dx or np.abs(curr[1]-prev[1]) > 0.5*dy:
            segments.append(np.array(seg))
            seg = [curr]
        else:
            seg.append(curr)
    if len(seg):
        segments.append(np.array(seg))
    return segments

def animate_KT(
    cell_history_df,
    killer_positions,
    target_positions,
    killer_radius=10.0,
    target_radius=12.0,
    dead_target_radius=4.0,
    *,
    killer_colour="navy",
    live_target_colour="green",
    dead_target_colour="salmon",
    traj_steps=10,
    traj_linewidth=0.7,
    cir_linewidth= 2,
    traj_alpha=0.7,
    figsize=(5, 5),
    box=(-600, 600, -600, 600),
    interval=10,
    boundary_condition='confined',
    ax_box=None,
    add_swatch_legend=True, legend_labels=None, fig_dpi=None, save_gif_path=None,
    frame_step=1, gif_fps=None, render_html=True, transparent_background=False
):
    """
    Animate Killer-Target cell simulation, showing box and periodic/reflecting trajectories.
    box: simulation box [xmin, xmax, ymin, ymax]
    ax_box: axis/plot box [xmin, xmax, ymin, ymax] (optional, can be bigger than sim box)
    traj_linewidth: trajectory width in points (non-negative)
    traj_alpha: trajectory opacity from 0 (transparent) to 1 (opaque)
    dead_target_radius: displayed dead-cell radius; does not change simulation mechanics
    add_swatch_legend: show a circle legend below the plot (legacy argument name)
    frame_step/traj_steps: stride/trail length in recorded frames; the last frame is included
    transparent_background: hide figure and axes backgrounds, keeping cells and labels visible
    """
    if not np.isfinite(traj_linewidth) or traj_linewidth < 0:
        raise ValueError("traj_linewidth must be finite and non-negative")
    if not np.isfinite(traj_alpha) or not 0 <= traj_alpha <= 1:
        raise ValueError("traj_alpha must be finite and between 0 and 1")
    radii = np.asarray([killer_radius, target_radius, dead_target_radius], dtype=float)
    if not np.isfinite(radii).all() or (radii < 0).any() or radii.max() == 0:
        raise ValueError("cell radii must be finite and non-negative, with at least one positive radius")
    if ax_box is None:
        ax_box = box

    recorded_steps = np.sort(cell_history_df['step'].unique())
    steps = len(recorded_steps)
    if steps == 0:
        raise ValueError("simulation history must contain at least one recorded frame")
    frame_step = max(1, int(frame_step))
    interval = max(1, int(round(interval)))
    N_KILLER = (cell_history_df['cell_type'] == 'killer').sum() // steps
    N_TARGET = (cell_history_df['cell_type'] == 'target').sum() // steps

    killer_traj = np.array(killer_positions)
    target_traj = np.array(target_positions)
    if (killer_traj.shape != (steps, N_KILLER, 2)
            or target_traj.shape != (steps, N_TARGET, 2)):
        raise ValueError("trajectories must align with recorded frames and cell counts")

    fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi, layout="constrained")
    if transparent_background:
        # Animation.save can force opaque colours; hidden patches stay transparent.
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
    ax.set_xlim(ax_box[0], ax_box[1])
    ax.set_ylim(ax_box[2], ax_box[3])
    ax.set_aspect('equal')

    # --- Draw dashed simulation box boundary ---
    draw_box(ax, box, color='gray', linestyle='dashed', linewidth=1)
    
    # Circle legend: colours, outlines and relative radii match the cells.
    if add_swatch_legend:
        if legend_labels is None:
            legend_labels = ['killer', 'alive target', 'dead target']
        if len(legend_labels) != 3:
            raise ValueError("legend_labels must contain killer, live-target and dead-target labels")
        handles = [
            Line2D([], [], marker='o', linestyle='none',
                   markersize=12 * radius / radii.max(),
                   markerfacecolor=colour, markeredgecolor='black',
                   markeredgewidth=cir_linewidth, alpha=1)
            for radius, colour in zip(radii, (killer_colour, live_target_colour, dead_target_colour))
        ]
        fig.legend(
            handles, legend_labels, loc='outside lower center', ncol=3,
            frameon=False, handlelength=1, handletextpad=0.5, columnspacing=1,
        )

    # --- Cell circles ---
    killer_circles = [Circle((0,0), radius=killer_radius, color=killer_colour, ec='black', lw=cir_linewidth, alpha=1) for _ in range(N_KILLER)]
    target_circles = [Circle((0,0), radius=target_radius, color=live_target_colour, ec='black', lw=cir_linewidth, alpha=1) for _ in range(N_TARGET)]
    dead_target_circles = [Circle((0,0), radius=dead_target_radius, color=dead_target_colour, ec='black', lw=cir_linewidth, alpha=1, visible=False) for _ in range(N_TARGET)]
    for c in killer_circles + target_circles + dead_target_circles:
        c.set_zorder(3)  # Keep cell fills above trajectory lines (default zorder 2).
        ax.add_patch(c)

    # --- Trajectory lines (only last N steps) ---
    killer_lines = [ [] for _ in range(N_KILLER) ]
    target_lines = [ [] for _ in range(N_TARGET) ]
    for i in range(N_KILLER):
        l, = ax.plot([], [], color=killer_colour, lw=traj_linewidth, alpha=traj_alpha)
        killer_lines[i].append(l)
    for j in range(N_TARGET):
        l, = ax.plot([], [], color=live_target_colour, lw=traj_linewidth, alpha=traj_alpha)
        target_lines[j].append(l)

    def animate(frame):
        step = recorded_steps[frame]
        traj_start = max(0, frame - traj_steps + 1)
        # Remove previous lines (only for periodic, to allow segments)
        if boundary_condition == "periodic":
            for group in killer_lines + target_lines:
                for l in group[1:]:
                    l.remove()
                group[:] = group[:1]

        # Trajectories
        for i in range(N_KILLER):
            xy = killer_traj[traj_start:frame+1, i, :]
            if boundary_condition == "periodic":
                for seg in periodic_traj_segment(xy, box):
                    line, = ax.plot(seg[:, 0], seg[:, 1], color=killer_colour, lw=traj_linewidth, alpha=traj_alpha)
                    killer_lines[i].append(line)
                killer_lines[i][0].set_data([], [])
            else:
                killer_lines[i][0].set_data(xy[:, 0], xy[:, 1])
        for j in range(N_TARGET):
            xy = target_traj[traj_start:frame+1, j, :]
            if boundary_condition == "periodic":
                for seg in periodic_traj_segment(xy, box):
                    line, = ax.plot(seg[:, 0], seg[:, 1], color=live_target_colour, lw=traj_linewidth, alpha=traj_alpha)
                    target_lines[j].append(line)
                target_lines[j][0].set_data([], [])
            else:
                target_lines[j][0].set_data(xy[:, 0], xy[:, 1])

        # Circles (alive/dead)
        this_df = cell_history_df[cell_history_df['step'] == step]
        for i, row in this_df[this_df['cell_type']=='killer'].iterrows():
            killer_circles[int(row['cell_id'])].center = (row['x'], row['y'])
        for j, row in this_df[this_df['cell_type']=='target'].iterrows():
            idx = int(row['cell_id'])
            if row['alive_status']:
                target_circles[idx].center = (row['x'], row['y'])
                target_circles[idx].set_visible(True)
                dead_target_circles[idx].set_visible(False)
            else:
                dead_target_circles[idx].center = (row['x'], row['y'])
                dead_target_circles[idx].set_visible(True)
                target_circles[idx].set_visible(False)
        ax.set_title(f"Step {step}  |  Time {this_df['time'].iloc[0]:.2f}")
        return (sum(killer_lines, []) + sum(target_lines, []) +
                killer_circles + target_circles + dead_target_circles)

    frames = list(range(0, steps, frame_step))
    if frames[-1] != steps - 1:
        frames.append(steps - 1)
    ani = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=True
    )

    if save_gif_path is not None:
        try:
            from matplotlib.animation import PillowWriter
            save_gif_path = Path(save_gif_path).expanduser()
            save_gif_path.parent.mkdir(parents=True, exist_ok=True)
            fps = int(gif_fps) if gif_fps is not None else max(1, int(round(1000 / interval)))
            ani.save(
                save_gif_path,
                writer=PillowWriter(fps=fps),
                # savefig_kwargs={"transparent": True, "facecolor": "none"},
            )
            print(f"Animation saved as GIF: {save_gif_path}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")
            plt.close(fig)
            raise

    plt.close(fig)
    if render_html:
        from IPython.display import HTML, display

        display(HTML(ani.to_jshtml()))
    return ani
