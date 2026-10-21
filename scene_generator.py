
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = (128,128)
#RESOLUTION = 0.5

PTL_PARAMS = {
    "table":  {"sigma": 7.0, "amplitude": 35.0},
    "chair":  {"sigma": 4.0, "amplitude": 20.0},
    "person": {"sigma": 9.0, "amplitude": 55.0}
}

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import logsumexp
def smooth_potential(attractive, repulsive, tau=5.0):
    fields = np.stack([attractive, repulsive], axis=0)

    return tau * logsumexp(fields / tau, axis=0)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import heapq
import numpy as np

import numpy as np
from scipy.interpolate import make_interp_spline

def trajectory_is_collision_free(trajectory, grid):
    coords = np.rint(trajectory).astype(int)

    inside = (
        (coords[:, 0] >= 0) &
        (coords[:, 0] < grid.shape[0]) &
        (coords[:, 1] >= 0) &
        (coords[:, 1] < grid.shape[1])
    )

    if not np.all(inside):
        return False

    return np.all(np.isfinite(
        grid[coords[:, 0], coords[:, 1]]
    ))

def smooth_path(path, num_control_points=15, num_samples=300):
    """
    Interpolate an A* path with a cubic B-spline.

    Parameters
    ----------
    path : array-like
        A* waypoints of shape (N, 2).
    num_control_points : int
        Number of path waypoints selected for interpolation.
    num_samples : int
        Number of points in the resulting smooth trajectory.

    Returns
    -------
    trajectory : np.ndarray
        Smooth trajectory of shape (num_samples, 2).
    control_points : np.ndarray
        Selected interpolation points.
    """
    path = np.asarray(path, dtype=float)

    if len(path) < 2:
        return path.copy(), path.copy()

    # Select evenly spaced points along the path
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc_length = np.concatenate(([0], np.cumsum(distances)))

    n = min(num_control_points, len(path))
    sample_distances = np.linspace(0, arc_length[-1], n)

    control_points = np.column_stack([
        np.interp(sample_distances, arc_length, path[:, i])
        for i in range(2)
    ])

    # Cubic spline if possible, lower degree otherwise
    degree = min(3, n - 1)

    spline = make_interp_spline(
        sample_distances,
        control_points,
        k=degree,
        axis=0
    )

    t = np.linspace(0, arc_length[-1], num_samples)
    trajectory = spline(t)

    return trajectory, control_points

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def plot_path(
    potential,
    path,
    robot_pos,
    goal_pos,
    obstacle_positions,
    trajectory,
    control_points
):
    """
    Interactively edit a cubic B-spline trajectory by dragging
    interpolation points.

    Close the plot window to return the original and modified
    trajectories and their interpolation points.
    """

    original_cp = np.asarray(control_points, dtype=float).copy()
    modified_cp = original_cp.copy()

    original_traj = np.asarray(trajectory, dtype=float).copy()
    modified_traj = original_traj.copy()

    # Fixed spline parameterization
    distances = np.linalg.norm(np.diff(original_cp, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(distances)))
    t /= t[-1]

    t_dense = np.linspace(0, 1, len(original_traj))

    def evaluate_spline(points):
        spline = make_interp_spline(
            t, points,
            k=min(3, len(points) - 1),
            axis=0
        )
        return spline(t_dense)

    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 9))

    im = ax.imshow(
        potential.T,
        origin="lower",
        cmap="viridis"
    )
    fig.colorbar(im, ax=ax, label="Potential")

    # Original A* path
    if path is not None:
        path = np.asarray(path)
        ax.plot(
            path[:, 0], path[:, 1],
            "--", color="white", alpha=0.4,
            linewidth=1.5, label="A* Path"
        )

    # Obstacles
    styles = {
        "table":  {"color": "red", "marker": "s"},
        "chair":  {"color": "orange", "marker": "^"},
        "person": {"color": "cyan", "marker": "o"}
    }

    for obs_type, positions in obstacle_positions.items():
        style = styles.get(
            obs_type, {"color": "gray", "marker": "o"}
        )

        ax.scatter(
            positions[:, 0], positions[:, 1],
            color=style["color"],
            marker=style["marker"],
            s=90, edgecolors="black",
            label=obs_type.capitalize(),
            zorder=5
        )

        for i, (x, y) in enumerate(positions):
            ax.annotate(
                f"{obs_type}_{i}",
                (x, y),
                xytext=(5, 7),
                textcoords="offset points",
                color="white",
                fontsize=8
            )

    # Original trajectory (fixed)
    ax.plot(
        original_traj[:, 0],
        original_traj[:, 1],
        color="white",
        linewidth=2.5,
        label="Original B-Spline",
        zorder=6
    )

    # Editable trajectory
    edited_line, = ax.plot(
        modified_traj[:, 0],
        modified_traj[:, 1],
        color="magenta",
        linewidth=3,
        label="Modified B-Spline",
        zorder=7
    )

    # Original control points
    ax.scatter(
        original_cp[:, 0],
        original_cp[:, 1],
        color="white",
        marker="x",
        s=40,
        label="Original Control Points",
        zorder=8
    )

    # Editable control points
    editable_scatter = ax.scatter(
        modified_cp[:, 0],
        modified_cp[:, 1],
        color="yellow",
        edgecolors="black",
        s=90,
        label="Editable Control Points",
        zorder=10
    )

    # Robot and goal
    robot = np.asarray(robot_pos).reshape(2)
    goal = np.asarray(goal_pos).reshape(2)

    ax.scatter(
        *robot, color="lime", marker="*",
        s=200, edgecolors="black",
        label="Robot", zorder=11
    )

    ax.scatter(
        *goal, color="red", marker="X",
        s=180, edgecolors="black",
        label="Goal", zorder=11
    )

    # Dragging state
    state = {"selected": None}

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return

        # Find nearest control point in screen pixels
        screen_points = ax.transData.transform(modified_cp)
        mouse = np.array([event.x, event.y])

        distances = np.linalg.norm(
            screen_points - mouse, axis=1
        )

        idx = np.argmin(distances)

        # Select only if cursor is close enough
        if distances[idx] < 12:
            state["selected"] = idx

    def on_motion(event):
        nonlocal modified_traj

        if state["selected"] is None:
            return

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        idx = state["selected"]

        # Keep first and last points fixed
        if idx == 0 or idx == len(modified_cp) - 1:
            return

        # Move selected control point
        modified_cp[idx] = [
            np.clip(event.xdata, 0, potential.shape[0] - 1),
            np.clip(event.ydata, 0, potential.shape[1] - 1)
        ]

        # Recompute cubic B-spline
        modified_traj = evaluate_spline(modified_cp)

        # Update plot
        edited_line.set_data(
            modified_traj[:, 0],
            modified_traj[:, 1]
        )

        editable_scatter.set_offsets(modified_cp)

        fig.canvas.draw_idle()

    def on_release(event):
        state["selected"] = None

    # Register mouse events
    fig.canvas.mpl_connect(
        "button_press_event", on_press
    )
    fig.canvas.mpl_connect(
        "motion_notify_event", on_motion
    )
    fig.canvas.mpl_connect(
        "button_release_event", on_release
    )

    ax.set_title("Drag Yellow Control Points to Edit Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.15, 1)
    )

    plt.tight_layout()
    plt.show(block=True)

    return (
        original_traj,
        modified_traj.copy(),
        original_cp,
        modified_cp.copy()
    )

def astar(grid, start, goal, potential, weight=1.0):
    """
    Find a minimum-cost path through a potential field using A*.

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid, with np.inf indicating obstacles.
    start, goal : array-like
        Start and goal coordinates (x, y).
    potential : np.ndarray
        Nonnegative repulsive traversal cost at each grid cell.
    weight : float
        Relative importance of the potential cost.

    Returns
    -------
    list[tuple[int, int]] | None
        Path from start to goal, or None if no path exists.
    """
    start = tuple(np.asarray(start).astype(int).reshape(2))
    goal = tuple(np.asarray(goal).astype(int).reshape(2))

    H, W = grid.shape

    if not (0 <= start[0] < H and 0 <= start[1] < W):
        raise ValueError("Start is outside the grid")
    if not (0 <= goal[0] < H and 0 <= goal[1] < W):
        raise ValueError("Goal is outside the grid")

    if not np.isfinite(grid[start]) or not np.isfinite(grid[goal]):
        return None

    if np.any(potential < 0) or not np.all(np.isfinite(potential)):
        raise ValueError("Potential must contain finite, nonnegative costs")

    # 8-connected neighbors
    directions = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    def heuristic(a, b):
        # Euclidean distance is admissible
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_set = [(heuristic(start, goal), start)]

    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            return path[::-1]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            x, y = neighbor

            if not (0 <= x < H and 0 <= y < W):
                continue

            if not np.isfinite(grid[neighbor]):
                continue

            # Prevent diagonal corner-cutting
            if dx != 0 and dy != 0:
                if not np.isfinite(grid[current[0] + dx, current[1]]):
                    continue
                if not np.isfinite(grid[current[0], current[1] + dy]):
                    continue

            movement_cost = np.hypot(dx, dy)

            # Travel distance plus potential penalty
            step_cost = movement_cost * (
                1.0 + weight * potential[neighbor]
            )

            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, np.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(neighbor, goal)

                heapq.heappush(open_set, (f_score, neighbor))

    return None

def plot_potential(potential, obstacle_positions, robot_pos=None, goal_pos=None):
  X, Y = np.indices(potential.shape)

  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(111, projection="3d")

  # Plot surface
  surf = ax.plot_surface(
      X, Y, potential,
      cmap="viridis",
      edgecolor="none",
      alpha=0.9
  )

  fig.colorbar(surf, ax=ax, shrink=0.6, label="Potential")

  # Marker/style choices for each obstacle type
  style_map = {
      "table":  {"marker": "s", "color": "red"},
      "chair":  {"marker": "^", "color": "orange"},
      "person": {"marker": "o", "color": "blue"},
  }

  legend_handles = []

  # Plot obstacles
  for obs_type, positions in obstacle_positions.items():
      positions_int = positions.astype(int)

      xs = positions_int[:, 0]
      ys = positions_int[:, 1]

      # Get the z value of the surface at each obstacle location
      zs = potential[xs, ys] + 2.0   # small lift so markers sit above surface

      style = style_map.get(obs_type, {"marker": "x", "color": "black"})

      ax.scatter(
          xs, ys, zs,
          marker=style["marker"],
          color=style["color"],
          s=80,
          label=obs_type
      )

      # Label each obstacle point
      for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
          ax.text(
              x, y, z + 1.0,
              f"{obs_type}_{i}",
              color=style["color"],
              fontsize=9
          )

      # Add one legend handle per obstacle type
      legend_handles.append(
          Line2D(
              [0], [0],
              marker=style["marker"],
              color="w",
              markerfacecolor=style["color"],
              markersize=10,
              label=obs_type
          )
      )

  # Plot robot if provided
  if robot_pos is not None:
      robot = np.asarray(robot_pos).reshape(2).astype(int)
      rz = potential[robot[0], robot[1]] + 2.0

      ax.scatter(robot[0], robot[1], rz, color="green", marker="*", s=150)
      ax.text(robot[0], robot[1], rz + 1.0, "robot", color="green", fontsize=10)

      legend_handles.append(
          Line2D(
              [0], [0],
              marker="*",
              color="w",
              markerfacecolor="green",
              markersize=12,
              label="robot"
          )
      )

  # Plot goal if provided
  if goal_pos is not None:
      goal = np.asarray(goal_pos).reshape(2).astype(int)
      gz = potential[goal[0], goal[1]] + 2.0

      ax.scatter(goal[0], goal[1], gz, color="magenta", marker="X", s=120)
      ax.text(goal[0], goal[1], gz + 1.0, "goal", color="magenta", fontsize=10)

      legend_handles.append(
          Line2D(
              [0], [0],
              marker="X",
              color="w",
              markerfacecolor="magenta",
              markersize=12,
              label="goal"
          )
      )

  ax.legend(handles=legend_handles, loc="upper right")

  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Potential")
  ax.set_title("3D Artificial Potential Field")

  plt.show()

def gen_obs():
  """
    Generate random obstacle positions for each obstacle type.

    Randomly samples 1–3 (x, y) coordinates for each obstacle
    type defined in PTL_PARAMS within the grid boundaries.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping obstacle types to arrays of shape (N, 2),
        where N is randomly sampled from {1, 2, 3}.
  """ 

  positions = []
  placements = {}
  for k, _ in PTL_PARAMS.items():
    num_samples = np.random.randint(1,4)
    curr_obs = np.random.uniform(low=[20, 20], high=[GRID_SIZE[0]-28, GRID_SIZE[1]-28], size = (num_samples, 2))
    placements[k] = curr_obs
    positions.append(curr_obs)

  return placements, np.vstack(positions)

def repl_fld(grid, obstacles):

  X, Y = np.indices(GRID_SIZE)
  potential = np.zeros_like(grid, dtype=np.float32)
  
  #For key v
  for obs_type, v in obstacles.items():
    sigma = PTL_PARAMS[obs_type]["sigma"]
    amplitude = PTL_PARAMS[obs_type]["amplitude"]
    
    x = v[:, 0, None, None] 
    y = v[:, 1, None, None] 

    dist_sq = (X[None, :,:] - x)**2 + (Y[None, :, :] - y)**2
    potential += amplitude * np.exp(-dist_sq / (2 * sigma**2)).sum(axis=0)

  return potential

def attr_fld(grid,goal, k_attr=0.01):
  X, Y = np.indices(grid.shape)
  goal = np.asarray(goal).reshape(2)
  dist_sq = (X - goal[0])**2 + (Y - goal[1])**2

  potential = 0.5 * k_attr * dist_sq

  return potential.astype(np.float32)
def gen_map():
  grid = np.zeros(GRID_SIZE, dtype=np.float32)

  obstacle_positions, pos = gen_obs()

  robot_pos = np.random.uniform(
      low=[0, 0], high=[16, 16], size=2
  )

  goal_pos = np.random.uniform(
      low=[110, 110], high=[128, 128], size=2
  )

  pos = pos.astype(int)

  obstacle_mask = np.zeros(GRID_SIZE, dtype=bool)
  obstacle_mask[pos[:, 0], pos[:, 1]] = True

  grid[obstacle_mask] = np.inf

  repulsive = repl_fld(grid, obstacle_positions)
  attractive = attr_fld(grid, goal_pos)

  potential = attractive + repulsive

  # Plan using repulsive potential
  path = astar(
    grid,
    robot_pos,
    goal_pos,
    repulsive,
    weight=0.1
  )

  if path is not None:
      trajectory, control_points = smooth_path(
          path,
          num_control_points=15,
          num_samples=300
      )

     # (
     #     original_trajectory,
     #     user_trajectory,
     #     original_control_points,
     #     user_control_points
     # ) = plot_path(
     #     potential,
     #     path,
     #     robot_pos,
     #     goal_pos,
     #     obstacle_positions,
     #     trajectory,
     #     control_points
     # )  
  return grid, potential, path

import numpy as np
from scipy.ndimage import distance_transform_edt

GRID_SIZE = (128, 128)

CLASS_IDS = {
    "table": 1,
    "chair": 2,
    "person": 3
}


def gen_sample(debug=False):
    """
    Generate a random navigation scene and an A* B-spline demonstration.

    Returns
    -------
    dict
        Scene features and reference trajectory for diffusion pretraining.
    """

    # Generate obstacle positions
    obstacle_positions, pos = gen_obs()

    robot_pos = np.random.uniform(
        low=[0, 0], high=[16, 16], size=2
    )

    goal_pos = np.random.uniform(
        low=[110, 110], high=[128, 128], size=2
    )

    # 1. Occupancy grid
    occupancy_grid = np.zeros(GRID_SIZE, dtype=np.uint8)

    # 2. Semantic map
    semantic_map = np.zeros(GRID_SIZE, dtype=np.uint8)

    # 3. Obstacle metadata
    obstacle_metadata = []

    for obs_type, positions in obstacle_positions.items():
        coords = positions.astype(int)

        occupancy_grid[coords[:, 0], coords[:, 1]] = 1
        semantic_map[coords[:, 0], coords[:, 1]] = CLASS_IDS[obs_type]

        for position in positions:
            obstacle_metadata.append({
                "position": position.copy(),
                "orientation": 0.0,
                "class": obs_type,
                "class_id": CLASS_IDS[obs_type],
                "dimensions": [1.0, 1.0]
            })

    # 4. Euclidean signed distance field
    obstacle_mask = occupancy_grid.astype(bool)

    outside = distance_transform_edt(~obstacle_mask)
    inside = distance_transform_edt(obstacle_mask)

    esdf = (outside - inside).astype(np.float32)

    # 5. Hard obstacle grid for A*
    grid = np.zeros(GRID_SIZE, dtype=np.float32)
    grid[obstacle_mask] = np.inf

    # 6. Repulsive potential field
    repulsive_potential = repl_fld(
        grid,
        obstacle_positions
    )
    # Generate potential fields
    repulsive = repl_fld(grid, obstacle_positions)
    attractive = attr_fld(grid, goal_pos)

    # Combined potential field
    potential = attractive + repulsive

    # Run A* using the combined potential
    path = astar(
        grid,
        robot_pos,
        goal_pos,
        potential,
        weight=0.1
    )

    if path is None:
        return None

    # Generate cubic B-spline trajectory
    original_trajectory, original_control_points = smooth_path(
        path,
        num_control_points=15,
        num_samples=300
    ) 
    
    if debug:

      plot_path(
            potential,
            path,
            robot_pos,
            goal_pos,
            obstacle_positions,
            original_trajectory,
            original_control_points
        )  

    # 9. Return training sample
    return {
        "occupancy_grid": occupancy_grid,
        "esdf": esdf,
        "semantic_map": semantic_map,
        "obstacle_metadata": obstacle_metadata,
        "robot_pos": robot_pos.astype(np.float32),
        "goal_pos": goal_pos.astype(np.float32),
        "original_trajectory": original_trajectory.astype(np.float32),

        # Optional
        "repulsive_potential": repulsive_potential.astype(np.float32),
        "original_control_points": original_control_points.astype(np.float32)
    }


