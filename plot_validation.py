import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from df_dataclass import InputFlatOutput
from df_functions import (
    cal_flat_derivs,
    compute_Attitude,
    compute_AttitudeAcc,
    compute_AttitudeAccIntermediate,
    compute_AttitudeIntermediate,
    compute_AttitudeRate,
    compute_AttitudeRateIntermediate,
    compute_control_input,
    eval_flat_derivs,
)


def run_reconstruction(t):
    m = 1.0
    J = np.eye(3)
    g = 9.81

    # x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # z = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # yaw = np.array([[1], [2], [3], [4]])

    x = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    z = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    yaw = np.array([[0], [0], [0], [0]])

    flat_output = InputFlatOutput(pos=np.column_stack((x, y, z)), psi=yaw)
    flat_derivs = cal_flat_derivs(flat_output)
    evaled = eval_flat_derivs(flat_derivs, t)

    attitude_intermediate = compute_AttitudeIntermediate(evaled, g)
    r_wb = compute_Attitude(attitude_intermediate)

    attitude_rate_intermediate = compute_AttitudeRateIntermediate(
        evaled, attitude_intermediate
    )
    omega_b = compute_AttitudeRate(attitude_rate_intermediate, attitude_intermediate)

    attitude_acc_intermediate = compute_AttitudeAccIntermediate(
        evaled, attitude_intermediate, attitude_rate_intermediate
    )
    omega_b_d = compute_AttitudeAcc(
        attitude_rate_intermediate, attitude_acc_intermediate, r_wb
    )

    control_input = compute_control_input(
        m, J, omega_b, omega_b_d, attitude_intermediate
    )

    return {
        "evaled": evaled,
        "attitude_intermediate": attitude_intermediate,
        "r_wb": r_wb,
        "omega_b": omega_b,
        "omega_b_d": omega_b_d,
        "control_input": control_input,
        "m": m,
        "J": J,
        "g": g,
    }


def plot_vector_series(t, series, title, ylabel, output_path):
    labels = ["x", "y", "z"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, label in enumerate(labels):
        ax.plot(t, series[:, i], label=label)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_yaw_series(t, evaled, output_path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, evaled.psi[:, 0], label="psi")
    ax.plot(t, evaled.psi_dot[:, 0], label="psi_dot")
    ax.plot(t, evaled.psi_dotdot[:, 0], label="psi_dotdot")
    ax.set_title("Yaw Series")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trajectory_with_body_axes(t, pos, r_wb, output_path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="black", linewidth=1.5)

    step = max(len(t) // 20, 1)
    samples = np.arange(0, len(t), step)
    axis_len = 0.05 * np.linalg.norm(np.ptp(pos, axis=0))
    if not np.isfinite(axis_len) or axis_len == 0:
        axis_len = 1.0

    colors = ["tab:red", "tab:green", "tab:blue"]
    for idx in samples:
        origin = pos[idx]
        for axis_idx, color in enumerate(colors):
            vec = r_wb[idx, :, axis_idx] * axis_len
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                vec[0],
                vec[1],
                vec[2],
                color=color,
                length=1.0,
                normalize=False,
                linewidth=0.8,
            )

    ax.set_title("Position Trajectory and Body Axes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rotation_checks(t, r_wb, output_path):
    identity = np.eye(3)
    orth_err = np.linalg.norm(
        r_wb @ np.swapaxes(r_wb, 1, 2) - identity, axis=(1, 2)
    )
    det_r = np.linalg.det(r_wb)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, orth_err)
    axes[0].set_ylabel("||R R^T - I||")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, det_r)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("det(R)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Rotation Matrix Checks")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_control_inputs(t, control_input, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(t, control_input.u1[:, 0], label="u1")
    axes[0].set_ylabel("thrust")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, control_input.u2, label="u2")
    axes[1].plot(t, control_input.u3, label="u3")
    axes[1].plot(t, control_input.u4, label="u4")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("moment")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Control Inputs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def skew(w):
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def project_to_rotation(r):
    u, _, vt = np.linalg.svd(r)
    r_projected = u @ vt
    if np.linalg.det(r_projected) < 0:
        u[:, -1] *= -1
        r_projected = u @ vt
    return r_projected


def interp_series(t_query, t, values):
    if values.ndim == 1:
        return np.interp(t_query, t, values)
    return np.array([np.interp(t_query, t, values[:, i]) for i in range(values.shape[1])])


def pack_state(pos, vel, r_wb, omega_b):
    return np.concatenate((pos, vel, r_wb.reshape(-1), omega_b))


def unpack_state(state):
    pos = state[0:3]
    vel = state[3:6]
    r_wb = state[6:15].reshape(3, 3)
    omega_b = state[15:18]
    return pos, vel, r_wb, omega_b


def quad_dynamics(state, t_query, t, u1, tau_b, m, j, g):
    pos, vel, r_wb, omega_b = unpack_state(state)
    thrust = float(interp_series(t_query, t, u1))
    tau = interp_series(t_query, t, tau_b)

    pos_dot = vel
    vel_dot = (thrust / m) * r_wb[:, 2] - np.array([0.0, 0.0, g])
    r_dot = r_wb @ skew(omega_b)
    j_omega = j @ omega_b
    omega_dot = np.linalg.solve(j, tau - np.cross(omega_b, j_omega))

    return pack_state(pos_dot, vel_dot, r_dot, omega_dot)


def rollout_quadrotor(t, evaled, r_wb, omega_b, control_input, m, j, g):
    tau_b = np.column_stack((control_input.u2, control_input.u3, control_input.u4))
    u1 = control_input.u1[:, 0]

    states = np.zeros((len(t), 18))
    states[0] = pack_state(evaled.pos[0], evaled.vel[0], r_wb[0], omega_b[0])

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        state = states[i]

        k1 = quad_dynamics(state, t[i], t, u1, tau_b, m, j, g)
        k2 = quad_dynamics(state + 0.5 * dt * k1, t[i] + 0.5 * dt, t, u1, tau_b, m, j, g)
        k3 = quad_dynamics(state + 0.5 * dt * k2, t[i] + 0.5 * dt, t, u1, tau_b, m, j, g)
        k4 = quad_dynamics(state + dt * k3, t[i + 1], t, u1, tau_b, m, j, g)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pos, vel, r_next, omega_next = unpack_state(next_state)
        states[i + 1] = pack_state(pos, vel, project_to_rotation(r_next), omega_next)

    pos = states[:, 0:3]
    vel = states[:, 3:6]
    r = states[:, 6:15].reshape(len(t), 3, 3)
    omega = states[:, 15:18]

    return {
        "pos": pos,
        "vel": vel,
        "r_wb": r,
        "omega_b": omega,
    }


def plot_reference_vs_rollout(t, reference, rollout, title, ylabel, output_path):
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for i, label in enumerate(labels):
        axes[0].plot(t, reference[:, i], label=f"ref {label}")
        axes[0].plot(t, rollout[:, i], linestyle="--", label=f"rollout {label}")

    error = rollout - reference
    for i, label in enumerate(labels):
        axes[1].plot(t, error[:, i], label=f"err {label}")

    axes[0].set_title(title)
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].set_xlabel("time")
    axes[1].set_ylabel("rollout - ref")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rollout_trajectory(t, ref_pos, rollout_pos, output_path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], label="reference", color="black")
    ax.plot(
        rollout_pos[:, 0],
        rollout_pos[:, 1],
        rollout_pos[:, 2],
        label="rollout",
        color="tab:orange",
        linestyle="--",
    )
    ax.set_title("Reference vs Dynamics Rollout Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(output_dir, exist_ok=True)

    t = np.linspace(1, 50, 200)
    data = run_reconstruction(t)
    evaled = data["evaled"]
    rollout = rollout_quadrotor(
        t,
        evaled,
        data["r_wb"],
        data["omega_b"],
        data["control_input"],
        data["m"],
        data["J"],
        data["g"],
    )

    plot_trajectory_with_body_axes(
        t,
        evaled.pos,
        data["r_wb"],
        os.path.join(output_dir, "trajectory_body_axes.png"),
    )
    plot_vector_series(
        t, evaled.pos, "Position", "position", os.path.join(output_dir, "position.png")
    )
    plot_vector_series(
        t, evaled.vel, "Velocity", "velocity", os.path.join(output_dir, "velocity.png")
    )
    plot_vector_series(
        t,
        evaled.acc,
        "Acceleration",
        "acceleration",
        os.path.join(output_dir, "acceleration.png"),
    )
    plot_vector_series(
        t, evaled.jerk, "Jerk", "jerk", os.path.join(output_dir, "jerk.png")
    )
    plot_vector_series(
        t, evaled.snap, "Snap", "snap", os.path.join(output_dir, "snap.png")
    )
    plot_yaw_series(t, evaled, os.path.join(output_dir, "yaw.png"))
    plot_vector_series(
        t,
        data["omega_b"],
        "Body Angular Velocity",
        "omega_B",
        os.path.join(output_dir, "omega_B.png"),
    )
    plot_vector_series(
        t,
        data["omega_b_d"],
        "Body Angular Acceleration",
        "omega_B_dot",
        os.path.join(output_dir, "omega_B_dot.png"),
    )
    plot_rotation_checks(
        t, data["r_wb"], os.path.join(output_dir, "rotation_checks.png")
    )
    plot_control_inputs(
        t, data["control_input"], os.path.join(output_dir, "control_inputs.png")
    )
    plot_rollout_trajectory(
        t,
        evaled.pos,
        rollout["pos"],
        os.path.join(output_dir, "rollout_trajectory.png"),
    )
    plot_reference_vs_rollout(
        t,
        evaled.pos,
        rollout["pos"],
        "Position Reference vs Dynamics Rollout",
        "position",
        os.path.join(output_dir, "rollout_position.png"),
    )
    plot_reference_vs_rollout(
        t,
        evaled.vel,
        rollout["vel"],
        "Velocity Reference vs Dynamics Rollout",
        "velocity",
        os.path.join(output_dir, "rollout_velocity.png"),
    )
    plot_reference_vs_rollout(
        t,
        data["omega_b"],
        rollout["omega_b"],
        "Body Angular Velocity Reference vs Dynamics Rollout",
        "omega_B",
        os.path.join(output_dir, "rollout_omega_B.png"),
    )

    print(f"Saved validation plots to: {output_dir}")


if __name__ == "__main__":
    main()
