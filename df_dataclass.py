import numpy as np
from dataclasses import dataclass

@dataclass
class InputFlatOutput():
    pos: np.ndarray
    psi: np.ndarray


@dataclass
class FlatDerivs():
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    jerk: np.ndarray
    snap: np.ndarray

    psi: np.ndarray
    psi_dot: np.ndarray
    psi_dotdot: np.ndarray


@dataclass
class EvalFlatDerivs:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    jerk: np.ndarray
    snap: np.ndarray

    psi: np.ndarray
    psi_dot: np.ndarray
    psi_dotdot: np.ndarray


@dataclass
class AttitudeIntermediate():
    f_W: np.ndarray
    zB_W: np.ndarray
    xc_W: np.ndarray
    yB_W: np.ndarray
    xB_W: np.ndarray



@dataclass
class AttitudeRateIntermediate():
    f_d_W: np.ndarray
    zB_d_W: np.ndarray
    xc_d_W: np.ndarray
    yB_d_W: np.ndarray
    xB_d_W: np.ndarray
    k: np.ndarray
    k_d: np.ndarray


@dataclass
class AttitudeAccIntermediate():
    zB_dd_W: np.ndarray
    yB_dd_W: np.ndarray
    xB_dd_W: np.ndarray


@dataclass
class ControlInput():
    u1: np.ndarray  # Net Force to +zB
    u2: np.ndarray  # xB rotation input
    u3: np.ndarray  # yB rotation input
    u4: np.ndarray  # zB rotation input