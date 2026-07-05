import numpy as np
from df_dataclass import (
    InputFlatOutput,
    FlatDerivs,
    EvalFlatDerivs,
    AttitudeIntermediate,
    AttitudeRateIntermediate,
    AttitudeAccIntermediate,
    ControlInput,
)

def polyno_deriv(polycoeff, order=1):
    l = polycoeff.copy()
    for i in range(order):
        N = np.shape(l)[0]
        power = np.arange(1,N)[:,None]
        l = power * l[1:]
    return l


def cal_flat_derivs(flat_output: InputFlatOutput):
    x = flat_output.pos[:,[0]]
    x_d = polyno_deriv(x)
    x_dd = polyno_deriv(x_d)
    x_ddd = polyno_deriv(x_dd)
    x_dddd = polyno_deriv(x_ddd)

    y = flat_output.pos[:,[1]]
    y_d = polyno_deriv(y)
    y_dd = polyno_deriv(y_d)
    y_ddd = polyno_deriv(y_dd)
    y_dddd = polyno_deriv(y_ddd)

    z = flat_output.pos[:,[2]]
    z_d = polyno_deriv(z)
    z_dd = polyno_deriv(z_d)
    z_ddd = polyno_deriv(z_dd)
    z_dddd = polyno_deriv(z_ddd)

    psi = flat_output.psi
    psi_d = polyno_deriv(psi)
    psi_dd = polyno_deriv(psi_d)

    flat_derivs = FlatDerivs(

        pos = np.column_stack((x, y, z)),
        vel = np.column_stack((x_d, y_d, z_d)),
        acc = np.column_stack((x_dd, y_dd, z_dd)),
        jerk = np.column_stack((x_ddd, y_ddd, z_ddd)),
        snap = np.column_stack((x_dddd, y_dddd, z_dddd)),
        
        psi = psi,
        psi_dot = psi_d,
        psi_dotdot = psi_dd,
    )
    return flat_derivs



def eval_flat_derivs(flat_derivs: FlatDerivs, t: np.ndarray):

    result = EvalFlatDerivs(

    pos = np.polynomial.polynomial.polyval(t, flat_derivs.pos).T,
    vel = np.polynomial.polynomial.polyval(t, flat_derivs.vel).T,
    acc = np.polynomial.polynomial.polyval(t, flat_derivs.acc).T,
    jerk = np.polynomial.polynomial.polyval(t, flat_derivs.jerk).T,
    snap = np.polynomial.polynomial.polyval(t, flat_derivs.snap).T,
    # these are shape of (N,3)
    # where N is len of linspace

    psi = np.polynomial.polynomial.polyval(t, flat_derivs.psi).T,
    psi_dot = np.polynomial.polynomial.polyval(t, flat_derivs.psi_dot).T,
    psi_dotdot = np.polynomial.polynomial.polyval(t, flat_derivs.psi_dotdot).T,
    # these are shape of (N,1)
    # where N is len of linspace
    )

    return result




def compute_AttitudeIntermediate(evaled_flat_derivs: EvalFlatDerivs, g:float):

    acc = evaled_flat_derivs.acc
    psi = evaled_flat_derivs.psi

    gravity = np.zeros_like(acc)
    gravity[:, 2] = g

    f_W =  acc + gravity
    zB_W = f_W / np.linalg.norm(f_W, axis=1, keepdims=True)
    xc_W = np.column_stack((np.cos(psi), np.sin(psi), np.zeros_like(psi)))
    L = np.cross(zB_W, xc_W)
    yB_W = L / np.linalg.norm(L, axis=1, keepdims=True)
    xB_W = np.cross(yB_W, zB_W)

    result = AttitudeIntermediate(
        f_W = f_W,
        zB_W = zB_W,
        xc_W = xc_W,
        yB_W = yB_W,
        xB_W = xB_W
    )

    return result


def compute_Attitude(attitude_intermediate: AttitudeIntermediate):

    xB_W = attitude_intermediate.xB_W
    yB_W = attitude_intermediate.yB_W
    zB_W = attitude_intermediate.zB_W

    R_WB = np.stack((xB_W, yB_W, zB_W), axis=2)    # (N,3,3)

    return R_WB


def compute_AttitudeRateIntermediate(evaled_flat_derivs: EvalFlatDerivs, attitude_intermediate: AttitudeIntermediate):

    f_d_W = evaled_flat_derivs.jerk
    psi = evaled_flat_derivs.psi
    psi_dot = evaled_flat_derivs.psi_dot

    f_W = attitude_intermediate.f_W
    zB_W = attitude_intermediate.zB_W
    xc_W = attitude_intermediate.xc_W
    yB_W = attitude_intermediate.yB_W

    xc_d_W = np.column_stack((-np.sin(psi), np.cos(psi), np.zeros_like(psi)))

    f_norm = np.linalg.norm(f_W, axis=1, keepdims=True)

    dot_zB_With_f_d_W = np.sum(zB_W*f_d_W, axis=1, keepdims=True)
    zB_d_W = ( f_d_W -  zB_W * dot_zB_With_f_d_W ) / f_norm

    k = np.cross(zB_W,xc_W)
    k_d = np.cross(zB_d_W,xc_W) + np.cross(zB_W,psi_dot*xc_d_W)
    k_norm = np.linalg.norm(k, axis=1, keepdims=True)

    dot_yB_With_k_d = np.sum(yB_W*k_d, axis=1, keepdims=True)
    yB_d_W = ( k_d -  yB_W * dot_yB_With_k_d ) / k_norm

    xB_d_W = np.cross(yB_d_W, zB_W) + np.cross(yB_W,zB_d_W)

    result = AttitudeRateIntermediate(
        f_d_W = f_d_W,
        zB_d_W = zB_d_W,
        xc_d_W = xc_d_W,
        yB_d_W = yB_d_W,
        xB_d_W = xB_d_W,
        k = k,
        k_d = k_d,
    )

    return result


def compute_AttitudeRate(attitude_rate_intermediate: AttitudeRateIntermediate, attitude_intermediate: AttitudeIntermediate):

    xB_d_W = attitude_rate_intermediate.xB_d_W
    yB_d_W = attitude_rate_intermediate.yB_d_W
    zB_d_W = attitude_rate_intermediate.zB_d_W

    xB_W = attitude_intermediate.xB_W
    yB_W = attitude_intermediate.yB_W
    zB_W = attitude_intermediate.zB_W

    R_WB = np.stack((xB_W, yB_W, zB_W), axis=2)
    R_dot_WB = np.stack((xB_d_W, yB_d_W, zB_d_W), axis=2)

    R_WB_T = np.swapaxes(R_WB, 1, 2)   # (N, 3, 3)
    omega_hat_B = R_WB_T @ R_dot_WB   # (N, 3, 3)

    omega_B = np.column_stack((
    omega_hat_B[:, 2, 1],
    omega_hat_B[:, 0, 2],
    omega_hat_B[:, 1, 0],
    ))

    return omega_B

def compute_AttitudeAccIntermediate(evaled_flat_derivs: EvalFlatDerivs, attitude_intermediate: AttitudeIntermediate, attitude_rate_intermediate: AttitudeRateIntermediate):

    f_d_W = evaled_flat_derivs.jerk
    f_dd_W = evaled_flat_derivs.snap
    psi_d = evaled_flat_derivs.psi_dot
    psi_dd = evaled_flat_derivs.psi_dotdot

    f_W = attitude_intermediate.f_W
    zB_W = attitude_intermediate.zB_W
    xc_W = attitude_intermediate.xc_W
    yB_W = attitude_intermediate.yB_W

    xc_d_W = attitude_rate_intermediate.xc_d_W
    yB_d_W = attitude_rate_intermediate.yB_d_W
    zB_d_W = attitude_rate_intermediate.zB_d_W
    k = attitude_rate_intermediate.k
    k_d = attitude_rate_intermediate.k_d

    f_norm = np.linalg.norm(f_W, axis=1, keepdims=True)
    k_norm = np.linalg.norm(k, axis=1, keepdims=True)

    zB_dd_W_1 = -2*(np.sum(zB_W*f_d_W, axis=1, keepdims=True) * zB_d_W)
    zB_dd_W_2 = -zB_W*np.sum(zB_d_W*f_d_W, axis=1, keepdims=True)
    zB_dd_W_3 = f_dd_W - zB_W*np.sum(zB_W*f_dd_W, axis=1, keepdims=True)
    zB_dd_W = ( zB_dd_W_1 + zB_dd_W_2 + zB_dd_W_3 ) / f_norm

    k_dd_1 = np.cross(zB_dd_W,xc_W)
    k_dd_2 = 2*np.cross(zB_d_W,psi_d*xc_d_W)
    k_dd_3 = np.cross(zB_W, (psi_dd*xc_d_W - psi_d*psi_d*xc_W))
    k_dd = k_dd_1 + k_dd_2 + k_dd_3

    yB_dd_W_1 = -2*(np.sum(yB_W*k_d, axis=1, keepdims=True) * yB_d_W)
    yB_dd_W_2 = -yB_W*np.sum(yB_d_W*k_d, axis=1, keepdims=True)
    yB_dd_W_3 = k_dd - yB_W*np.sum(yB_W*k_dd, axis=1, keepdims=True)
    yB_dd_W = ( yB_dd_W_1 + yB_dd_W_2 + yB_dd_W_3 ) / k_norm

    xB_dd_W = np.cross(yB_dd_W, zB_W) + 2*np.cross(yB_d_W,zB_d_W)+ np.cross(yB_W,zB_dd_W)

    result = AttitudeAccIntermediate(
        zB_dd_W = zB_dd_W,
        yB_dd_W = yB_dd_W,
        xB_dd_W = xB_dd_W,
    )

    return result

def compute_AttitudeAcc(attitude_rate_intermediate: AttitudeRateIntermediate, attitude_acc_intermediate: AttitudeAccIntermediate, R_WB: np.ndarray):

    xB_d_W = attitude_rate_intermediate.xB_d_W
    yB_d_W = attitude_rate_intermediate.yB_d_W
    zB_d_W = attitude_rate_intermediate.zB_d_W

    xB_dd_W = attitude_acc_intermediate.xB_dd_W
    yB_dd_W = attitude_acc_intermediate.yB_dd_W
    zB_dd_W = attitude_acc_intermediate.zB_dd_W

    R_WB_d = np.stack((xB_d_W, yB_d_W, zB_d_W), axis=2)    # (N,3,3)
    R_WB_dd = np.stack((xB_dd_W, yB_dd_W, zB_dd_W), axis=2)    # (N,3,3)

    omega_hat_B_d = np.swapaxes(R_WB_d,1,2) @ R_WB_d + np.swapaxes(R_WB,1,2) @ R_WB_dd

    omega_B_d = np.column_stack((
    omega_hat_B_d[:, 2, 1],
    omega_hat_B_d[:, 0, 2],
    omega_hat_B_d[:, 1, 0],
    ))

    return omega_B_d

# todo
def compute_control_input(m: np.ndarray, J: np.ndarray, omega_B: np.ndarray, omega_B_d: np.ndarray, attitude_intermediate: AttitudeIntermediate):
    f_W = attitude_intermediate.f_W
    f_norm = np.linalg.norm(f_W, axis=1, keepdims=True)
    
    u1 = m * f_norm

    u_attitude = np.cross(omega_B, omega_B@J.T) + omega_B_d@J.T
        # convention: each column is each axis -> equation of this line is right.
    
    result = ControlInput(
        u1 = u1,
        u2 = u_attitude[:,0],
        u3 = u_attitude[:,1],
        u4 = u_attitude[:,2], 
    )

    return result