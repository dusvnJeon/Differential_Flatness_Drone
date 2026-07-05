import numpy as np
from df_dataclass import InputFlatOutput
from df_functions import (
    cal_flat_derivs,
    eval_flat_derivs,
    compute_AttitudeIntermediate,
    compute_Attitude,
    compute_AttitudeRateIntermediate,
    compute_AttitudeRate,
    compute_AttitudeAccIntermediate,
    compute_AttitudeAcc,
    compute_control_input,
)

# Objective : validate differential flatness concept with numerical example
# convention : each column contrains the polynomial coefficients of a single axis
    # But Evaluated values are 2D matrix (N,3)
        # N : len of time series, 3 : each axis
# Only matrix

m = 1
J = np.eye(3)
g = 9.81

t = np.linspace(1,50,200)

# x = np.array([1,2,3,4,5,6,7,8])
# y = np.array([1,2,3,4,5,6,7,8])
# z = np.array([1,2,3,4,5,6,7,8])
# yaw = np.array([[1],[2],[3],[4]])

x = np.array([0, 0, 1, 0, 0, 0, 0, 0])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0])
z = np.array([0, 0, 0, 0, 0, 0, 0, 0])
yaw = np.array([[0], [0], [0], [0]])


flat_output = InputFlatOutput(pos = np.column_stack((x, y, z)), psi=yaw)
flat_derivs = cal_flat_derivs(flat_output)
evaled_flat_derivs = eval_flat_derivs(flat_derivs, t)
attitude_intermediate = compute_AttitudeIntermediate(evaled_flat_derivs,g)
R_WB = compute_Attitude(attitude_intermediate)
attitude_rate_intermediate = compute_AttitudeRateIntermediate(evaled_flat_derivs, attitude_intermediate)
omega_B = compute_AttitudeRate(attitude_rate_intermediate, attitude_intermediate)

attitude_acc_intermediate = compute_AttitudeAccIntermediate(evaled_flat_derivs, attitude_intermediate, attitude_rate_intermediate)
omega_B_d = compute_AttitudeAcc(attitude_rate_intermediate, attitude_acc_intermediate, R_WB)

control_input = compute_control_input(m, J, omega_B, omega_B_d, attitude_intermediate)


print(R_WB@np.swapaxes(R_WB, 1, 2))
print(np.shape(flat_derivs.psi))
print(np.shape(evaled_flat_derivs.psi))
print(omega_B)
print(omega_B_d)
print(control_input)
print(R_WB)