# TAC control rule (algorithm sketch)

Each epoch after validation:

1. Compute PB from validation NLL and parameter-displacement complexity C_t.
2. Compute PD from positive risk mismatch V, gradient norm H, and update norm U.
3. Online-standardize PB/PD, then fuse r = alpha * z_PB + (1 - alpha) * z_PD.
4. Smooth rbar_t = beta * rbar_{t-1} + (1 - beta) * r_t.
5. If rbar fails to improve for `p_lr` active epochs: eta <- max(gamma * eta, eta_min).

Training continues for the configured epoch budget. The reported model is the best validation-accuracy checkpoint.
