import math
import torch
pi = math.pi

def abeles_magnetic_nonspinflip(
        q: torch.Tensor,
        thickness: torch.Tensor,
        roughness: torch.Tensor,
        sld: torch.Tensor,
        sld_magnetic: torch.Tensor,
        **kwargs,
):
    """Simulates reflectivity curves for non spin flip polarized neutron reflectometry using the Abeles matrix method (independently for each spin). 

    Args:
        q (Tensor): q values
        thickness (Tensor): layer thicknesses
        roughness (Tensor): interlayer roughnesses
        sld (Tensor): nuclear SLDs of the layers
        sld_magnetic (Tensor): magnetic SLDs of the layers

    Returns:
        Tuple[Tensor, Tensor]: simulated reflectivity curves for spin-up and spin-down
    """
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    sld = torch.cat([torch.zeros(batch_size, 1).to(sld), sld], -1)[:, None]
    sld_magnetic = torch.cat([torch.zeros(batch_size, 1).to(sld_magnetic), sld_magnetic], -1)[:, None]
    thickness = torch.cat([torch.zeros(batch_size, 1).to(thickness), thickness], -1)[:, None]
    roughness = roughness[:, None] ** 2

    sld = sld * 1e-6 + 1e-30j
    sld_magnetic = sld_magnetic * 1e-6

    k_z0 = (q / 2).to(c_dtype)

    if k_z0.dim() == 1:
        k_z0 = k_z0.unsqueeze(0)

    if k_z0.dim() == 2:
        k_z0 = k_z0.unsqueeze(-1)

    k_n_up = torch.sqrt(k_z0 ** 2 - 4 * math.pi * (sld + sld_magnetic))
    k_n_down = torch.sqrt(k_z0 ** 2 - 4 * math.pi * (sld - sld_magnetic))

    k_n_up, k_np1_up = k_n_up[..., :-1], k_n_up[..., 1:]
    k_n_down, k_np1_down = k_n_down[..., :-1], k_n_down[..., 1:]

    beta_up = 1j * thickness * k_n_up
    beta_down = 1j * thickness * k_n_down

    exp_beta_up = torch.exp(beta_up)
    exp_m_beta_up = torch.exp(-beta_up)

    exp_beta_down = torch.exp(beta_down)
    exp_m_beta_down = torch.exp(-beta_down)

    rn_up = (k_n_up - k_np1_up) / (k_n_up + k_np1_up) * torch.exp(-2 * k_n_up * k_np1_up * roughness)
    rn_down = (k_n_down - k_np1_down) / (k_n_down + k_np1_down) * torch.exp(-2 * k_n_down * k_np1_down * roughness)

    c_matrices_up = torch.stack([
        torch.stack([exp_beta_up, rn_up * exp_m_beta_up], -1),
        torch.stack([rn_up * exp_beta_up, exp_m_beta_up], -1),
    ], -1)

    c_matrices_down = torch.stack([
        torch.stack([exp_beta_down, rn_down * exp_m_beta_down], -1),
        torch.stack([rn_down * exp_beta_down, exp_m_beta_down], -1),
    ], -1)

    c_matrices_up = [c.squeeze(-3) for c in c_matrices_up.split(1, -3)]
    c_matrices_down = [c.squeeze(-3) for c in c_matrices_down.split(1, -3)]

    m_up, c_matrices_up = c_matrices_up[0], c_matrices_up[1:]
    m_down, c_matrices_down = c_matrices_down[0], c_matrices_down[1:]

    m_up_00 = m_up[..., 0, 0]
    m_up_01 = m_up[..., 0, 1]
    m_up_10 = m_up[..., 1, 0]
    m_up_11 = m_up[..., 1, 1]

    m_down_00 = m_down[..., 0, 0]
    m_down_01 = m_down[..., 0, 1]
    m_down_10 = m_down[..., 1, 0]
    m_down_11 = m_down[..., 1, 1]

    for c_up, c_down in zip(c_matrices_up, c_matrices_down):
      c_up_00 = c_up[..., 0, 0]
      c_up_01 = c_up[..., 0, 1]
      c_up_10 = c_up[..., 1, 0]
      c_up_11 = c_up[..., 1, 1]

      c_down_00 = c_down[..., 0, 0]
      c_down_01 = c_down[..., 0, 1]
      c_down_10 = c_down[..., 1, 0]
      c_down_11 = c_down[..., 1, 1]

      m_up_00_new = m_up_00 * c_up_00 + m_up_01 * c_up_10
      m_up_01_new = m_up_00 * c_up_01 + m_up_01 * c_up_11
      m_up_10_new = m_up_10 * c_up_00 + m_up_11 * c_up_10
      m_up_11_new = m_up_10 * c_up_01 + m_up_11 * c_up_11
      m_up_00, m_up_01, m_up_10, m_up_11 = m_up_00_new, m_up_01_new, m_up_10_new, m_up_11_new
    
      m_down_00_new = m_down_00 * c_down_00 + m_down_01 * c_down_10
      m_down_01_new = m_down_00 * c_down_01 + m_down_01 * c_down_11
      m_down_10_new = m_down_10 * c_down_00 + m_down_11 * c_down_10
      m_down_11_new = m_down_10 * c_down_01 + m_down_11 * c_down_11
      m_down_00, m_down_01, m_down_10, m_down_11 = m_down_00_new, m_down_01_new, m_down_10_new, m_down_11_new
      
    r_up = (m_up_10 / m_up_00).abs() ** 2
    r_down = (m_down_10 / m_down_00).abs() ** 2

    r_up = torch.clamp_max_(r_up, 1.)
    r_down = torch.clamp_max_(r_down, 1.)

    return torch.stack([r_up, r_down], dim=1)


def abeles_magnetic_spinflip(
        q: torch.Tensor,
        thickness: torch.Tensor,
        roughness: torch.Tensor, ###roughness just as an approximation
        sld: torch.Tensor,
        sld_magnetic: torch.Tensor,
        magnetization_angle: torch.Tensor,
        **kwargs,
):
    """
    """

    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    sld = torch.cat([torch.zeros(batch_size, 1).to(sld), sld], -1)[:, None]
    sld_magnetic = torch.cat([torch.zeros(batch_size, 1).to(sld_magnetic), sld_magnetic], -1)[:, None]
    thickness = torch.cat([torch.zeros(batch_size, 1).to(thickness), thickness], -1)[:, None]
    roughness = roughness[:, None] ** 2

    magnetization_angle = torch.cat([torch.zeros(batch_size, 1).to(magnetization_angle), magnetization_angle], -1)
    half_theta_n = (magnetization_angle[..., 1:] - magnetization_angle[..., :-1])[:, None] / 2 
    half_theta_n = torch.deg2rad(half_theta_n)

    sld = sld * 1e-6 + 1e-30j
    sld_magnetic = sld_magnetic * 1e-6

    k_z0 = (q / 2).to(c_dtype)

    if k_z0.dim() == 1:
        k_z0 = k_z0.unsqueeze(0)

    if k_z0.dim() == 2:
        k_z0 = k_z0.unsqueeze(-1)

    k_n_up = torch.sqrt(k_z0 ** 2 - 4 * math.pi * (sld + sld_magnetic))
    k_n_down = torch.sqrt(k_z0 ** 2 - 4 * math.pi * (sld - sld_magnetic))

    k_n_up, k_np1_up = k_n_up[..., :-1], k_n_up[..., 1:]
    k_n_down, k_np1_down = k_n_down[..., :-1], k_n_down[..., 1:]

    beta_up = 1j * thickness * k_n_up
    beta_down = 1j * thickness * k_n_down

    exp_beta_up = torch.exp(beta_up)
    exp_m_beta_up = torch.exp(-beta_up)

    exp_beta_down = torch.exp(beta_down)
    exp_m_beta_down = torch.exp(-beta_down)

    rn_up_up = (k_n_up - k_np1_up) / (k_n_up + k_np1_up) * torch.exp(-2 * k_n_up * k_np1_up * roughness)
    rn_up_down = (k_n_up - k_np1_down) / (k_n_up + k_np1_down) * torch.exp(-2 * k_n_up * k_np1_down * roughness)
    rn_down_up = (k_n_down - k_np1_up) / (k_n_down + k_np1_up) * torch.exp(-2 * k_n_down * k_np1_up * roughness)
    rn_down_down = (k_n_down - k_np1_down) / (k_n_down + k_np1_down) * torch.exp(-2 * k_n_down * k_np1_down * roughness)

    fn_up_up = torch.cos(half_theta_n) * (k_n_up + k_np1_up) / (2 * k_n_up)
    fn_up_down = torch.sin(half_theta_n) * (k_n_up + k_np1_down) / (2 * k_n_up)
    fn_down_up = -torch.sin(half_theta_n) * (k_n_down + k_np1_up) / (2 * k_n_down)
    fn_down_down = torch.cos(half_theta_n) * (k_n_down + k_np1_down) / (2 * k_n_down)

    c_matrices = torch.stack([
        torch.stack([fn_up_up * exp_beta_up, fn_up_up * rn_up_up * exp_m_beta_up, fn_down_up * exp_beta_down, fn_down_up * rn_down_up * exp_m_beta_down], -1),
        torch.stack([fn_up_up * rn_up_up * exp_beta_up, fn_up_up * exp_m_beta_up, fn_down_up * rn_down_up * exp_beta_down, fn_down_up * exp_m_beta_down], -1),
        torch.stack([fn_up_down * exp_beta_up, fn_up_down * rn_up_down * exp_m_beta_up, fn_down_down * exp_beta_down, fn_down_down * rn_down_down * exp_m_beta_down], -1),
        torch.stack([fn_up_down * rn_up_down * exp_beta_up, fn_up_down * exp_m_beta_up, fn_down_down * rn_down_down * exp_beta_down, fn_down_down * exp_m_beta_down], -1),
    ], -1)

    c_matrices = [c.squeeze(-3) for c in c_matrices.split(1, -3)]

    m, c_matrices = c_matrices[0], c_matrices[1:]
    m_00 = m[..., 0, 0]
    m_01 = m[..., 0, 1]
    m_02 = m[..., 0, 2]
    m_03 = m[..., 0, 3]
    m_10 = m[..., 1, 0]
    m_11 = m[..., 1, 1]
    m_12 = m[..., 1, 2]
    m_13 = m[..., 1, 3]
    m_20 = m[..., 2, 0]
    m_21 = m[..., 2, 1]
    m_22 = m[..., 2, 2]
    m_23 = m[..., 2, 3]
    m_30 = m[..., 3, 0]
    m_31 = m[..., 3, 1]
    m_32 = m[..., 3, 2]
    m_33 = m[..., 3, 3]

    for c in c_matrices:
        c_00 = c[..., 0, 0]
        c_01 = c[..., 0, 1]
        c_02 = c[..., 0, 2]
        c_03 = c[..., 0, 3]
        c_10 = c[..., 1, 0]
        c_11 = c[..., 1, 1]
        c_12 = c[..., 1, 2]
        c_13 = c[..., 1, 3]
        c_20 = c[..., 2, 0]
        c_21 = c[..., 2, 1]
        c_22 = c[..., 2, 2]
        c_23 = c[..., 2, 3]
        c_30 = c[..., 3, 0]
        c_31 = c[..., 3, 1]
        c_32 = c[..., 3, 2]
        c_33 = c[..., 3, 3]

        m_00_new = m_00 * c_00 + m_01 * c_10 + m_02 * c_20 + m_03 * c_30
        m_01_new = m_00 * c_01 + m_01 * c_11 + m_02 * c_21 + m_03 * c_31
        m_02_new = m_00 * c_02 + m_01 * c_12 + m_02 * c_22 + m_03 * c_32
        m_03_new = m_00 * c_03 + m_01 * c_13 + m_02 * c_23 + m_03 * c_33

        m_10_new = m_10 * c_00 + m_11 * c_10 + m_12 * c_20 + m_13 * c_30
        m_11_new = m_10 * c_01 + m_11 * c_11 + m_12 * c_21 + m_13 * c_31
        m_12_new = m_10 * c_02 + m_11 * c_12 + m_12 * c_22 + m_13 * c_32
        m_13_new = m_10 * c_03 + m_11 * c_13 + m_12 * c_23 + m_13 * c_33

        m_20_new = m_20 * c_00 + m_21 * c_10 + m_22 * c_20 + m_23 * c_30
        m_21_new = m_20 * c_01 + m_21 * c_11 + m_22 * c_21 + m_23 * c_31
        m_22_new = m_20 * c_02 + m_21 * c_12 + m_22 * c_22 + m_23 * c_32
        m_23_new = m_20 * c_03 + m_21 * c_13 + m_22 * c_23 + m_23 * c_33

        m_30_new = m_30 * c_00 + m_31 * c_10 + m_32 * c_20 + m_33 * c_30
        m_31_new = m_30 * c_01 + m_31 * c_11 + m_32 * c_21 + m_33 * c_31
        m_32_new = m_30 * c_02 + m_31 * c_12 + m_32 * c_22 + m_33 * c_32
        m_33_new = m_30 * c_03 + m_31 * c_13 + m_32 * c_23 + m_33 * c_33
        

        m_00, m_01, m_02, m_03, m_10, m_11, m_12, m_13, m_20, m_21, m_22, m_23, m_30, m_31, m_32, m_33  = (
            m_00_new, m_01_new, m_02_new, m_03_new, m_10_new, m_11_new, m_12_new, m_13_new, m_20_new, m_21_new, m_22_new, m_23_new, m_30_new, m_31_new, m_32_new, m_33_new
        )

 
    r_up_up = ((m_10 * m_22 - m_12 * m_20) / (m_00 * m_22 - m_02 * m_20)).abs() ** 2
    r_up_down = ((m_30 * m_22 - m_32 * m_20) / (m_00 * m_22 - m_02 * m_20)).abs() ** 2
    r_down_up = ((m_12 * m_00 - m_10 * m_02) / (m_00 * m_22 - m_02 * m_20)).abs() ** 2
    r_down_down = ((m_32 * m_00 - m_30 * m_02) / (m_00 * m_22 - m_02 * m_20)).abs() ** 2

    return torch.stack([r_up_up, r_up_down, r_down_up, r_down_down], dim=1)