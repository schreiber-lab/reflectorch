import torch
import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as pe

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional, Tuple


def plot_multilayer_sketch(
    inference_result,
    figsize: Tuple[float, float] = (8.0, 4.5),
    title: Optional[str] = None,
    font_size: int = 12,
    precision: int = 2,

    box_width: float = 10.0,
    layer_height: float = 1.0,
    media_height: float = 0.8,
    gap: float = 0.04,

    layer_colors: Optional[List[str]] = None,
    layer_alpha: float = 0.96,

    ambient_color: str = "#F2F2F2",
    substrate_color: str = "#E3E3E3",

    edgecolor: str = "black",
    layer_linewidth: float = 1.2,
    media_linewidth: float = 1.6,

    protruding_layers: bool = True,
    shadow_alpha: float = 0.30,
    shadow_offset: Tuple[float, float] = (1.5, -1.5),

    color_by_sld: bool = False,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,

    thickness_unit: str = r"$\AA$",
    roughness_unit: str = r"$\AA$",
    sld_unit: str = r"$10^{-6}\ \AA^{-2}$",
    show_roughness: bool = True,
    roughness_symbol: str = r"$\sigma$",
    thickness_symbol: str = r"$d$",
    sld_symbol: str = r"$\rho$",

    ambient_name: str = "Fronting",
    substrate_name: str = "Backing",
):
    params_dict = inference_result.param_model.to_standard_params(
        torch.atleast_2d(torch.as_tensor(inference_result.params_array, device=inference_result.device))
    )

    thickness = np.atleast_1d(params_dict["thickness"].squeeze().detach().cpu().numpy())
    roughness = np.atleast_1d(params_dict["roughness"].squeeze().detach().cpu().numpy())
    sld = np.atleast_1d(params_dict["sld"].squeeze().detach().cpu().numpy())

    n_layers = thickness.shape[0]
    if roughness.shape[0] < n_layers + 1:
        raise ValueError(f"Expected roughness length {n_layers+1}, got {roughness.shape[0]}")
    if sld.shape[0] < n_layers + 1:
        raise ValueError(f"Expected sld length {n_layers+1}, got {sld.shape[0]}")

    ambient_rho = inference_result.ambient_sld.squeeze().detach().cpu().item() if inference_result.ambient_sld is not None else 0.0
    layer_rhos = sld[:n_layers]
    substrate_rho = sld[n_layers]
    sigma_top_sub = roughness[n_layers]

    if color_by_sld:
        layer_vals = np.real(layer_rhos) if np.iscomplexobj(layer_rhos) else layer_rhos

        norm = mpl.colors.Normalize(
            vmin=float(np.min(layer_vals)) if vmin is None else float(vmin),
            vmax=float(np.max(layer_vals)) if vmax is None else float(vmax),
        )
        cmap_obj = mpl.cm.get_cmap(cmap)
        layer_facecolors = [cmap_obj(norm(float(v))) for v in layer_vals]
    else:
        if layer_colors is None:
            base = ["#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99"]
            layer_facecolors = [base[i % len(base)] for i in range(n_layers)]
        else:
            if len(layer_colors) != n_layers:
                raise ValueError(f"layer_colors must have length {n_layers}, got {len(layer_colors)}")
            layer_facecolors = layer_colors

    def bold_title(name: str) -> str:
        safe = name.replace(" ", r"\ ")
        return rf"$\bf{{{safe}}}$"

    def fmt_layer_line(i: int) -> str:
        d = float(thickness[i])
        sigma_top = float(roughness[i])
        rho = layer_rhos[i]
        header = bold_title(f"Layer {i+1}")

        rho_txt = fmt_rho_value(rho, precision)

        if show_roughness:
            return (
                f"{header}   "
                f"{thickness_symbol}={d:.{precision}f} {thickness_unit}   "
                f"{roughness_symbol}={sigma_top:.{precision}f} {roughness_unit}   "
                f"{sld_symbol}={rho_txt} {sld_unit}"
            )
        return (
            f"{header}   "
            f"{thickness_symbol}={d:.{precision}f} {thickness_unit}   "
            f"{sld_symbol}={rho_txt} {sld_unit}"
        )

    def fmt_media_line(name: str, rho, sigma_top: Optional[float] = None) -> str:
        header = bold_title(name)
        rho_txt = fmt_rho_value(rho, precision)

        if show_roughness and sigma_top is not None:
            return (
                f"{header}   "
                f"{roughness_symbol}={sigma_top:.{precision}f} {roughness_unit}   "
                f"{sld_symbol}={rho_txt} {sld_unit}"
            )
        return f"{header}   {sld_symbol}={rho_txt} {sld_unit}"
    
    def fmt_rho_value(rho, precision: int) -> str:
        if np.iscomplexobj(rho):
            re = float(np.real(rho))
            im = float(np.imag(rho))
            return f"{re:.{precision}f} + i * {im:.{precision}f}"
        return f"{float(rho):.{precision}f}"

    blocks = [("ambient", media_height, ambient_color, ambient_rho, None)]
    for i in range(n_layers):
        blocks.append((f"layer{i+1}", layer_height, layer_facecolors[i], float(layer_rhos[i]), i))
    blocks.append(("substrate", media_height, substrate_color, substrate_rho, None))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    total_height = sum(h for _, h, *_ in blocks) + gap * (len(blocks) - 1)

    y = 0.0
    for name, h, facecolor, rho_val, idx in blocks:
        is_media = name in ("ambient", "substrate")

        rect = Rectangle(
            (0.0, y),
            box_width,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=(media_linewidth if is_media else layer_linewidth),
            alpha=(1.0 if is_media else layer_alpha),
        )

        if protruding_layers and not is_media:
            rect.set_path_effects([
                pe.SimplePatchShadow(offset=shadow_offset, alpha=shadow_alpha),
                pe.Normal(),
            ])

        ax.add_patch(rect)

        x_text = box_width / 2
        y_text = y + h / 2

        if name == "ambient":
            text = fmt_media_line(ambient_name, ambient_rho, sigma_top=None)
        elif name == "substrate":
            text = fmt_media_line(substrate_name, substrate_rho, sigma_top=sigma_top_sub)
        else:
            text = fmt_layer_line(idx)

        ax.text(x_text, y_text, text, fontsize=font_size, ha="center", va="center")
        y += h + gap

    ax.set_xlim(0, box_width)
    ax.set_ylim(-0.05, total_height)
    ax.invert_yaxis()
    ax.axis("off")

    if title is not None:
        ax.set_title(title, fontsize=font_size + 2)

    if color_by_sld and show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])

        label_symbol = sld_symbol
        if np.iscomplexobj(layer_rhos):
            label_symbol = rf"\Re({sld_symbol})"

        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(f"{label_symbol} [{sld_unit}]", fontsize=font_size)

    plt.tight_layout()
    return fig, ax