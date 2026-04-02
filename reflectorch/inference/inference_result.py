import json
from pathlib import Path
import re
from chainconsumer import ChainConsumer
from matplotlib.ticker import FormatStrFormatter
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from reflectorch.inference.data import ReflectivityData
from reflectorch.inference.importance_sampling import ImportanceSampling, SNISBackend
from reflectorch.inference.log_posterior import ReflectivityLogPosterior
from reflectorch.inference.mcmc import run_mcmc
from reflectorch.inference.plotting import (
    plot_reflectivity_v2, 
    plot_sld_profile, 
    plot_sampled_reflectivity_curves, 
    plot_sampled_sld_profiles,
    plot_sampled_profiles_multi_type,
)
from reflectorch.inference.multilayer_sketch import plot_multilayer_sketch

logging.getLogger("chainconsumer").setLevel(logging.ERROR)

PROFILE_PLOT_SPECS = {
    "sld": {
        "ylabel": r"SLD [$10^{-6}\ \mathrm{\AA^{-2}}$]",
        "sample_proxy_label": "sampled SLDs",
    },
    "imag_sld": {
        "ylabel": r"Imag. SLD [$10^{-6}\ \mathrm{\AA^{-2}}$]",
        "sample_proxy_label": "sampled imaginary SLDs",
    },
    "solvent_vf": {
        "ylabel": r"Solvent volume fraction",
        "sample_proxy_label": "sampled solvent volume fractions",
    },
}

def get_profile_plot_spec(profile_type: str) -> Dict[str, Any]:
    return PROFILE_PLOT_SPECS.get(
        str(profile_type),
        {
            "ylabel": str(profile_type).replace("_", " ").title(),
            "color": None,
            "sample_proxy_label": f"sampled {str(profile_type).replace('_', ' ')}",
        },
    )

def load_result(path, device="cpu"):
    """
    Load a saved inference result from an NPZ file.

    This is a convenience wrapper around
    :meth:`InferenceResultBase.from_npz` and returns the appropriate
    concrete result type stored in the file, for example
    :class:`PointInferenceResult` or :class:`PosteriorInferenceResult`.

    Args:
        path:
            Path to an NPZ file created with :meth:`to_npz`.
        device:
            Torch device used when rebuilding the underlying
            :class:`InferenceModel` during loading.

    Returns:
        A reconstructed inference result object.
    """
    return InferenceResultBase.from_npz(path, device=device)



class InferenceResultBase:
    SCHEMA_VERSION = 1
    KIND = "base"
    def __init__(
        self,
        inference_model: Any,
        data: ReflectivityData,
        prior_bounds: np.ndarray,
        ambient_sld: Optional[float],
        device: Union[str, torch.device],
    ):
        self.inference_model = inference_model
        self.data = data
        self.prior_bounds = np.asarray(prior_bounds)
        self.device = torch.device(device)

        self.param_model = inference_model.trainer.loader.prior_sampler.param_model

        supports_shift = self.param_model.supports_zero_ambient_sld_shift()
        if supports_shift:
            self.sld_indices = self.param_model.get_sld_indices()
        else:
            self.sld_indices = None

        self.ambient_sld = (
            torch.atleast_2d(torch.as_tensor(ambient_sld, device=self.device))
            if ambient_sld is not None
            else None
        )

        self.log_prob_fn = ReflectivityLogPosterior(
            param_model=self.param_model,
            q_exp=self.data.q,
            curve_exp=self.data.R,
            sigmas_exp=self.data.dR,
            q_resolution=self.data.dq,
            prior_bounds=self.prior_bounds,
            ambient_sld=ambient_sld,
            sld_indices=self.sld_indices,
            device=self.device,
        )

    @staticmethod
    def _merge_kwargs(defaults: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = dict(defaults)
        if user:
            out.update(user)
        return out

    @torch.inference_mode()
    def _subsample_rows(
        self,
        x: torch.Tensor,
        max_rows: Optional[int],
        seed: Optional[int] = 0,
    ) -> torch.Tensor:
        if max_rows is None or x.shape[0] <= max_rows:
            return x
        g = torch.Generator(device=x.device)
        if seed is not None:
            g.manual_seed(int(seed))
        idx = torch.randperm(x.shape[0], generator=g, device=x.device)[:max_rows]
        return x[idx]

    @torch.inference_mode()
    def _select_samples(
        self,
        which: str = "raw",
        max_rows: Optional[int] = None,
        seed: Optional[int] = 0,
        selection: str = "random",
        return_logp: bool = False,
    ):
        """
        Select samples from one of the stored sample sets.

        Args:
            which: raw, mcmc, or snis depending on subclass support.
            max_rows: Maximum number of rows to return. If None, return all.
            seed: Random seed used for random subsampling.
            selection: Selection strategy:
                - "random": uniform random subset
                - "top_logp": top samples by exact log posterior
            return_logp: If True, also return the corresponding log posterior values
                for the selected rows (or None if not computed).

        Returns:
            samples_cpu, logp_cpu_or_none
        """
        samples_cpu = self._get_samples(which=which)

        if selection == "random":
            selected = self._subsample_rows(samples_cpu, max_rows=max_rows, seed=seed)
            if return_logp:
                logp = self.log_prob_fn(selected.to(self.device)).detach().cpu()
                return selected, logp
            return selected, None

        if selection == "top_logp":
            samples = samples_cpu.to(self.device)
            logp = self.log_prob_fn(samples)

            bad = ~torch.isfinite(logp)
            if bad.any():
                logp = logp.clone()
                logp[bad] = -float("inf")

            if max_rows is None or max_rows >= samples.shape[0]:
                idx = torch.argsort(logp, descending=True)
            else:
                idx = torch.topk(logp, k=int(max_rows), largest=True).indices

            selected = samples[idx].detach().cpu()
            selected_logp = logp[idx].detach().cpu()
            return selected, selected_logp if return_logp else None

        raise ValueError("selection must be one of: 'random', 'top_logp'")

    @torch.inference_mode()
    def _prepare_theta_sim(self, theta: torch.Tensor) -> torch.Tensor:
        if self.ambient_sld is None or self.sld_indices is None:
            return theta
        
        theta_sim = theta.clone()
        theta_sim[..., self.sld_indices] = theta_sim[..., self.sld_indices] - self.ambient_sld
        return theta_sim

    def _default_plot_kwargs_sampled_curves(self) -> Dict[str, Any]:
        return dict(
            exp_label="exp. data",
            exp_color="blue",
            exp_marker="o",
            exp_facecolor="none",
            exp_ms=5,
            exp_alpha=1.0,
            exp_errcolor=None,
            exp_elinewidth=0.8,
            exp_capsize=0.0,
            exp_capthick=0.8,
            exp_zorder=2,

            sample_color="lightgreen",
            sample_lw=1.0,
            sample_alpha_scale=120,
            sample_zorder=1,

            proxy_label="sampled curves",
            legend=True,
            legend_fontsize=14,
            figsize=(6, 6),
            axis_label_size=18,
            tick_label_size=14,
            tight_layout=True,
        )

    def _default_plot_kwargs_sampled_profiles(self, profile_type: str = "sld") -> Dict[str, Any]:
        spec = self._get_profile_plot_spec(profile_type)
        return dict(
            sample_color="lightgreen",
            sample_lw=1.0,
            sample_alpha_scale=50,
            sample_zorder=1,
            proxy_label=spec["sample_proxy_label"],
            z_label=r"z [$\mathrm{\AA}$]",
            sld_label=spec["ylabel"],
            color_plot=True,
            cmap="jet",
            color_plot_alpha_scale=200,
            colorbar=True,
            colorbar_label="",
            legend=False,
            legend_fontsize=14,
            legend_loc="upper left",
            axis_label_size=18,
            tick_label_size=14,
            figsize=(8, 6),
            tight_layout=True,
        )
    
    def get_data(self) -> ReflectivityData:
        """
        Return the experimental reflectivity dataset stored in this result.

        Returns:
            A :class:`ReflectivityData` object containing the measured
            q-values, reflectivity values, and optional experimental
            uncertainties and q-resolution information.
        """
        return self.data

    @torch.inference_mode()
    def _batched_reflectivity(
        self,
        theta_sim: torch.Tensor,  # [N, dim]
        q: torch.Tensor,
        dq: Optional[torch.Tensor] = None,
        max_batch: int = 2048,
        conditioning_ambient_sld: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reflectivity curves in mini-batches to avoid OOM. Returns tensor [N, n_q] on CPU.
        """

        N = theta_sim.shape[0]
        pieces = []

        dq_in = None
        if dq is not None:
            dq_in = dq
            if not torch.is_tensor(dq_in):
                dq_in = torch.as_tensor(dq_in, device=self.device)
            dq_in = torch.atleast_2d(dq_in).to(self.device)

        cond_in = None
        if conditioning_ambient_sld is not None:
            cond_in = conditioning_ambient_sld
            if not torch.is_tensor(cond_in):
                cond_in = torch.as_tensor(cond_in, device=self.device)
            cond_in = torch.atleast_2d(cond_in).to(self.device)

        for start in range(0, N, max_batch):
            stop = min(start + max_batch, N)
            th_b = theta_sim[start:stop]

            if dq_in is None:
                dq_b = None
            else:
                # dq_in is [1, n_q] or [B, n_q]
                if dq_in.shape[0] == 1 and th_b.shape[0] > 1:
                    dq_b = dq_in.expand(th_b.shape[0], -1)
                else:
                    dq_b = dq_in[start:stop] if dq_in.shape[0] > 1 else dq_in

            cond_b = None
            if cond_in is not None:
                if cond_in.shape[0] == 1 and th_b.shape[0] > 1:
                    cond_b = cond_in.expand(th_b.shape[0], -1)
                else:
                    cond_b = cond_in[start:stop] if cond_in.shape[0] > 1 else cond_in

            refl_kwargs = {}
            if dq_b is not None:
                refl_kwargs["dq"] = dq_b
            if cond_b is not None:
                refl_kwargs["conditioning_ambient_sld"] = cond_b

            curves_b = self.param_model.reflectivity(
                q=q,
                parametrized_model=th_b,
                **refl_kwargs,
            )
            pieces.append(curves_b.detach().cpu())

        return torch.cat(pieces, dim=0)

    def _get_samples(self, which: str) -> torch.Tensor:
        """Must be implemented by subclasses"""
        raise NotImplementedError

    @torch.inference_mode()
    def get_sampled_curves(
        self,
        which: str = "raw",
        max_sim_batch: int = 2048,
        max_plot_samples: Optional[int] = 5000,
        seed: Optional[int] = 0,
        selection: str = "random",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate reflectivity curves from a selected set of parameter samples.

        The chosen sample set is converted to the parameter representation
        used internally by the reflectivity simulator, including any required
        ambient-SLD handling, and the corresponding reflectivity curves are
        computed in mini-batches.

        Args:
            which:
                Identifier of the sample set to use. Supported values depend on
                the concrete result subclass. Typical values include ``"raw"``,
                ``"mcmc"``, and ``"snis"``.
            max_sim_batch:
                Maximum number of samples simulated in one batch. Reduce this
                value if curve simulation runs out of memory.
            max_plot_samples:
                Maximum number of parameter samples to simulate. If the stored
                sample set is larger, a subset is selected according to
                ``selection``.
            seed:
                Random seed used when subsampling samples.
            selection:
                Strategy used to select samples when ``max_plot_samples`` is
                smaller than the stored sample set. Supported values are:

                - ``"random"``: uniform random subsampling
                - ``"top_logp"``: select the samples with the highest exact
                log posterior values

        Returns:
            A tuple ``(q_sampled, sampled_curves)`` where:

            - ``q_sampled`` is a NumPy array of shape ``(n_q,)`` containing
            the q-grid used for simulation
            - ``sampled_curves`` is a NumPy array of shape ``(N, n_q)``
            containing the simulated reflectivity curves
        """
        samples_cpu, _ = self._select_samples(
                which=which,
                max_rows=max_plot_samples,
                seed=seed,
                selection=selection,
                return_logp=False,
            )
        samples = samples_cpu.to(self.device)

        theta_sim = self._prepare_theta_sim(samples)

        q = torch.as_tensor(self.data.q, device=self.device, dtype=torch.float64)
        dq = None
        if self.data.dq is not None:
            dq = torch.atleast_2d(
                torch.as_tensor(self.data.dq, device=self.device, dtype=torch.float64)
            )

        supports_shift = self.param_model.supports_zero_ambient_sld_shift()

        conditioning_ambient_sld_t = None
        if self.ambient_sld is not None and not supports_shift:
            conditioning_ambient_sld_t = self.ambient_sld

        curves = self._batched_reflectivity(
            theta_sim=theta_sim,
            q=q,
            dq=dq,
            max_batch=max_sim_batch,
            conditioning_ambient_sld=conditioning_ambient_sld_t,
        )

        return np.asarray(self.data.q), curves.numpy()

    @torch.inference_mode()
    def plot_sampled_curves(
        self,
        which: str = "raw",
        max_sim_batch: int = 2048,
        max_plot_samples: Optional[int] = 5000,
        seed: Optional[int] = 0,
        selection: str = "random",
        plot_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Plot reflectivity curves simulated from a selected parameter sample set.

        This method is a convenience wrapper around
        :meth:`get_sampled_curves`. It overlays the experimental data stored in
        the result with an ensemble of sampled reflectivity curves.

        Args:
            which:
                Identifier of the sample set to use. Supported values depend on
                the concrete result subclass.
            max_sim_batch:
                Maximum number of simulated curves per batch.
            max_plot_samples:
                Maximum number of parameter samples to visualize.
            seed:
                Random seed used when subsampling samples.
            selection:
                Sample-selection strategy. Supported values are ``"random"``
                and ``"top_logp"``.
            plot_kwargs:
                Optional dictionary of plotting keyword arguments passed to the
                reflectivity plotting helper.
            **kwargs:
                Additional plotting keyword arguments merged into
                ``plot_kwargs``.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axis.
        """
        if kwargs:
            plot_kwargs = dict(plot_kwargs or {})
            plot_kwargs.update(kwargs)

        merged = self._merge_kwargs(self._default_plot_kwargs_sampled_curves(), plot_kwargs)

        q_sampled, curves = self.get_sampled_curves(
            which=which,
            max_sim_batch=max_sim_batch,
            max_plot_samples=max_plot_samples,
            seed=seed,
            selection=selection,
        )

        fig, ax = plot_sampled_reflectivity_curves(
            q_exp=self.data.q,
            curve_exp=self.data.R,
            yerr=self.data.dR,
            xerr=None,
            q_sampled=q_sampled,
            sampled_curves=curves,
            **merged,
        )
        return fig, ax
    
    @torch.inference_mode()
    def _compute_profiles(
        self,
        theta: torch.Tensor,
        *,
        profile_type: str = "sld",
        sld_profile_padding_left: float,
        sld_profile_padding_right: float,
        num_points_zaxis: int,
        z_axis: Optional[torch.Tensor] = None,
        **profile_kwargs,
    ):
        """
        Compute a named 1D profile from the parameter model.

        Args:
            theta: Parameter tensor of shape (B, dim)
            profile_type: e.g. "sld", "imag_sld".
            sld_profile_padding_left: Left padding factor for generated z axis
            sld_profile_padding_right: Right padding factor for generated z axis
            num_points_zaxis: Number of z points if z_axis is not provided
            z_axis: Optional explicit z axis
            **profile_kwargs: Forwarded to the param-model profile method

        Returns:
            z_axis, profiles
        """
        return self.param_model.profile(
            theta,
            profile_type=profile_type,
            ambient_sld=self.ambient_sld,
            z_axis=z_axis,
            num=num_points_zaxis,
            padding_left=sld_profile_padding_left,
            padding_right=sld_profile_padding_right,
            **profile_kwargs,
        )
    
    @torch.inference_mode()
    def get_sampled_sld_profiles(
        self,
        which: str = "raw",
        profile_type: str = "sld",
        max_plot_samples: Optional[int] = 5000,
        seed: Optional[int] = 0,
        selection: str = "random",
        sld_profile_padding_left: float = 0.4,
        sld_profile_padding_right: float = 1.3,
        num_points_zaxis: int = 1024,
        **profile_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one-dimensional profiles implied by a selected parameter sample set.

        Despite the historical method name, this method is not limited to the
        real SLD profile. Any profile type supported by the underlying
        parameter model may be requested, for example ``"sld"`` or ``"imag_sld"``.

        Args:
            which:
                Identifier of the sample set to use. Supported values depend on
                the concrete result subclass.
            profile_type:
                Name of the profile to compute. Use
                :meth:`available_profile_types` to inspect the supported
                profile types for the current parameter model.
            max_plot_samples:
                Maximum number of parameter samples used to compute profiles.
            seed:
                Random seed used when subsampling samples.
            selection:
                Sample-selection strategy. Supported values are ``"random"``
                and ``"top_logp"``.
            sld_profile_padding_left:
                Left padding factor used when automatically constructing the
                depth axis.
            sld_profile_padding_right:
                Right padding factor used when automatically constructing the
                depth axis.
            num_points_zaxis:
                Number of points used for the generated depth axis.
            **profile_kwargs:
                Additional keyword arguments forwarded to the parameter-model
                profile generator.

        Returns:
            A tuple ``(z_axis, profiles)`` where:

            - ``z_axis`` is a NumPy array of shape ``(n_z,)`` containing the
            depth coordinate
            - ``profiles`` is a NumPy array of shape ``(N, n_z)`` containing
            one profile per selected sample
        """
        samples_cpu, _ = self._select_samples(
            which=which,
            max_rows=max_plot_samples,
            seed=seed,
            selection=selection,
            return_logp=False,
        )
        samples = samples_cpu.to(self.device)

        z_axis, profiles = self._compute_profiles(
            samples,
            profile_type=profile_type,
            sld_profile_padding_left=sld_profile_padding_left,
            sld_profile_padding_right=sld_profile_padding_right,
            num_points_zaxis=num_points_zaxis,
            **profile_kwargs,
        )

        return (
            z_axis.squeeze().detach().cpu().numpy(),
            profiles.detach().cpu().numpy(),
        )

    @torch.inference_mode()
    def plot_sampled_sld_profiles(
        self,
        which: str = "raw",
        profile_type: str = "sld",
        max_plot_samples: Optional[int] = 5000,
        seed: Optional[int] = 0,
        selection: str = "random",
        sld_profile_padding_left: float = 0.4,
        sld_profile_padding_right: float = 1.3,
        num_points_zaxis: int = 1024,

        show_prior_envelope: bool = False,
        n_prior: int = 5000,
        prior_q_lo: float = 0.00,
        prior_q_hi: float = 1.00,
        prior_seed: Optional[int] = 0,
        prior_alpha: float = 0.1,

        plot_kwargs: Optional[Dict[str, Any]] = None,
        profile_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Plot profiles implied by a selected parameter sample set.

        This method visualizes an ensemble of one-dimensional profiles derived
        from posterior samples. The requested profile type may be the real SLD
        profile or any other profile supported by the parameter model.

        Args:
            which:
                Identifier of the sample set to use. Supported values depend on
                the concrete result subclass.
            profile_type:
                Name of the profile to plot.
            max_plot_samples:
                Maximum number of parameter samples used for plotting.
            seed:
                Random seed used when subsampling samples.
            selection:
                Sample-selection strategy. Supported values are ``"random"``
                and ``"top_logp"``.
            sld_profile_padding_left:
                Left padding factor used when constructing the depth axis.
            sld_profile_padding_right:
                Right padding factor used when constructing the depth axis.
            num_points_zaxis:
                Number of points used for the generated depth axis.
            show_prior_envelope:
                If ``True``, also draw a quantile envelope obtained from
                uniformly sampled prior parameters.
            n_prior:
                Number of prior samples used to estimate the prior envelope.
            prior_q_lo:
                Lower quantile of the prior envelope.
            prior_q_hi:
                Upper quantile of the prior envelope.
            prior_seed:
                Random seed used for prior-envelope sampling.
            prior_alpha:
                Transparency of the prior-envelope fill.
            plot_kwargs:
                Optional plotting keyword arguments for the profile plotting
                helper.
            profile_kwargs:
                Optional keyword arguments forwarded to the parameter-model
                profile generator.
            **kwargs:
                Additional plotting keyword arguments merged into
                ``plot_kwargs``.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axis.
        """
        profile_kwargs = dict(profile_kwargs or {})

        if kwargs:
            plot_kwargs = dict(plot_kwargs or {})
            plot_kwargs.update(kwargs)
        
        merged = self._merge_kwargs(
            self._default_plot_kwargs_sampled_profiles(profile_type=profile_type),
            plot_kwargs,
        )

        z_axis, sld_profiles = self.get_sampled_sld_profiles(
            which=which,
            profile_type=profile_type,
            max_plot_samples=max_plot_samples,
            seed=seed,
            selection=selection,
            sld_profile_padding_left=sld_profile_padding_left,
            sld_profile_padding_right=sld_profile_padding_right,
            num_points_zaxis=num_points_zaxis,
            **profile_kwargs,
        )

        fig, ax = plot_sampled_sld_profiles(
            z_sld=z_axis,
            sampled_slds=sld_profiles,
            **merged,
        )

        if show_prior_envelope:
            pb = np.asarray(self.prior_bounds)
            dim = pb.shape[0]

            rng = np.random.default_rng(prior_seed if prior_seed is not None else None)
            u = rng.random((n_prior, dim), dtype=np.float64)
            theta_prior_np = pb[:, 0][None, :] + u * (pb[:, 1] - pb[:, 0])[None, :]

            theta_prior = torch.as_tensor(theta_prior_np, device=self.device, dtype=torch.float32)

            z_p, prof_p = self._compute_profiles(
                theta_prior,
                profile_type=profile_type,
                sld_profile_padding_left=sld_profile_padding_left,
                sld_profile_padding_right=sld_profile_padding_right,
                num_points_zaxis=num_points_zaxis,
                **profile_kwargs,
            )

            z_p = z_p.squeeze().detach().cpu().numpy()
            prof_p = prof_p.detach().cpu().numpy()

            lo = np.quantile(prof_p, prior_q_lo, axis=0)
            hi = np.quantile(prof_p, prior_q_hi, axis=0)

            ax.fill_between(z_p, lo, hi, alpha=prior_alpha, label=f"prior {prior_q_lo:.0%}-{prior_q_hi:.0%}")

        return fig, ax
    
    @torch.inference_mode()
    def plot_sampled_profiles_multi_type(
        self,
        which: str = "raw",
        profile_types: Sequence[str] = ("sld",),
        max_plot_samples: Optional[int] = 5000,
        seed: Optional[int] = 0,
        selection: str = "random",
        sld_profile_padding_left: float = 0.4,
        sld_profile_padding_right: float = 1.3,
        num_points_zaxis: int = 1024,
        use_twin_axis: bool = False,
        twin_axis_index: int = 1,
        profile_labels: Optional[Sequence[str]] = None,
        profile_colors: Optional[Sequence[Optional[str]]] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        profile_kwargs_by_type: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Plot several sampled profile types on the same figure.

        This method is useful when comparing related one-dimensional profiles such
        as ``"sld"``, ``"imag_sld"``, or ``"solvent_vf"``
        for the same posterior sample cloud.

        All requested profile types are evaluated on the same sampled parameter set
        and plotted against a shared depth axis. By default they are drawn on a
        single y-axis. If exactly two profile types are requested, the second one
        can optionally be shown on a twin y-axis on the right-hand side of the
        figure.

        Args:
            which:
                Sample-set selector. Supported values depend on the concrete result
                type and typically include ``"raw"`` and, when available, ``"mcmc"``.
            profile_types:
                Sequence of profile-type names to plot.
            max_plot_samples:
                Maximum number of posterior samples used for plotting.
            seed:
                Random seed used when subsampling posterior samples.
            selection:
                Sample-selection strategy. Supported values are typically
                ``"random"`` and ``"top_logp"``.
            sld_profile_padding_left:
                Left padding factor used when constructing the depth axis.
            sld_profile_padding_right:
                Right padding factor used when constructing the depth axis.
            num_points_zaxis:
                Number of points used for the depth axis.
            use_twin_axis:
                If ``True`` and exactly two profile types are requested, draw the
                second profile type on a twin y-axis on the right.
            twin_axis_index:
                Index of the profile type that should use the right-hand axis when
                ``use_twin_axis=True``.
            profile_labels:
                Optional display labels, one per profile type. If omitted, labels
                are derived from ``PROFILE_PLOT_SPECS`` or from the raw
                ``profile_types`` names.
            profile_colors:
                Optional colors, one per profile type.
            plot_kwargs:
                Optional plotting keyword arguments forwarded to the underlying
                multi-profile plotting helper.
            profile_kwargs_by_type:
                Optional mapping from profile type to an additional dictionary of
                keyword arguments passed to the corresponding profile computation.

        Returns:
            If ``use_twin_axis=False``, returns ``(fig, ax)``.
            If ``use_twin_axis=True``, returns ``(fig, ax_left, ax_right)``.

        Notes:
            In the current implementation this method is intended for the standard
            line-ensemble profile plots and does not support the colormap-based
            plotting mode.
        """
        profile_types = list(profile_types)
        if len(profile_types) == 0:
            raise ValueError("Need at least one profile type.")

        profile_kwargs_by_type = dict(profile_kwargs_by_type or {})
        plot_kwargs = dict(plot_kwargs or {})

        z_axis_ref = None
        sampled_profiles_list = []

        for ptype in profile_types:
            kwargs_i = dict(profile_kwargs_by_type.get(ptype, {}))
            z_axis_i, prof_i = self.get_sampled_sld_profiles(
                which=which,
                profile_type=ptype,
                max_plot_samples=max_plot_samples,
                seed=seed,
                selection=selection,
                sld_profile_padding_left=sld_profile_padding_left,
                sld_profile_padding_right=sld_profile_padding_right,
                num_points_zaxis=num_points_zaxis,
                **kwargs_i,
            )

            if z_axis_ref is None:
                z_axis_ref = np.asarray(z_axis_i)
            else:
                if z_axis_ref.shape != np.asarray(z_axis_i).shape or not np.allclose(z_axis_ref, z_axis_i):
                    raise RuntimeError(
                        f"Profile type {ptype!r} produced a different z-axis. "
                        "All profile types must share the same z-grid for combined plotting."
                    )

            sampled_profiles_list.append(np.asarray(prof_i))

        if profile_labels is None:
            profile_labels = [
                self._get_profile_plot_spec(ptype).get(
                    "sample_proxy_label",
                    f"sampled {str(ptype).replace('_', ' ')}",
                )
                for ptype in profile_types
            ]

        ylabel_left = None
        ylabel_right = None

        if use_twin_axis and len(profile_types) == 2:
            left_idx = 1 - int(twin_axis_index)
            right_idx = int(twin_axis_index)
            ylabel_left = self._get_profile_plot_spec(profile_types[left_idx]).get("ylabel")
            ylabel_right = self._get_profile_plot_spec(profile_types[right_idx]).get("ylabel")
        else:
            if len(profile_types) == 1:
                ylabel_left = self._get_profile_plot_spec(profile_types[0]).get("ylabel")
            else:
                ylabel_left = "Profile value"

        return plot_sampled_profiles_multi_type(
            z_axis=z_axis_ref,
            sampled_profiles_list=sampled_profiles_list,
            profile_types=profile_types,
            profile_labels=profile_labels,
            profile_colors=profile_colors,
            use_twin_axis=use_twin_axis,
            twin_axis_index=twin_axis_index,
            ylabel_left=ylabel_left,
            ylabel_right=ylabel_right,
            **plot_kwargs,
        )
    
    def available_profile_types(self) -> List[str]:
        """
        Return the profile types supported by the underlying parameter model.

        Returns:
            A list of strings such as ``"sld"``, ``"imag_sld"``,
            or other model-specific profile names.
        """
        return self.param_model.available_profile_types()
    
    def _get_profile_plot_spec(self, profile_type: str) -> Dict[str, Any]:
        return get_profile_plot_spec(profile_type)
    
    @torch.inference_mode()
    def corner_plot(
        self,
        which: str = "raw",  # "raw", "mcmc", "snis"
        max_plot_samples: Optional[int] = 20000,
        label: str = "posterior",
        seed: Optional[int] = 0,
        selection: str = "random",
        param_names: Optional[Sequence[str]] = None,
        param_names_kwargs: Optional[Dict[str, Any]] = None,
        chain_kwargs: Optional[Dict[str, Any]] = None,
        configure_kwargs: Optional[Dict[str, Any]] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        extents: Union[None, str, List[Tuple[float, float]]] = None,
        ignore_params: Optional[Sequence[int]] = None,
        disable_offset_text: bool = True,

        handle_constant: str = "drop",  # "jitter" | "drop" | "ignore"
        constant_tol: float = 1e-12,
        jitter_scale: float = 1e-6,  # relative to extent width
    ):
        """
        Create a corner plot of a selected parameter sample set.

        Samples are prepared for plotting with ChainConsumer. The method
        includes safeguards for parameters that are effectively constant, which
        can otherwise cause plotting failures.

        Args:
            which:
                Identifier of the sample set to plot.
            max_plot_samples:
                Maximum number of parameter samples used for plotting.
            label:
                Name of the sample set shown in the plot legend.
            seed:
                Random seed used for subsampling samples and, if requested, for
                adding jitter to nearly constant parameters.
            selection:
                Sample-selection strategy. Supported values are ``"random"``
                and ``"top_logp"``.
            param_names:
                Optional parameter labels to use in the plot. If omitted,
                labels are obtained from the parameter model in LaTeX form.
            param_names_kwargs:
                Optional keyword arguments forwarded to the parameter-label
                generator.
            chain_kwargs:
                Optional keyword arguments passed to
                ``ChainConsumer.add_chain``.
            configure_kwargs:
                Optional keyword arguments passed to
                ``ChainConsumer.configure``.
            plot_kwargs:
                Optional keyword arguments passed to
                ``ChainConsumer.plotter.plot``.
            extents:
                Plot-range specification. Supported values are:

                - ``None``: let ChainConsumer determine the ranges
                - ``"prior_bounds"``: use the stored prior bounds
                - ``"full_ranges"``: use the full training ranges from the
                prior sampler
                - a list of ``(min, max)`` tuples
            ignore_params:
                Optional sequence of parameter indices to exclude from the
                plot.
            disable_offset_text:
                If ``True``, suppress scientific-notation offset text on the
                axes.
            handle_constant:
                Strategy for parameters with standard deviation below
                ``constant_tol``. Supported values are:

                - ``"drop"``: remove such parameters from the plot
                - ``"jitter"``: add a small amount of noise for plotting only
                - ``"ignore"``: leave them unchanged
            constant_tol:
                Standard-deviation threshold used to classify a parameter as
                effectively constant.
            jitter_scale:
                Relative jitter scale used when
                ``handle_constant="jitter"``.

        Returns:
            The Matplotlib figure produced by ChainConsumer.
        """

        samples, _ = self._select_samples(
            which=which,
            max_rows=max_plot_samples,
            seed=seed,
            selection=selection,
            return_logp=False,
        )

        if ignore_params:
            dim = samples.shape[1]
            ignore = [i for i in ignore_params]
            if any((i < 0 or i >= dim) for i in ignore):
                raise IndexError(f"ignore_params has out-of-range indices for dim={dim}: {ignore}")
            keep_mask = torch.ones(dim, dtype=torch.bool, device=samples.device)
            keep_mask[ignore] = False
            samples = samples[:, keep_mask]
        else:
            keep_mask = None

        if param_names is not None:
            param_names = list(param_names)
        else:
            param_names_kwargs = dict(param_names_kwargs or {})
            param_names = self.param_model.get_param_labels_latex(**param_names_kwargs)

        if extents in (None, "none"):
            use_extents = None
        elif isinstance(extents, list):
            use_extents = extents
        elif extents == "prior_bounds":
            pb = np.asarray(self.prior_bounds)
            use_extents = [(float(pb[i, 0]), float(pb[i, 1])) for i in range(pb.shape[0])]
        elif extents == "full_ranges":
            prior_sampler = self.inference_model.trainer.loader.prior_sampler
            minb = prior_sampler.min_bounds.squeeze(0).detach().cpu().numpy()
            maxb = prior_sampler.max_bounds.squeeze(0).detach().cpu().numpy()
            use_extents = [(float(minb[i]), float(maxb[i])) for i in range(minb.shape[0])]
        else:
            raise ValueError(
                "extents must be one of: None, 'prior_bounds', 'full_ranges', or a list of (min,max). "
                f"Got: {extents}"
            )

        if keep_mask is not None:
            keep_cpu = keep_mask.detach().cpu().numpy().tolist()
            if len(param_names) == len(keep_cpu):
                param_names = [p for p, k in zip(param_names, keep_cpu) if k]
            if use_extents is not None and len(use_extents) == len(keep_cpu):
                use_extents = [e for e, k in zip(use_extents, keep_cpu) if k]

        x = samples.detach().cpu().numpy()

        std = np.std(x, axis=0)
        constant_mask = std < float(constant_tol)

        if np.any(constant_mask):
            if handle_constant == "drop":
                keep = ~constant_mask
                x = x[:, keep]
                param_names = [p for p, k in zip(param_names, keep) if k]
                if use_extents is not None and len(use_extents) == len(keep):
                    use_extents = [e for e, k in zip(use_extents, keep) if k]

            elif handle_constant == "jitter":
                rng = np.random.default_rng(int(seed) if seed is not None else None)

                if use_extents is not None and len(use_extents) == x.shape[1]:
                    widths = np.array([max(e[1] - e[0], 1.0) for e in use_extents], dtype=float)
                else:
                    widths = np.ones(x.shape[1], dtype=float)

                noise = rng.normal(
                    loc=0.0,
                    scale=jitter_scale * widths[constant_mask],
                    size=(x.shape[0], int(np.sum(constant_mask))),
                )
                x[:, constant_mask] = x[:, constant_mask] + noise

            elif handle_constant == "ignore":
                pass
            else:
                raise ValueError("handle_constant must be one of: 'jitter', 'drop', 'ignore'")

        c = ChainConsumer()

        chain_defaults = dict(parameters=param_names, name=label)
        if chain_kwargs:
            chain_defaults.update(chain_kwargs)
        c.add_chain(x, **chain_defaults)

        default_configure = dict(
            serif=False,
            flip=True,
            plot_hists=True,
            colors=["blue"],
            spacing=1.5,
            cloud=False,
            sigmas=[0, 1, 2],
            diagonal_tick_labels=True,
            tick_font_size=8,
            label_font_size=16,
            max_ticks=4,
            linestyles="-",
            linewidths=1.0,
            shade=True,
            shade_alpha=0.5,
            kde=False,
            marker_style=".",
            legend_kwargs={"fontsize": 15},
        )
        if configure_kwargs:
            default_configure.update(configure_kwargs)

        c.configure(**default_configure)

        default_plot = dict(figsize=(10, 10), extents=use_extents)
        if plot_kwargs:
            default_plot.update(plot_kwargs)

        try:
            fig = c.plotter.plot(**default_plot)
        except IndexError:
            c = ChainConsumer()
            c.add_chain(x, **chain_defaults)
            default_configure["plot_hists"] = False
            c.configure(**default_configure)
            fig = c.plotter.plot(**default_plot)

        if disable_offset_text:
            for ax in fig.get_axes():
                ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax.xaxis.get_offset_text().set_visible(False)
                ax.yaxis.get_offset_text().set_visible(False)

            strip_bracket = lambda s: re.sub(r"\s*\[[^\]]*\]\s*$", "", s or "")
            for ax in fig.get_axes():
                ax.set_xlabel(strip_bracket(ax.get_xlabel()))
                ax.set_ylabel(strip_bracket(ax.get_ylabel()))

        return fig
    
    @torch.inference_mode()
    def get_corner_chain(
        self,
        which: str = "raw",
        max_plot_samples: Optional[int] = 20000,
        seed: Optional[int] = 0,
        selection: str = "random",
        param_names: Optional[Sequence[str]] = None,
        param_names_kwargs: Optional[Dict[str, Any]] = None,
        extents: Union[None, str, List[Tuple[float, float]]] = None,
        ignore_params: Optional[Sequence[int]] = None,
        handle_constant: str = "drop",
        constant_tol: float = 1e-12,
        jitter_scale: float = 1e-6,
    ):
        """
        Prepare samples, labels, and axis ranges for corner-style plotting.

        This method performs the same sample preparation and constant-parameter
        handling as :meth:`corner_plot`, but returns the processed arrays
        instead of creating a plot.

        Args:
            which:
                Identifier of the sample set to prepare.
            max_plot_samples:
                Maximum number of samples to return.
            seed:
                Random seed used for subsampling and optional jitter.
            selection:
                Sample-selection strategy. Supported values are ``"random"``
                and ``"top_logp"``.
            param_names:
                Optional parameter labels. If omitted, labels are obtained from
                the parameter model in LaTeX form.
            param_names_kwargs:
                Optional keyword arguments forwarded to the parameter-label
                generator.
            extents:
                Plot-range specification. Supported values are ``None``,
                ``"prior_bounds"``, ``"full_ranges"``, or a list of
                ``(min, max)`` tuples.
            ignore_params:
                Optional sequence of parameter indices to exclude.
            handle_constant:
                Strategy for effectively constant parameters.
            constant_tol:
                Standard-deviation threshold used to classify parameters as
                constant.
            jitter_scale:
                Relative jitter scale used when
                ``handle_constant="jitter"``.

        Returns:
            A tuple ``(x, param_names, use_extents)`` where:

            - ``x`` is a NumPy array of shape ``(N, d)``
            - ``param_names`` is the list of labels corresponding to the
            columns of ``x``
            - ``use_extents`` is either ``None`` or a list of plot extents
        """
        samples, _ = self._select_samples(
            which=which,
            max_rows=max_plot_samples,
            seed=seed,
            selection=selection,
            return_logp=False,
        )

        if ignore_params:
            dim = samples.shape[1]
            keep_mask = torch.ones(dim, dtype=torch.bool, device=samples.device)
            keep_mask[list(ignore_params)] = False
            samples = samples[:, keep_mask]
        else:
            keep_mask = None

        if param_names is not None:
            param_names = list(param_names)
        else:
            param_names_kwargs = dict(param_names_kwargs or {})
            param_names = self.param_model.get_param_labels_latex(**param_names_kwargs)

        if extents in (None, "none"):
            use_extents = None
        elif isinstance(extents, list):
            use_extents = extents
        elif extents == "prior_bounds":
            pb = np.asarray(self.prior_bounds)
            use_extents = [(float(pb[i, 0]), float(pb[i, 1])) for i in range(pb.shape[0])]
        elif extents == "full_ranges":
            prior_sampler = self.inference_model.trainer.loader.prior_sampler
            minb = prior_sampler.min_bounds.squeeze(0).detach().cpu().numpy()
            maxb = prior_sampler.max_bounds.squeeze(0).detach().cpu().numpy()
            use_extents = [(float(minb[i]), float(maxb[i])) for i in range(minb.shape[0])]
        else:
            raise ValueError("extents must be None, 'prior_bounds', 'full_ranges', or a list of (min,max).")

        if keep_mask is not None:
            keep_cpu = keep_mask.detach().cpu().numpy().tolist()
            if len(param_names) == len(keep_cpu):
                param_names = [p for p, k in zip(param_names, keep_cpu) if k]
            if use_extents is not None and len(use_extents) == len(keep_cpu):
                use_extents = [e for e, k in zip(use_extents, keep_cpu) if k]

        x = samples.detach().cpu().numpy()

        std = np.std(x, axis=0)
        constant_mask = std < float(constant_tol)
        if np.any(constant_mask):
            if handle_constant == "drop":
                keep = ~constant_mask
                x = x[:, keep]
                param_names = [p for p, k in zip(param_names, keep) if k]
                if use_extents is not None and len(use_extents) == len(keep):
                    use_extents = [e for e, k in zip(use_extents, keep) if k]
            elif handle_constant == "jitter":
                rng = np.random.default_rng(int(seed) if seed is not None else None)
                if use_extents is not None and len(use_extents) == x.shape[1]:
                    widths = np.array([max(e[1] - e[0], 1.0) for e in use_extents], dtype=float)
                else:
                    widths = np.ones(x.shape[1], dtype=float)
                noise = rng.normal(
                    loc=0.0,
                    scale=jitter_scale * widths[constant_mask],
                    size=(x.shape[0], int(np.sum(constant_mask))),
                )
                x[:, constant_mask] = x[:, constant_mask] + noise
            elif handle_constant == "ignore":
                pass
            else:
                raise ValueError("handle_constant must be one of: 'jitter', 'drop', 'ignore'")

        return x, param_names, use_extents
    
    @torch.inference_mode()
    def violin_plot(
        self,
        which: str = "raw",
        params: Optional[Sequence[int]] = None,
        max_plot_samples: Optional[int] = 20000,
        seed: Optional[int] = 0,
        selection: str = "random",
        param_names: Optional[Sequence[str]] = None,
        param_names_kwargs: Optional[Dict[str, Any]] = None,
        orient: str = "h",   # "h" or "v"
        sort_by: str = "index",  # "index" | "median" | "std"
        show_means: bool = False,
        show_medians: bool = True,
        show_extrema: bool = False,
        show_box: bool = True,
        show_points: bool = False,
        points_max: int = 300,
        widths: float = 0.8,
        figsize: Tuple[float, float] = (10, 6),
        title: Optional[str] = None,
        show_grid: bool = False,
        grid_alpha: float = 0.3,
        violin_facecolor: Optional[str] = None,
        violin_edgecolor: Optional[str] = "black",
        violin_linewidth: float = 1.2,
        violin_alpha: float = 0.55,
        label_fontsize: int = 18,
        tick_fontsize: int = 16,
        title_fontsize: Optional[int] = 18,
    ):
        """
        Create a violin plot for one or more inferred parameters.

        Args:
            which:
                Identifier of the sample set to plot.
            params:
                Optional sequence of parameter indices to include. If omitted,
                all parameters are used.
            max_plot_samples:
                Maximum number of samples used for plotting.
            seed:
                Random seed used when subsampling samples and optional point
                overlays.
            selection:
                Sample-selection strategy. Supported values are ``"random"``
                and ``"top_logp"``.
            param_names:
                Optional parameter labels. If omitted, labels are obtained from
                the parameter model in LaTeX form.
            param_names_kwargs:
                Optional keyword arguments forwarded to the parameter-label
                generator.
            orient:
                Plot orientation, either ``"h"`` for horizontal or ``"v"`` for
                vertical.
            sort_by:
                Optional sorting rule for the displayed parameters. Supported
                values are ``"index"``, ``"median"``, and ``"std"``.
            show_means:
                If ``True``, draw the mean of each distribution.
            show_medians:
                If ``True``, draw the median of each distribution.
            show_extrema:
                If ``True``, draw extrema markers.
            show_box:
                If ``True``, overlay a boxplot on the violin plot.
            show_points:
                If ``True``, overlay a random subset of sample points.
            points_max:
                Maximum number of points used when ``show_points=True``.
            widths:
                Relative violin width.
            figsize:
                Figure size in inches.
            title:
                Optional plot title.
            show_grid:
                If ``True``, draw a grid along the value axis.
            grid_alpha:
                Grid transparency.
            violin_facecolor:
                Optional fill color for the violin bodies.
            violin_edgecolor:
                Edge color for the violin bodies.
            violin_linewidth:
                Line width for violin outlines.
            violin_alpha:
                Transparency of the violin bodies.
            label_fontsize:
                Axis-label font size.
            tick_fontsize:
                Tick-label font size.
            title_fontsize:
                Title font size.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axis.
        """
        samples_cpu, _ = self._select_samples(
            which=which,
            max_rows=max_plot_samples,
            seed=seed,
            selection=selection,
            return_logp=False,
        )

        dim = samples_cpu.shape[1]
        if params is None:
            params = list(range(dim))
        params = list(params)

        x = samples_cpu[:, params].numpy()  # shape (N, P)

        if param_names is not None:
            param_names = list(param_names)
        else:
            param_names_kwargs = dict(param_names_kwargs or {})
            param_names = self.param_model.get_param_labels_latex(**param_names_kwargs)

        labels = [param_names[i] if i < len(param_names) else f"p{i}" for i in params]

        if sort_by != "index":
            if sort_by == "median":
                score = np.median(x, axis=0)
            elif sort_by == "std":
                score = np.std(x, axis=0)
            else:
                raise ValueError("sort_by must be one of: 'index', 'median', 'std'")
            order = np.argsort(score)
            x = x[:, order]
            labels = [labels[i] for i in order]

        data = [x[:, j] for j in range(x.shape[1])]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        parts = ax.violinplot(
            dataset=data,
            showmeans=show_means,
            showmedians=show_medians,
            showextrema=show_extrema,
            widths=widths,
            vert=(orient == "v"),
        )

        for body in parts["bodies"]:
            if violin_facecolor is not None:
                body.set_facecolor(violin_facecolor)
            body.set_edgecolor(violin_edgecolor)
            body.set_linewidth(violin_linewidth)
            body.set_alpha(violin_alpha)

        for k in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if k in parts:
                parts[k].set_linewidth(1.0)

        pos = np.arange(1, len(labels) + 1)

        if show_box:
            ax.boxplot(
                data,
                vert=(orient == "v"),
                positions=pos,
                widths=0.15 if orient == "v" else 0.3,
                patch_artist=False,
                showfliers=False,
                manage_ticks=False,
            )

        if show_points:
            rng = np.random.default_rng(seed if seed is not None else None)
            n_pts = min(points_max, x.shape[0])
            idx = rng.choice(x.shape[0], size=n_pts, replace=False)

            for j, p in enumerate(pos):
                vals = x[idx, j]
                jitter = rng.normal(0.0, 0.03, size=n_pts)
                if orient == "v":
                    ax.plot(np.full(n_pts, p) + jitter, vals, ".", alpha=0.25)
                else:
                    ax.plot(vals, np.full(n_pts, p) + jitter, ".", alpha=0.25)

        if show_grid:
            ax.grid(True, axis="x" if orient == "h" else "y", alpha=grid_alpha)
        else:
            ax.grid(False)

        if orient == "v":
            ax.set_xticks(pos)
            ax.set_xticklabels(labels, rotation=0, ha="right")
            ax.set_ylabel("Value", fontsize=label_fontsize)
        else:
            ax.set_yticks(pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Value", fontsize=label_fontsize)
            ax.invert_yaxis()

        ax.tick_params(axis="both", labelsize=tick_fontsize)

        if title is not None:
            ax.set_title(title, fontsize=title_fontsize if title_fontsize is not None else label_fontsize)

        plt.tight_layout()
        return fig, ax
    
    def to_npz(self, path: Union[str, Path], compress: bool = True):
        """
        Save this inference result to an NPZ file.

        The saved file contains the experimental data, prior bounds, result
        arrays, and metadata needed to rebuild the underlying
        :class:`InferenceModel` when loading the file again.

        Args:
            path:
                Output file path.
            compress:
                If ``True``, write a compressed NPZ file. If ``False``, write a
                standard uncompressed NPZ file.

        Returns:
            The output path as a :class:`Path` object.

        Notes:
            The concrete result subclass determines which arrays are stored.
            Loading the file later may require access to the original model
            configuration and weight files.
    """
        path = Path(path)
        meta, arrays = self._serialize_for_npz()
        meta["schema_version"] = int(self.SCHEMA_VERSION)
        meta["kind"] = str(self.KIND)

        payload = {"metadata_json": _json_to_npbytes(meta)}
        payload.update(arrays)

        if compress:
            np.savez_compressed(path, **payload)
        else:
            np.savez(path, **payload)

        return path

    @classmethod
    def from_npz(cls, path: str, device="cpu"):
        """
        Load an inference result from an NPZ file.

        The saved metadata is used to reconstruct the corresponding
        :class:`InferenceModel`, after which the appropriate concrete result
        class is instantiated and populated with the stored arrays.

        Args:
            path:
                Path to an NPZ file created with :meth:`to_npz`.
            device:
                Torch device used when rebuilding the underlying inference
                model.

        Returns:
            A reconstructed inference result object of the appropriate type.

        Raises:
            ValueError:
                If the file schema version is unsupported, the stored result
                kind is unknown, or the metadata is incomplete.

        Notes:
            This method is typically called indirectly via :func:`load_result`.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = _npbytes_to_json(z["metadata_json"])

            sv = int(meta.get("schema_version", 0))
            if sv != cls.SCHEMA_VERSION:
                raise ValueError(f"Unsupported schema_version={sv} (expected {cls.SCHEMA_VERSION}).")

            kind = meta.get("kind", None)
            if kind not in ("point", "posterior"):
                raise ValueError(f"Unknown kind in file: {kind}")

            mi = meta.get("model_info", None)
            if not isinstance(mi, dict) or not mi.get("config_name"):
                raise ValueError("Missing model_info/config_name in file; cannot rebuild inference model.")

            from reflectorch.inference.inference_model import InferenceModel
            inference_model = InferenceModel(
                config_name=mi.get("config_name"),
                model_name=mi.get("model_name"),
                root_dir=mi.get("root_dir"),
                weights_format=mi.get("weights_format", "safetensors"),
                repo_id=mi.get("repo_id"),
                device=device,
            )

            if kind == "point":
                result_cls = PointInferenceResult
            elif kind == "posterior":
                result_cls = PosteriorInferenceResult
            else:
                raise ValueError(f"Unknown kind in file: {kind}")

            return result_cls._construct_from_npz(meta, z, inference_model=inference_model, device=device)

    def _serialize_for_npz(self) -> Tuple[dict, dict]:
        raise NotImplementedError

class PointInferenceResult(InferenceResultBase):
    """
    Result object for point-estimate reflectivity inference.

    A point result stores the neural point prediction produced by
    :class:`InferenceModel.predict` or
    :meth:`InferenceModel.preprocess_and_predict`, and may additionally
    store a polished solution obtained by local numerical fitting.

    The object distinguishes between three point-estimate variants:

    - ``"predicted"``: the raw neural-network prediction
    - ``"polished"``: the locally refined solution, if available
    - ``"main"``: the default point estimate used by this object,
      defined as the polished solution if available and otherwise the
      raw prediction

    In addition to working with the point estimate itself, this class can
    optionally store MCMC samples initialized around the current main
    solution. Once such samples have been generated with
    :meth:`run_sampling_mcmc`, the inherited sample-based plotting and
    profile methods can be used with ``which="mcmc"``.
    """
    KIND = "point"
    def __init__(
        self,
        inference_model: Any,
        prediction_dict: Dict[str, Any],
        data: ReflectivityData,
        prior_bounds: np.ndarray,
        ambient_sld: Optional[float],
        device: Union[str, torch.device],
    ):
        super().__init__(
            inference_model=inference_model,
            data=data,
            prior_bounds=prior_bounds,
            ambient_sld=ambient_sld,
            device=device,
        )

        self.predicted_params_array = prediction_dict["predicted_params_array"]
        self.polished_params_array = prediction_dict.get("polished_params_array", None)
        self.polished_params_error_array = prediction_dict.get("polished_params_error_array", None)

        self.params_array = self.polished_params_array if self.polished_params_array is not None else self.predicted_params_array  

        self.param_names = prediction_dict.get("param_names", None) or self.param_model.get_param_labels()
        self.mcmc_samples: Optional[np.ndarray] = None

    @torch.inference_mode()
    def _get_samples(self, which: str) -> torch.Tensor:
        which = str(which).lower()
        if which != "mcmc":
            raise ValueError("PointInferenceResult supports only which='mcmc'.")
        if self.mcmc_samples is None:
            raise RuntimeError("Run `run_sampling_mcmc()` first.")
        
        return torch.as_tensor(self.mcmc_samples, device="cpu")
    
    def get_point_estimate(self) -> np.ndarray:
        """
        Return the current main point estimate as a NumPy array.

        The returned array corresponds to the result used internally by this
        object as its default estimate:

        - the polished parameters if a polished solution is available
        - otherwise the raw neural-network prediction

        Returns:
            A NumPy array of shape ``(n_params,)`` containing the main point
            estimate in physical parameter space.
        """
        return np.asarray(self.params_array)

    def get_mcmc_samples(self) -> Optional[np.ndarray]:
        """
        Return stored MCMC samples, if available.

        MCMC samples are not generated automatically. They are only present
        after calling :meth:`run_sampling_mcmc`.

        Returns:
            A NumPy array of shape ``(N, n_params)`` containing MCMC samples
            in physical parameter space, or ``None`` if no MCMC run has been
            performed.
        """
        return None if self.mcmc_samples is None else np.asarray(self.mcmc_samples)

    def print_table(
        self,
        which: str = "main",
        param_names=None,
        width=10,
        precision=3,
        header=True,
        show_prior_bounds: bool = False,
        show_polishing_errors: bool = False,
        show_training_ranges: bool = False,
    ):
        """
        Print a formatted parameter table for the stored point estimate.

        Args:
            which:
                Selects which point-estimate variant to print. Supported
                values are:

                - ``"main"``: the default result used by this object
                - ``"predicted"``: the raw neural-network prediction
                - ``"polished"``: the polished result, if available
                - ``"both"``: print predicted and polished values side by side
            param_names:
                Optional sequence of parameter labels. If omitted, the stored
                parameter names are used.
            width:
                Field width used for numeric columns.
            precision:
                Decimal precision used for numeric values.
            header:
                If ``True``, print a header row.
            show_prior_bounds:
                If ``True``, also print the lower and upper prior bounds for
                each parameter.
            show_polishing_errors:
                If ``True`` and polishing uncertainties are available, also
                print parameter uncertainties from the polishing stage.
            show_training_ranges:
                If ``True``, also print the global training min/max ranges
                from the prior sampler.

        Raises:
            RuntimeError:
                If ``which="polished"`` is requested but no polished result is
                available.
            ValueError:
                If ``which`` is not one of the supported values.
        """
        which = str(which).lower()
        param_names = param_names or self.param_names
        num_fmt = f"{{:>{width}.{precision}f}}"
        name_w = max(14, max((len(str(n)) for n in param_names), default=14))

        predicted = np.asarray(self.predicted_params_array)
        polished = (
            None if self.polished_params_array is None else np.asarray(self.polished_params_array)
        )

        polished_errors = getattr(self, "polished_params_error_array", None)
        if polished_errors is not None:
            polished_errors = np.asarray(polished_errors)
            if polished is not None and polished_errors.shape[0] != polished.shape[0]:
                if polished_errors.shape[0] > polished.shape[0]:
                    polished_errors = polished_errors[:polished.shape[0]]
                else:
                    polished_errors = None

        prior_min = prior_max = None
        if show_prior_bounds:
            pb = np.asarray(self.prior_bounds)
            prior_min, prior_max = pb[:, 0], pb[:, 1]

        train_min = train_max = None
        if show_training_ranges:
            prior_sampler = self.inference_model.trainer.loader.prior_sampler
            train_min = prior_sampler.min_bounds.squeeze().detach().cpu().numpy()
            train_max = prior_sampler.max_bounds.squeeze().detach().cpu().numpy()

        if which == "main":
            main_vals = np.asarray(self.params_array)
            use_pol_err = (
                show_polishing_errors
                and polished is not None
                and np.allclose(main_vals, polished)
                and polished_errors is not None
            )

            columns = [("Parameter", name_w), ("Result", width)]
            if use_pol_err:
                columns.append(("Pol. err.", width))
            if show_prior_bounds:
                columns.extend([("Prior min", width), ("Prior max", width)])
            if show_training_ranges:
                columns.extend([("Train min", width), ("Train max", width)])

            if header:
                hdr = "  ".join(
                    title.ljust(col_w) if title == "Parameter" else title.rjust(col_w)
                    for title, col_w in columns
                )
                print(hdr)
                print("-" * len(hdr))

            for i, name in enumerate(param_names):
                row = [str(name).ljust(name_w), num_fmt.format(main_vals[i])]
                if use_pol_err:
                    row.append(num_fmt.format(polished_errors[i]))
                if show_prior_bounds:
                    row.append(num_fmt.format(prior_min[i]))
                    row.append(num_fmt.format(prior_max[i]))
                if show_training_ranges:
                    row.append(num_fmt.format(train_min[i]))
                    row.append(num_fmt.format(train_max[i]))
                print("  ".join(row))

        elif which == "predicted":
            columns = [("Parameter", name_w), ("Predicted", width)]
            if show_prior_bounds:
                columns.extend([("Prior min", width), ("Prior max", width)])
            if show_training_ranges:
                columns.extend([("Train min", width), ("Train max", width)])

            if header:
                hdr = "  ".join(
                    title.ljust(col_w) if title == "Parameter" else title.rjust(col_w)
                    for title, col_w in columns
                )
                print(hdr)
                print("-" * len(hdr))

            for i, name in enumerate(param_names):
                row = [str(name).ljust(name_w), num_fmt.format(predicted[i])]
                if show_prior_bounds:
                    row.append(num_fmt.format(prior_min[i]))
                    row.append(num_fmt.format(prior_max[i]))
                if show_training_ranges:
                    row.append(num_fmt.format(train_min[i]))
                    row.append(num_fmt.format(train_max[i]))
                print("  ".join(row))

        elif which == "polished":
            if polished is None:
                raise RuntimeError("No polished parameters available.")

            use_pol_err = show_polishing_errors and polished_errors is not None

            columns = [("Parameter", name_w), ("Polished", width)]
            if use_pol_err:
                columns.append(("Pol. err.", width))
            if show_prior_bounds:
                columns.extend([("Prior min", width), ("Prior max", width)])
            if show_training_ranges:
                columns.extend([("Train min", width), ("Train max", width)])

            if header:
                hdr = "  ".join(
                    title.ljust(col_w) if title == "Parameter" else title.rjust(col_w)
                    for title, col_w in columns
                )
                print(hdr)
                print("-" * len(hdr))

            for i, name in enumerate(param_names):
                row = [str(name).ljust(name_w), num_fmt.format(polished[i])]
                if use_pol_err:
                    row.append(num_fmt.format(polished_errors[i]))
                if show_prior_bounds:
                    row.append(num_fmt.format(prior_min[i]))
                    row.append(num_fmt.format(prior_max[i]))
                if show_training_ranges:
                    row.append(num_fmt.format(train_min[i]))
                    row.append(num_fmt.format(train_max[i]))
                print("  ".join(row))

        elif which == "both":
            columns = [("Parameter", name_w), ("Predicted", width)]

            if polished is not None:
                columns.append(("Polished", width))
                if show_polishing_errors and polished_errors is not None:
                    columns.append(("Pol. err.", width))

            if show_prior_bounds:
                columns.extend([("Prior min", width), ("Prior max", width)])
            if show_training_ranges:
                columns.extend([("Train min", width), ("Train max", width)])

            if header:
                hdr = "  ".join(
                    title.ljust(col_w) if title == "Parameter" else title.rjust(col_w)
                    for title, col_w in columns
                )
                print(hdr)
                print("-" * len(hdr))

            for i, name in enumerate(param_names):
                row = [str(name).ljust(name_w), num_fmt.format(predicted[i])]

                if polished is not None:
                    row.append(num_fmt.format(polished[i]))
                    if show_polishing_errors and polished_errors is not None:
                        row.append(num_fmt.format(polished_errors[i]))

                if show_prior_bounds:
                    row.append(num_fmt.format(prior_min[i]))
                    row.append(num_fmt.format(prior_max[i]))
                if show_training_ranges:
                    row.append(num_fmt.format(train_min[i]))
                    row.append(num_fmt.format(train_max[i]))

                print("  ".join(row))

        else:
            raise ValueError("which must be one of: 'main', 'predicted', 'polished', 'both'")

    @torch.inference_mode()
    def get_curve(
        self,
        which: str = "main",
        dense_q_grid: bool = False,
        n_q_dense: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the reflectivity curve implied by a selected point estimate.

        Args:
            which:
                Selects which point-estimate variant to use. Supported values
                are:

                - ``"main"``
                - ``"predicted"``
                - ``"polished"``
            dense_q_grid:
                If ``True``, evaluate the reflectivity curve on a dense
                uniform q-grid spanning the experimental q-range instead of
                using the stored experimental q-values.
            n_q_dense:
                Number of q-points used when ``dense_q_grid=True``.

        Returns:
            A tuple ``(q_pred, r_pred)`` where:

            - ``q_pred`` is a NumPy array of q-values
            - ``r_pred`` is a NumPy array containing the corresponding
            reflectivity values

        Raises:
            RuntimeError:
                If ``which="polished"`` is requested but no polished result is
                available.
            ValueError:
                If ``which`` is not one of the supported values.
        """
        params = self._get_point_params(which=which)

        theta = torch.atleast_2d(torch.as_tensor(params, device=self.device))
        theta_sim = self._prepare_theta_sim(theta)

        q_pred_np = np.asarray(self.data.q)

        if dense_q_grid:
            q_min = float(q_pred_np.min())
            q_max = float(q_pred_np.max())
            q_pred_np = np.linspace(q_min, q_max, int(n_q_dense))

        q = torch.as_tensor(q_pred_np, device=self.device, dtype=torch.float64)

        dq = None
        if self.data.dq is not None:
            if np.isscalar(self.data.dq):
                dq = torch.atleast_2d(
                    torch.as_tensor(self.data.dq, device=self.device, dtype=torch.float64)
                )
            else:
                dq_np = np.asarray(self.data.dq)
                if not dense_q_grid:
                    dq_interp = dq_np
                else:
                    dq_interp = np.interp(q_pred_np, np.asarray(self.data.q), dq_np)
                dq = torch.atleast_2d(
                    torch.as_tensor(dq_interp, device=self.device, dtype=torch.float64)
                )

        curve_sim = self.param_model.reflectivity(
            q=q,
            parametrized_model=theta_sim,
            dq=dq,
        ).squeeze().detach().cpu().numpy()

        return q_pred_np, curve_sim            

    @torch.inference_mode()
    def plot_curve(
        self,
        which: str = "main",
        plot_kwargs: Optional[Dict[str, Any]] = None,
        dense_q_grid_plot: bool = False,
        n_q_dense: int = 200,
        **kwargs,
    ):
        """
        Plot the experimental reflectivity curve together with the model curve
        implied by a selected point estimate.

        Args:
            which:
                Selects which point-estimate variant to plot. Supported values
                are:

                - ``"main"``
                - ``"predicted"``
                - ``"polished"``
                - ``"both"``: overlay predicted and polished curves
            plot_kwargs:
                Optional dictionary of plotting keyword arguments forwarded to
                the reflectivity plotting helper.
            dense_q_grid_plot:
                If ``True``, evaluate the model curve on a dense uniform
                q-grid spanning the experimental q-range.
            n_q_dense:
                Number of q-points used when ``dense_q_grid_plot=True``.
            **kwargs:
                Additional plotting keyword arguments merged into
                ``plot_kwargs``.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axis.

        Raises:
            RuntimeError:
                If ``which="polished"`` is requested but no polished result is
                available.
            ValueError:
                If ``which`` is invalid, or if generic ``color`` / ``label``
                are passed when ``which="both"`` instead of the more specific
                ``pred_color`` / ``pol_color`` and ``pred_label`` /
                ``pol_label``.
        """
        plot_kwargs = dict(plot_kwargs or {})
        if kwargs:
            plot_kwargs.update(kwargs)

        which = str(which).lower()

        if which != "both":
            if "color" in plot_kwargs and "pred_color" not in plot_kwargs:
                plot_kwargs["pred_color"] = plot_kwargs.pop("color")
            if "label" in plot_kwargs and "pred_label" not in plot_kwargs:
                plot_kwargs["pred_label"] = plot_kwargs.pop("label")
        else:
            if "color" in plot_kwargs:
                raise ValueError(
                    "For which='both', use 'pred_color' and/or 'pol_color' instead of 'color'."
                )
            if "label" in plot_kwargs:
                raise ValueError(
                    "For which='both', use 'pred_label' and/or 'pol_label' instead of 'label'."
                )

        q_pred = r_pred = None
        q_pol = r_pol = None

        if which == "main":
            defaults = dict(
                pred_color="red",
                pred_label="result",
            )

            q_pred, r_pred = self.get_curve(
                which="main",
                dense_q_grid=dense_q_grid_plot,
                n_q_dense=n_q_dense,
            )

        elif which == "predicted":
            defaults = dict(
                pred_color="lightgreen",
                pred_label="predicted",
            )

            q_pred, r_pred = self.get_curve(
                which="predicted",
                dense_q_grid=dense_q_grid_plot,
                n_q_dense=n_q_dense,
            )

        elif which == "polished":
            defaults = dict(
                pred_color="orange",
                pred_label="polished",
            )

            q_pred, r_pred = self.get_curve(
                which="polished",
                dense_q_grid=dense_q_grid_plot,
                n_q_dense=n_q_dense,
            )

        elif which == "both":
            defaults = dict(
                pred_color="lightgreen",
                pred_label="predicted",
                pol_color="orange",
                pol_label="polished",
            )

            q_pred, r_pred = self.get_curve(
                which="predicted",
                dense_q_grid=dense_q_grid_plot,
                n_q_dense=n_q_dense,
            )

            if self.polished_params_array is not None:
                q_pol, r_pol = self.get_curve(
                    which="polished",
                    dense_q_grid=dense_q_grid_plot,
                    n_q_dense=n_q_dense,
                )

        else:
            raise ValueError(
                "which must be one of: 'main', 'predicted', 'polished', 'both'"
            )

        merged = self._merge_kwargs(defaults, plot_kwargs)

        fig, ax = plot_reflectivity_v2(
            q_exp=self.data.q,
            r_exp=self.data.R,
            yerr=self.data.dR,
            xerr=None,
            q_pred=q_pred,
            r_pred=r_pred,
            q_pol=q_pol,
            r_pol=r_pol,
            **merged,
        )
        return fig, ax

    @torch.inference_mode()
    def get_sld_profile(
        self,
        which: str = "main",
        profile_type: str = "sld",
        sld_profile_padding_left: float = 0.4,
        sld_profile_padding_right: float = 1.3,
        num_points_zaxis: int = 1024,
        **profile_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a one-dimensional profile implied by a selected point estimate.

        Despite the historical method name, this method is not limited to the
        real SLD profile. Any profile type supported by the parameter model
        may be requested.

        Args:
            which:
                Selects which point-estimate variant to use. Supported values
                are ``"main"``, ``"predicted"``, and ``"polished"``.
            profile_type:
                Name of the profile to compute, for example ``"sld"`` or any
                other profile type supported by the parameter model.
            sld_profile_padding_left:
                Left padding factor used when constructing the depth axis.
            sld_profile_padding_right:
                Right padding factor used when constructing the depth axis.
            num_points_zaxis:
                Number of points used for the generated depth axis.
            **profile_kwargs:
                Additional keyword arguments forwarded to the parameter-model
                profile generator.

        Returns:
            A tuple ``(z_sld, profile)`` where:

            - ``z_sld`` is a NumPy array of depth coordinates
            - ``profile`` is a NumPy array containing the requested profile

        Raises:
            RuntimeError:
                If ``which="polished"`` is requested but no polished result is
                available.
            ValueError:
                If ``which`` is invalid.
        """
        params = self._get_point_params(which=which)
        
        theta = torch.atleast_2d(
            torch.as_tensor(params, device=self.device, dtype=torch.float32)
        )

        z_axis, profiles = self._compute_profiles(
            theta,
            profile_type=profile_type,
            sld_profile_padding_left=sld_profile_padding_left,
            sld_profile_padding_right=sld_profile_padding_right,
            num_points_zaxis=num_points_zaxis,
            **profile_kwargs,
        )

        return (
            z_axis.squeeze().detach().cpu().numpy(),
            profiles.squeeze().detach().cpu().numpy(),
        )       

    @torch.inference_mode()
    def plot_sld_profile(
        self,
        which: str = "main",
        profile_type: str = "sld",
        sld_profile_padding_left: float = 0.4,
        sld_profile_padding_right: float = 1.3,
        num_points_zaxis: int = 1024,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        profile_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Plot a one-dimensional profile implied by a selected point estimate.

        Args:
            which:
                Selects which point-estimate variant to plot. Supported values
                are:

                - ``"main"``
                - ``"predicted"``
                - ``"polished"``
                - ``"both"``: overlay predicted and polished profiles
            profile_type:
                Name of the profile to plot.
            sld_profile_padding_left:
                Left padding factor used when constructing the depth axis.
            sld_profile_padding_right:
                Right padding factor used when constructing the depth axis.
            num_points_zaxis:
                Number of points used for the generated depth axis.
            plot_kwargs:
                Optional dictionary of plotting keyword arguments for the
                profile plotting helper.
            profile_kwargs:
                Optional keyword arguments forwarded to the parameter-model
                profile generator.
            **kwargs:
                Additional plotting keyword arguments merged into
                ``plot_kwargs``.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axis.

        Raises:
            RuntimeError:
                If ``which="polished"`` is requested but no polished result is
                available.
            ValueError:
                If ``which`` is invalid, or if generic ``color`` / ``label``
                are passed when ``which="both"`` instead of the more specific
                ``pred_color`` / ``pol_color`` and ``pred_label`` /
                ``pol_label``.
        """
        profile_kwargs = dict(profile_kwargs or {})

        plot_kwargs = dict(plot_kwargs or {})
        if kwargs:
            plot_kwargs.update(kwargs)

        which = str(which).lower()
        spec = self._get_profile_plot_spec(profile_type)

        if which != "both":
            if "color" in plot_kwargs and "pred_color" not in plot_kwargs:
                plot_kwargs["pred_color"] = plot_kwargs.pop("color")
            if "label" in plot_kwargs and "pred_label" not in plot_kwargs:
                plot_kwargs["pred_label"] = plot_kwargs.pop("label")
        else:
            if "color" in plot_kwargs:
                raise ValueError(
                    "For which='both', use 'pred_color' and/or 'pol_color' instead of 'color'."
                )
            if "label" in plot_kwargs:
                raise ValueError(
                    "For which='both', use 'pred_label' and/or 'pol_label' instead of 'label'."
                )

        z_sld = None
        sld_pred = None
        sld_pol = None

        if which == "main":
            defaults = dict(
                pred_color="red",
                pred_label=None,
            )

            z_sld, sld_pred = self.get_sld_profile(
                which="main",
                profile_type=profile_type,
                sld_profile_padding_left=sld_profile_padding_left,
                sld_profile_padding_right=sld_profile_padding_right,
                num_points_zaxis=num_points_zaxis,
                **profile_kwargs
            )

        elif which == "predicted":
            defaults = dict(
                pred_color="lightgreen",
                pred_label="predicted",
            )

            z_sld, sld_pred = self.get_sld_profile(
                which="predicted",
                profile_type=profile_type,
                sld_profile_padding_left=sld_profile_padding_left,
                sld_profile_padding_right=sld_profile_padding_right,
                num_points_zaxis=num_points_zaxis,
                **profile_kwargs
            )

        elif which == "polished":
            defaults = dict(
                pred_color="orange",
                pred_label="polished",
            )

            z_sld, sld_pred = self.get_sld_profile(
                which="polished",
                profile_type=profile_type,
                sld_profile_padding_left=sld_profile_padding_left,
                sld_profile_padding_right=sld_profile_padding_right,
                num_points_zaxis=num_points_zaxis,
                **profile_kwargs
            )

        elif which == "both":
            defaults = dict(
                pred_color="lightgreen",
                pred_label="predicted",
                pol_color="orange",
                pol_label="polished",
            )

            z_sld, sld_pred = self.get_sld_profile(
                which="predicted",
                profile_type=profile_type,
                sld_profile_padding_left=sld_profile_padding_left,
                sld_profile_padding_right=sld_profile_padding_right,
                num_points_zaxis=num_points_zaxis,
                **profile_kwargs
            )

            if self.polished_params_array is not None:
                _, sld_pol = self.get_sld_profile(
                    which="polished",
                    profile_type=profile_type,
                    sld_profile_padding_left=sld_profile_padding_left,
                    sld_profile_padding_right=sld_profile_padding_right,
                    num_points_zaxis=num_points_zaxis,
                    **profile_kwargs
                )

        else:
            raise ValueError(
                "which must be one of: 'main', 'predicted', 'polished', 'both'"
            )

        merged = self._merge_kwargs(defaults, plot_kwargs)
        merged.setdefault("sld_label", spec["ylabel"])

        fig, ax = plot_sld_profile(
            z_sld=z_sld,
            sld_pred=sld_pred,
            sld_pol=sld_pol,
            **merged,
        )
        return fig, ax
    
    @torch.inference_mode()
    def plot_multilayer_sketch(self, **kwargs):
        """
        Plot a schematic multilayer representation of the inferred structure.

        This is a convenience wrapper around the multilayer-sketch plotting
        helper and uses the current main point estimate.

        Args:
            **kwargs:
                Additional keyword arguments forwarded to the multilayer-sketch
                plotting helper.
        """
        plot_multilayer_sketch(self, **kwargs)

    @torch.inference_mode()
    def run_sampling_mcmc(self, num_chains: int = 4096, init_radius: float = 0.1, **kwargs):
        """
        Run MCMC sampling initialized around the current main point estimate.

        The current main point estimate is repeated across ``num_chains``
        initial walkers and then perturbed multiplicatively by a small random
        factor controlled by ``init_radius``.

        Args:
            num_chains:
                Number of initial chains or walkers.
            init_radius:
                Relative perturbation radius used when initializing walkers
                around the current main point estimate.
            **kwargs:
                Additional keyword arguments forwarded to the MCMC runner.

        Returns:
            The raw object returned by the MCMC runner.

        Notes:
            After this method is called successfully, the flattened MCMC
            samples are stored in :attr:`mcmc_samples` and the inherited
            sample-based plotting methods can be used with ``which="mcmc"``.
        """
        thetas = torch.from_numpy(np.asarray(self.params_array)).repeat(num_chains, 1).to(self.device)
        thetas = thetas * (1.0 + init_radius * (torch.rand_like(thetas) - 0.5))

        pb = torch.as_tensor(self.prior_bounds, device=self.device, dtype=thetas.dtype)
        fixed_mask = (pb[:, 1] - pb[:, 0]).abs() <= 1e-7
        if fixed_mask.any():
            thetas[:, fixed_mask] = pb[fixed_mask, 0]

        mcmc_result = run_mcmc(
            init_coords=thetas,
            log_prob_fn=self.log_prob_fn,
            **kwargs,
        )

        self.mcmc_samples = mcmc_result[0].get_chain(flat=True).cpu().numpy()
        return mcmc_result

    @torch.inference_mode()
    def plot_mcmc_curves(
        self,
        max_sim_batch: int = 2048,
        max_plot_samples: Optional[int] = 5000,
        seed: Optional[int] = 0,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Plot reflectivity curves simulated from stored MCMC samples.

        This is a convenience wrapper around :meth:`plot_sampled_curves` with
        ``which="mcmc"``.

        Args:
            max_sim_batch:
                Maximum number of simulated curves per batch.
            max_plot_samples:
                Maximum number of MCMC samples used for plotting.
            seed:
                Random seed used when subsampling MCMC samples.
            plot_kwargs:
                Optional plotting keyword arguments.
            **kwargs:
                Additional plotting keyword arguments merged into
                ``plot_kwargs``.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axis.

        Raises:
            RuntimeError:
                If no MCMC samples are stored yet.
        """
        return self.plot_sampled_curves(
            which="mcmc",
            max_sim_batch=max_sim_batch,
            max_plot_samples=max_plot_samples,
            seed=seed,
            plot_kwargs=plot_kwargs,
            **kwargs,
        )

    @torch.inference_mode()
    def plot_mcmc_sld_profiles(
        self,
        max_plot_samples: Optional[int] = 5000,
        seed: Optional[int] = 0,
        profile_type: str = "sld",
        sld_profile_padding_left: float = 0.4,
        sld_profile_padding_right: float = 1.3,
        num_points_zaxis: int = 1024,
        plot_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Plot profiles computed from stored MCMC samples.

        This is a convenience wrapper around :meth:`plot_sampled_sld_profiles`
        with ``which="mcmc"``.

        Args:
            max_plot_samples:
                Maximum number of MCMC samples used for plotting.
            seed:
                Random seed used when subsampling MCMC samples.
            profile_type:
                Name of the profile to plot.
            sld_profile_padding_left:
                Left padding factor used when constructing the depth axis.
            sld_profile_padding_right:
                Right padding factor used when constructing the depth axis.
            num_points_zaxis:
                Number of points used for the generated depth axis.
            plot_kwargs:
                Optional plotting keyword arguments.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axis.

        Raises:
            RuntimeError:
                If no MCMC samples are stored yet.
        """
        return self.plot_sampled_sld_profiles(
            which="mcmc",
            profile_type=profile_type,
            max_plot_samples=max_plot_samples,
            seed=seed,
            sld_profile_padding_left=sld_profile_padding_left,
            sld_profile_padding_right=sld_profile_padding_right,
            num_points_zaxis=num_points_zaxis,
            plot_kwargs=plot_kwargs,
        )

    @torch.inference_mode()
    def mcmc_corner_plots(
        self,
        max_plot_samples: Optional[int] = 20000,
        label: str = "MCMC",
        seed: Optional[int] = 0,
        param_names_kwargs: Optional[Dict[str, Any]] = None,
        chain_kwargs: Optional[Dict[str, Any]] = None,
        configure_kwargs: Optional[Dict[str, Any]] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        extents: Union[None, str, List[Tuple[float, float]]] = None,
    ):
        """
        Create a corner plot for stored MCMC samples.

        This is a convenience wrapper around :meth:`corner_plot` with
        ``which="mcmc"``.

        Args:
            max_plot_samples:
                Maximum number of MCMC samples used for plotting.
            label:
                Label used for the chain in the plot legend.
            seed:
                Random seed used when subsampling MCMC samples.
            param_names_kwargs:
                Optional keyword arguments forwarded to the parameter-label
                generator.
            chain_kwargs:
                Optional keyword arguments passed to
                ``ChainConsumer.add_chain``.
            configure_kwargs:
                Optional keyword arguments passed to
                ``ChainConsumer.configure``.
            plot_kwargs:
                Optional keyword arguments passed to
                ``ChainConsumer.plotter.plot``.
            extents:
                Plot-range specification. Supported values are ``None``,
                ``"prior_bounds"``, ``"full_ranges"``, or a list of
                ``(min, max)`` tuples.

        Returns:
            The Matplotlib figure produced by ChainConsumer.

        Raises:
            RuntimeError:
                If no MCMC samples are stored yet.
        """
        return self.corner_plot(
            which="mcmc",
            max_plot_samples=max_plot_samples,
            label=label,
            seed=seed,
            param_names_kwargs=param_names_kwargs,
            chain_kwargs=chain_kwargs,
            configure_kwargs=configure_kwargs,
            plot_kwargs=plot_kwargs,
            extents=extents,
        )

    def available_point_variants(self) -> List[str]:
        """
        Return the point-estimate variants available in this result.

        Returns:
            A list of supported selectors for point-estimate-based methods.

        Notes:
            The public point-estimate API accepts at least ``"main"`` and
            ``"predicted"``, and additionally ``"polished"`` if a polished
            solution is available.
        """
        variants = ["main", "predicted"]
        if self.polished_params_array is not None:
            variants.append("polished")
        return variants
    
    def _get_point_params(self, which: str = "main") -> np.ndarray:
        which = str(which).lower()
        if which == "main":
            return np.asarray(self.params_array)
        if which == "predicted":
            return np.asarray(self.predicted_params_array)
        if which == "polished":
            if self.polished_params_array is None:
                raise RuntimeError("No polished parameters available.")
            return np.asarray(self.polished_params_array)
        raise ValueError("which must be one of: 'main', 'predicted', 'polished'")

    def __repr__(self):
        n = len(self.params_array)
        mcmc = None if self.mcmc_samples is None else int(self.mcmc_samples.shape[0])
        dq_str = "None" if self.data.dq is None else ("float" if np.isscalar(self.data.dq) else "pointwise")
        return (
            f"PointInferenceResult(dim={n}, mcmc_samples={mcmc}, "
            f"dq={dq_str}, ambient_sld={'yes' if self.ambient_sld is not None else 'no'})"
        )

    def summary(self, show_prior_bounds: bool = True):
        """
        Print a short textual summary of this point-inference result.

        The summary includes:

        - the object representation
        - whether the current main result is the polished or predicted
        solution
        - a formatted parameter table for the main point estimate

        Args:
            show_prior_bounds:
                If ``True``, include prior bounds in the printed parameter
                table.
        """
        print(self.__repr__())
        if self.polished_params_array is not None:
            print("Using polished parameters.")
        else:
            print("Using predicted parameters.")
        self.print_table(show_prior_bounds=show_prior_bounds)

    def _serialize_for_npz(self):
        arrays = {}

        arrays["q"] = np.asarray(self.data.q)
        arrays["R"] = np.asarray(self.data.R)
        if self.data.dR is not None:
            arrays["dR"] = np.asarray(self.data.dR)
        if self.data.dq is not None:
            arrays["dq"] = np.asarray(self.data.dq)

        arrays["prior_bounds"] = np.asarray(self.prior_bounds)
        arrays["predicted_params_array"] = np.asarray(self.predicted_params_array)
        if self.polished_params_array is not None:
            arrays["polished_params_array"] = np.asarray(self.polished_params_array)
        if self.polished_params_error_array is not None:
            arrays["polished_params_error_array"] = np.asarray(self.polished_params_error_array)

        if getattr(self, "mcmc_samples", None) is not None:
            arrays["mcmc_samples"] = np.asarray(self.mcmc_samples)

        meta = {
            "ambient_sld": None if self.ambient_sld is None else float(self.ambient_sld.squeeze().cpu().item()),
            "param_names": list(self.param_names) if self.param_names is not None else None,
            "model_info": {
                "config_name": getattr(self.inference_model, "config_name", None),
                "model_name": getattr(self.inference_model, "model_name", None),
                "root_dir": getattr(self.inference_model, "root_dir", None),
                "weights_format": getattr(self.inference_model, "weights_format", None),
                "repo_id": getattr(self.inference_model, "repo_id", None),
            },
        }
        return meta, arrays

    @staticmethod
    def _construct_from_npz(meta, z, inference_model=None, device="cpu"):
        q = z["q"]
        R = z["R"]
        dR = z["dR"] if "dR" in z.files else None
        dq = z["dq"] if "dq" in z.files else None

        data = ReflectivityData(q=q, R=R, dR=dR, dq=dq)

        pred = dict(
            predicted_params_array=z["predicted_params_array"],
            polished_params_array=(z["polished_params_array"] if "polished_params_array" in z.files else None),
            polished_params_error_array=(z["polished_params_error_array"] if "polished_params_error_array" in z.files else None),
            param_names=meta.get("param_names", None),
        )

        res = PointInferenceResult(
            inference_model=inference_model,
            prediction_dict=pred,
            data=data,
            prior_bounds=z["prior_bounds"],
            ambient_sld=meta.get("ambient_sld", None),
            device=device,
        )

        if "mcmc_samples" in z.files:
            res.mcmc_samples = np.asarray(z["mcmc_samples"])

        return res


class PosteriorInferenceResult(InferenceResultBase):
    """
    Result object for posterior-sampling inference.

    A posterior result stores one or more parameter sample sets for a
    single observation. The primary sample set is the raw neural posterior
    sample set produced by :class:`InferenceModel.sample` or
    :meth:`InferenceModel.preprocess_and_sample`. Additional sample sets
    may be generated later by optional refinement or resampling methods.

    Supported sample variants are:

    - ``"raw"``:
      direct neural posterior samples
    - ``"mcmc"``:
      samples obtained by MCMC refinement of the exact reflectivity
      posterior
    - ``"snis"``:
      approximately unweighted samples produced by self-normalized
      importance sampling (SNIS)

    Inherited plotting, curve-simulation, profile-simulation, corner-plot,
    and violin-plot methods can be applied to any sample set that is
    currently available.
    """
    KIND = "posterior"
    def __init__(
        self,
        inference_model: Any,
        prediction_dict: Dict[str, Any],
        data: ReflectivityData,
        prior_bounds: np.ndarray,
        ambient_sld: Optional[float],
        device: Union[str, torch.device],
        snis_backend: Optional[SNISBackend] = None,
    ):
        super().__init__(
            inference_model=inference_model,
            data=data,
            prior_bounds=prior_bounds,
            ambient_sld=ambient_sld,
            device=device,
        )

        raw = prediction_dict.get("predicted_params_array", None)
        if raw is None:
            raise ValueError("prediction_dict must contain 'predicted_params_array' (raw posterior samples).")

        raw = np.asarray(raw)
        if raw.ndim == 1:
            raw = raw[None, :]

        self.raw_samples_array = raw
        self.param_names = prediction_dict.get("param_names", None) or self.param_model.get_param_labels()

        self.mcmc_samples: Optional[np.ndarray] = None

        self.importance_sampling = ImportanceSampling(backend=snis_backend)
        self.snis_samples: Optional[np.ndarray] = None
        self.snis_log_weights: Optional[np.ndarray] = None

    @torch.inference_mode()
    def _get_samples(self, which: str) -> torch.Tensor:
        which = str(which).lower()
        if which == "raw":
            return torch.as_tensor(self.raw_samples_array, device="cpu")
        if which == "mcmc":
            if self.mcmc_samples is None:
                raise RuntimeError("No MCMC samples. Run `run_mcmc_refinement()` first.")
            return torch.as_tensor(self.mcmc_samples, device="cpu")
        if which in ("snis", "sir", "is"):
            if self.snis_samples is None:
                raise RuntimeError("No SNIS samples. Run `run_neural_snis()` then `resample_snis()`.")
            return torch.as_tensor(self.snis_samples, device="cpu")
        raise ValueError("'which' must be one of: 'raw', 'mcmc', 'snis'")
    
    def get_samples(self, which: str = "raw") -> np.ndarray:
        """
        Return a selected posterior sample set as a NumPy array.

        Args:
            which:
                Identifier of the sample set to return. Supported values are:

                - ``"raw"``: raw neural posterior samples
                - ``"mcmc"``: MCMC-refined samples, if available
                - ``"snis"``: SNIS-resampled samples, if available

        Returns:
            A NumPy array of shape ``(N, n_params)`` containing parameter
            samples in physical parameter space.

        Raises:
            RuntimeError:
                If the requested sample set has not been generated yet.
            ValueError:
                If ``which`` is not one of the supported values.
        """
        return self._get_samples(which).cpu().numpy()

    @torch.inference_mode()
    def run_mcmc_refinement(
        self,
        num_chains: int = 512,
        seed: Optional[int] = 0,
        init_strategy: 'str' = 'random', 
        **mcmc_kwargs,
    ):
        """
        Run MCMC refinement of the exact reflectivity posterior.

        Initial walker positions are chosen from the raw neural posterior
        sample set and are then used to initialize an MCMC run targeting the
        exact reflectivity log posterior stored in this result object.

        Args:
            num_chains:
                Number of initial chains or walkers.
            seed:
                Random seed used when subsampling raw samples.
            init_strategy:
                Strategy used to choose the initial walkers from the raw sample
                set. Supported values are:

                - ``"random"``: choose a random subset of raw samples
                - ``"topn"`` or ``"top_num_chains"``: choose the raw samples
                with the highest exact log posterior values
            **mcmc_kwargs:
                Additional keyword arguments forwarded to the MCMC runner.

        Returns:
            The raw object returned by the MCMC runner.

        Notes:
            After a successful run, the flattened MCMC samples are stored in
            :attr:`mcmc_samples` and become available via ``which="mcmc"`` in
            inherited sample-based methods.
        """
        raw = torch.as_tensor(self.raw_samples_array, device=self.device)

        if str(init_strategy).lower() == "random":
            init = self._subsample_rows(raw, max_rows=num_chains, seed=seed)
        elif str(init_strategy).lower() in ("topn", "top_num_chains"):
            logp_raw = self.log_prob_fn(raw)

            logp = logp_raw.clone()
            bad = ~torch.isfinite(logp)
            if bad.any():
                logp[bad] = -float("inf")

            k = min(int(num_chains), int(raw.shape[0]))
            top_idx = torch.topk(logp, k=k, largest=True).indices
            init = raw[top_idx]  # shape (k, dim)

            if k < num_chains:
                raise ValueError("Requested more chains than raw samples")
        else:
            raise ValueError("init_strategy must be 'random' or 'topn'.")

        pb = torch.as_tensor(self.prior_bounds, device=self.device, dtype=init.dtype)
        fixed_mask = (pb[:, 1] - pb[:, 0]).abs() <= 1e-7
        if fixed_mask.any():
            init[:, fixed_mask] = pb[fixed_mask, 0]

        mcmc_result = run_mcmc(
            init_coords=init,
            log_prob_fn=self.log_prob_fn,
            **mcmc_kwargs,
        )

        self.mcmc_samples = mcmc_result[0].get_chain(flat=True).cpu().numpy()
        return mcmc_result
    
    @torch.inference_mode()
    def run_mcmc_from_prior(
        self,
        num_chains: int = 512,
        **mcmc_kwargs,
    ):
        """
        Run MCMC starting from uniformly sampled points inside the prior box.

        This method is an alternative to :meth:`run_mcmc_refinement` when you
        want to initialize MCMC independently of the raw neural posterior
        samples.

        Args:
            num_chains:
                Number of initial chains or walkers.
            **mcmc_kwargs:
                Additional keyword arguments forwarded to the MCMC runner.

        Returns:
            The raw object returned by the MCMC runner.

        Notes:
            After a successful run, the flattened MCMC samples are stored in
            :attr:`mcmc_samples`.
        """
        pb = torch.as_tensor(self.prior_bounds, device=self.device, dtype=torch.float32)
        min_bounds = pb[:, 0]
        max_bounds = pb[:, 1]

        u = torch.rand(num_chains, min_bounds.numel(), device=self.device, dtype=torch.float32)
        init = u * (max_bounds - min_bounds) + min_bounds

        mcmc_result = run_mcmc(
            init_coords=init,
            log_prob_fn=self.log_prob_fn,
            **mcmc_kwargs,
        )

        self.mcmc_samples = mcmc_result[0].get_chain(flat=True).cpu().numpy()
        return mcmc_result
    
    def run_neural_snis(
        self,
        target_neff: int = 500,
        max_num_samples: int = 2**20,
        batch_size: int = 4096,
        verbose: bool = True,
        resample: bool = True,
    ) -> None:
        """
        Run self-normalized importance sampling (SNIS) using the neural
        posterior as the proposal distribution.

        Proposal samples are repeatedly drawn from the inference model together
        with their proposal log densities. Each proposal sample is then scored
        under the exact reflectivity posterior, and the importance-sampling
        pool is updated until the target effective sample size is reached or
        the proposal budget is exhausted.

        Optionally, the weighted SNIS pool is then resampled to produce an
        approximately unweighted sample set of size ``target_neff``.

        Args:
            target_neff:
                Target effective sample size used as the stopping criterion.
            max_num_samples:
                Maximum total number of proposal samples drawn from the neural
                proposal.
            batch_size:
                Number of proposal samples drawn per iteration.
            verbose:
                If ``True``, display progress information.
            resample:
                If ``True``, call :meth:`resample_snis` at the end of the run
                and store an approximately unweighted SNIS sample set.

        Raises:
            RuntimeError:
                If the neural proposal path does not return proposal log
                probabilities.
            ValueError:
                If the returned proposal log probabilities contain invalid
                values.

        Notes:
            This method requires a proposal sampler that can return the log
            density of its own samples in physical parameter space. In
            practice, this means that the underlying inference path must
            support ``enable_importance_sampling=True`` and return
            ``unscaled_log_prob``.
        """
        self.importance_sampling.reset()
        self.snis_samples = None
        self.snis_log_weights = None

        n = 0
        neff = 0

        from tqdm.notebook import tqdm, trange

        pbar = tqdm(
            total=target_neff, disable=not verbose, desc="SNIS", unit=" Neff"
        )

        while self.importance_sampling.neff < target_neff:
            out = self.inference_model.preprocess_and_sample(
                reflectivity_curve=self.data.R,
                q_values=self.data.q,
                prior_bounds=self.prior_bounds.copy(),
                sigmas=self.data.dR,
                q_resolution=self.data.dq,
                ambient_sld=(float(self.ambient_sld.squeeze().cpu().item()) if self.ambient_sld is not None else None),

                num_samples=batch_size,

                enable_importance_sampling=True,
                clip_prediction=False,
                calc_sampled_curves=False,
                calc_log_likelihoods=False,
                return_result_as_dict=True,
            )

            if "unscaled_log_prob" not in out:
                raise RuntimeError(
                    "Neural-SNIS requires proposal log-probabilities. "
                    "Expected `unscaled_log_prob` in the sampling output. "
                    "Ensure your trainer path implements sample_and_log_prob and that "
                    "`enable_importance_sampling=True` computes and returns unscaled_log_prob."
                )

            thetas = out["predicted_params_array"]
            log_probs = out["unscaled_log_prob"]

            if not torch.isfinite(log_probs).all():
                raise ValueError("Log probabilities contain NaNs or Infs. ")

            thetas = torch.as_tensor(thetas, device=self.device, dtype=torch.float32)

            log_posteriors = self.log_prob_fn(thetas)  # log_likelihood + log_prior

            self.importance_sampling.update(log_posteriors, log_probs, thetas)

            pbar.update(round(self.importance_sampling.neff - neff))
            neff = self.importance_sampling.neff
            n += batch_size
            

            # ###############
            # log_probs_t = torch.as_tensor(log_probs, device=self.device, dtype=torch.float32)
            # log_post_t = torch.as_tensor(log_posteriors, device=self.device, dtype=torch.float32)
            # finite = torch.isfinite(log_probs_t) & torch.isfinite(log_post_t)

            # if finite.any():
            #     logw = log_post_t[finite] - log_probs_t[finite]
            #     logw_centered = logw - logw.max()
            #     w = torch.exp(logw_centered)
            #     w = w / w.sum()
            #     neff_batch = 1.0 / torch.sum(w**2)

            #     print(
            #         f"[SNIS debug] "
            #         f"finite={finite.float().mean().item():.3f}  "
            #         f"logq(mean/std)=({log_probs_t[finite].mean().item():.2f}, {log_probs_t[finite].std().item():.2f})  "
            #         f"logp(mean/std)=({log_post_t[finite].mean().item():.2f}, {log_post_t[finite].std().item():.2f})  "
            #         f"logw(std)={logw.std().item():.2f}  "
            #         f"batch_neff={neff_batch.item():.2f}/{logw.numel()}"
            #     )
            # else:
            #     print("[SNIS debug] no finite overlap between proposal and target scores")

            if n >= max_num_samples:
                if verbose:
                    print(f"Maximum number of samples reached. Neff = {neff:.1f}")
                break
        
        if resample:
            self.resample_snis(num_samples=target_neff)

        self.print_snis_summary()
    
    @torch.inference_mode()
    def resample_snis(self, num_samples: int):
        """
        Draw approximately unweighted posterior samples from the current SNIS pool.

        This method performs sampling-importance resampling using the weighted
        proposal pool accumulated by :meth:`run_neural_snis` or
        :meth:`run_prior_snis`.

        Args:
            num_samples:
                Number of resampled posterior samples to draw.

        Returns:
            A NumPy array of shape ``(num_samples, n_params)`` containing
            approximately unweighted posterior samples in physical parameter
            space.

        Raises:
            RuntimeError:
                If no SNIS pool is currently available.
        """
        if self.importance_sampling.size == 0:
            raise RuntimeError("No SNIS pool available. Run `run_neural_snis()` first.")

        params, logw = self.importance_sampling.sample(num_samples)
        self.snis_samples = params.numpy()
        self.snis_log_weights = logw.numpy()
        return self.snis_samples
    
    def get_snis_sample_eff_estimation(self) -> float:
        """
        Estimate the empirical efficiency of the current SNIS pool.

        The efficiency is defined as the effective sample size divided by the
        total number of proposal samples stored in the importance-sampling
        pool.

        Returns:
            A float equal to ``Neff / N``.

        Raises:
            RuntimeError:
                If no SNIS pool is currently available.
        """
        N = self.importance_sampling.size
        if N == 0:
            raise RuntimeError("No SNIS pool available. Run `run_neural_snis()` first.")
        return float(self.importance_sampling.neff) / N
    
    def print_snis_summary(self):
        """
        Print a short summary of the current SNIS pool.

        The printed summary includes:

        - the number of stored proposal samples
        - the effective sample size
        - the empirical efficiency ``Neff / N``
        - the average number of proposals required per effective sample
        """
        N = self.importance_sampling.size
        neff = self.importance_sampling.neff
        eff = self.get_snis_sample_eff_estimation() if N > 0 else 0.0
        inv_eff = (N / neff) if neff > 0 else float("inf")

        print("SNIS summary")
        print(f"- N proposals              : {N}")
        print(f"- Neff                     : {neff:.2f}")
        print(f"- efficiency (Neff / N)    : {eff:.4f}")
        print(f"- proposals per eff. sample: {inv_eff:.2f}")

    def run_prior_snis(
        self,
        num_samples: int = 100_000,
        batch_size: int = 4096,
        verbose: bool = True,
    ):
        """
        Run self-normalized importance sampling using the uniform box prior as
        the proposal distribution.

        This method is mainly useful as a baseline against neural-SNIS. The
        proposal is the uniform prior over the stored physical prior bounds,
        and each proposal sample is scored under the exact reflectivity
        posterior.

        Args:
            num_samples:
                Total number of prior proposal samples to draw.
            batch_size:
                Number of proposal samples evaluated per iteration.
            verbose:
                If ``True``, display progress information.

        Notes:
            The resulting weighted proposal pool is stored in the internal
            importance-sampling object and can be resampled later with
            :meth:`resample_snis`.
        """

        self.importance_sampling.reset()
        self.snis_samples = None
        self.snis_log_weights = None

        pb = torch.as_tensor(self.prior_bounds, device=self.device, dtype=torch.float32)
        min_bounds = pb[:, 0]
        max_bounds = pb[:, 1]

        widths = max_bounds - min_bounds
        fixed = widths.abs() <= 1e-7
        safe_widths = widths.clone()
        safe_widths[fixed] = 1.0

        log_q_const = -torch.sum(torch.log(safe_widths))

        from tqdm.notebook import tqdm

        n_done = 0

        pbar = tqdm(
            total=num_samples,
            disable=not verbose,
            desc="Prior-IS",
            unit=" samples",
        )

        while n_done < num_samples:
            n = min(batch_size, num_samples - n_done)

            u = torch.rand(n, min_bounds.numel(), device=self.device, dtype=torch.float32)
            thetas = u * (max_bounds - min_bounds) + min_bounds

            if fixed.any():
                thetas[:, fixed] = min_bounds[fixed]

            log_q = torch.full((n,), log_q_const, device=self.device, dtype=torch.float32)
            log_p = self.log_prob_fn(thetas)

            self.importance_sampling.update(log_p, log_q, thetas)

            n_done += n
            pbar.update(n)

            neff_now = float(self.importance_sampling.neff)
            eff_now = neff_now / max(1, self.importance_sampling.size)

            pbar.set_description(
                f"Prior-IS Neff={neff_now:.2f} Eff={eff_now:.4e}"
            )

        pbar.close()

        self.print_snis_summary()

    @torch.inference_mode()
    def map_estimate_raw(
        self,
        return_logp: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Return an approximate MAP estimate from the raw neural posterior samples.

        This method evaluates the exact reflectivity log posterior on the raw
        sample set and returns the sample with the highest value.

        Args:
            return_logp:
                If ``True``, also return the corresponding log posterior value.

        Returns:
            If ``return_logp=False``, a NumPy array of shape ``(n_params,)``
            containing the approximate MAP estimate.

            If ``return_logp=True``, returns a tuple
            ``(theta_map, logp_map)``.
        """

        params = torch.as_tensor(self.raw_samples_array, device=self.device, dtype=torch.float32)

        logp = self.log_prob_fn(params)
        idx = torch.argmax(logp)
        theta = params[idx].detach().cpu().numpy()
        lp = float(logp[idx].detach().cpu().item())
        return (theta, lp) if return_logp else theta

    def __repr__(self):
        raw_n = int(self.raw_samples_array.shape[0])
        dim = int(self.raw_samples_array.shape[1])
        mcmc = None if self.mcmc_samples is None else int(self.mcmc_samples.shape[0])
        dq_str = "None" if self.data.dq is None else ("float" if np.isscalar(self.data.dq) else "pointwise")
        return (
            f"PosteriorInferenceResult(dim={dim}, N_raw={raw_n}, N_mcmc={mcmc}, "
            f"dq={dq_str}, ambient_sld={'yes' if self.ambient_sld is not None else 'no'})"
        )

    def summary(self):
        """
        Print a short textual summary of this posterior-inference result.

        The summary prints the object representation, including the parameter
        dimensionality and the sizes of any stored raw and MCMC sample sets.
        """
        print(self.__repr__())

    def _serialize_for_npz(self):
        arrays = {}

        arrays["q"] = np.asarray(self.data.q)
        arrays["R"] = np.asarray(self.data.R)

        if self.data.dR is not None:
            arrays["dR"] = np.asarray(self.data.dR)

        if self.data.dq is not None:
            arrays["dq"] = np.asarray(self.data.dq)

        arrays["prior_bounds"] = np.asarray(self.prior_bounds)
        arrays["raw_samples_array"] = np.asarray(self.raw_samples_array)

        if self.mcmc_samples is not None:
            arrays["mcmc_samples"] = np.asarray(self.mcmc_samples)

        if self.snis_samples is not None:
            arrays["snis_samples"] = np.asarray(self.snis_samples)
        if self.snis_log_weights is not None:
            arrays["snis_log_weights"] = np.asarray(self.snis_log_weights)

        meta = {
            "ambient_sld": None if self.ambient_sld is None else float(self.ambient_sld.squeeze().cpu().item()),
            "param_names": list(self.param_names) if self.param_names is not None else None,

            "model_info": {
                "config_name": getattr(self.inference_model, "config_name", None),
                "model_name": getattr(self.inference_model, "model_name", None),
                "root_dir": getattr(self.inference_model, "root_dir", None),
                "weights_format": getattr(self.inference_model, "weights_format", None),
                "repo_id": getattr(self.inference_model, "repo_id", None),
            },
        }

        return meta, arrays

    @staticmethod
    def _construct_from_npz(meta, z, inference_model=None, device="cpu"):
        if inference_model is None:
            mi = meta.get("model_info", None)
            if not mi or not mi.get("config_name"):
                raise ValueError("Need inference_model or metadata['model_info'] with at least config_name.")
            
            from reflectorch.inference.inference_model import InferenceModel
            inference_model = InferenceModel(
                config_name=mi.get("config_name"),
                model_name=mi.get("model_name"),
                root_dir=mi.get("root_dir"),
                weights_format=mi.get("weights_format", "safetensors"),
                repo_id=mi.get("repo_id"),
                device=device,
            )

        q = z["q"]
        R = z["R"]
        dR = z["dR"] if "dR" in z.files else None
        dq = z["dq"] if "dq" in z.files else None

        data = ReflectivityData(q=q, R=R, dR=dR, dq=dq)

        pred = {
            "predicted_params_array": z["raw_samples_array"],
            "param_names": meta.get("param_names", None),
        }

        res = PosteriorInferenceResult(
            inference_model=inference_model,
            prediction_dict=pred,
            data=data,
            prior_bounds=z["prior_bounds"],
            ambient_sld=meta.get("ambient_sld", None),
            device=device,
        )

        if "mcmc_samples" in z.files:
            res.mcmc_samples = np.asarray(z["mcmc_samples"])
        if "snis_samples" in z.files:
            res.snis_samples = np.asarray(z["snis_samples"])
        if "snis_log_weights" in z.files:
            res.snis_log_weights = np.asarray(z["snis_log_weights"])

        return res

def _json_to_npbytes(d: dict) -> np.ndarray:
    b = json.dumps(d, ensure_ascii=False).encode("utf-8")
    return np.frombuffer(b, dtype=np.uint8)

def _npbytes_to_json(arr: np.ndarray) -> dict:
    if arr.dtype == np.uint8:
        b = arr.tobytes()
    else:
        b = bytes(arr.tolist()) if arr.ndim else bytes(arr)
    return json.loads(b.decode("utf-8"))