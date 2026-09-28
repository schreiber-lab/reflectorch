# -*- coding: utf-8 -*-
"""
reflectorch_nexus_saver.py
==========================

NeXus/HDF5 writer for reflectorch inference results whose output structure
matches EXACTLY the curated multiscan file layout
(``xrr_C60_DIP_multiscan.h5``: entry_0000 -> data/{analysis, processed_data,
raw_data}, instrument, process/{footprint_correction, reflectorch}, sample,
user).

Location in the repo: ``reflectorch/inference/reflectorch_nexus_saver.py``
and export it in ``reflectorch/inference/__init__.py``::

    from reflectorch.inference.reflectorch_nexus_saver import write_reflectorch_nexus, PARAM_SPECS

The writer is deliberately explicit: every group, dataset, dtype and
attribute of the curated schema is spelled out below, so the produced file is
structurally identical to the reference (verified with a recursive
structure diff).

Main entry point
----------------
:func:`write_reflectorch_nexus` - one call, writes the whole file.

The physical parameters are described by :data:`PARAM_SPECS` (dataset name,
long_name, units, and the corresponding ``<name>_lower`` / ``<name>_upper``
bound dataset names). Override/extend it for other models.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import h5py
except ImportError as _e:  # h5py is not a core reflectorch dependency
    raise ImportError("Saving to NeXus requires h5py: pip install h5py") from _e

try:
    from importlib.metadata import version as _pkg_version
    REFLECTORCH_VERSION_DEFAULT = _pkg_version("reflectorch")
except Exception:  # pragma: no cover
    REFLECTORCH_VERSION_DEFAULT = "unknown"

__all__ = ["write_reflectorch_nexus", "PARAM_SPECS",
           "DEFAULT_SAMPLE_INFO", "DEFAULT_USER_INFO",
           "DEFAULT_INSTRUMENT_INFO", "DEFAULT_FOOTPRINT_INFO"]


# ─────────────────────────────────────────────────────────────────────────────
# Parameter schema
# (dataset name under data/analysis, long_name, units,
#  bound dataset base name under process/reflectorch/input_parameters/bounds,
#  units attribute attached to the bound datasets - None means no units attr,
#  exactly as in the curated reference file)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_SPECS: List[dict] = [
    dict(name="thick",    long_name="Film Thickness",
         units="angstrom",     bound_name="film_thickness", bound_units=None),
    dict(name="rough",    long_name="Film Roughness",
         units="angstrom",     bound_name="film_rough",     bound_units=None),
    dict(name="SLD",      long_name="Scattering Length Density",
         units="1e-6 Ang^-2",  bound_name="film_SLD",       bound_units=None),
    dict(name="Si_rough", long_name="Substrate Roughness",
         units="angstrom",     bound_name="Si_rough",       bound_units=None),
    dict(name="r_scale",  long_name="Intensity scale nuisance parameter",
         units="dimensionless", bound_name="r_scale",
         bound_units="dimensionless"),
    dict(name="q_shift",  long_name="Q-shift nuisance parameter",
         units="1/angstrom",   bound_name="q_shift",
         bound_units="1/angstrom"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Default metadata blocks (values of the curated C60:DIP reference file).
# Pass your own dicts to override any field.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_INFO = dict(
    name="C60:DIP Gradient Thin Film",
    description=("Gradient thin film of C60 and Diindenoperylene (DIP) on "
                 "Silicon. The continuous lateral composition gradient "
                 "smoothly changes from purely C60 at one end of the sample "
                 "to purely DIP at the opposite end."),
    length=52.0,      # mm
    width=11.0,       # mm
    height=0.5,       # mm
    stack="air | C60:DIP | Si",
    sample_history="",
    materials=[
        dict(group="layer_1_film", name="C60:DIP Blend",
             chemical_formula="C60 + C32H16",
             description="Mixed organic gradient layer",
             thickness=2.5e-08),           # meters
        dict(group="layer_2_Si", name="Silicon",
             chemical_formula="Si",
             description="Wafer substrate (100)",
             thickness=0.0005),            # meters
    ],
)

DEFAULT_USER_INFO = dict(
    name="Dmitry Lapkin",
    role="Principal Investigator",
    affiliation="University of Tübingen",
    email="dmitry.lapkin@uni-tuebingen.de",
    ORCID="0000-0000-0000-0000",
    telephone_number="",
)

DEFAULT_INSTRUMENT_INFO = dict(
    name="GE Diffractometer System XRD 3003 TT",
    angle_of_incidence=0.075,               # degrees (nominal)
    count_time=1.0,                          # seconds
    detector_description="Point detector",
    monochromator_name=("Ni/C multilayer mirror and a germanium "
                        "channel-cut crystal"),
    wavelength=1.5405452421779864,           # Angstrom (Cu K-alpha equiv.)
    source_name="GE Diffractometer System XRD 3003 TT",
    source_probe="X-ray",
    source_type="Diffractometer",
    notes=dict(DOI="", beamtime_id="",
               description=("XRR of C60:DIP gradient thin film, 51 positions, "
                            "lab diffractometer"),
               proposal_id="", title=""),
)

DEFAULT_FOOTPRINT_INFO = dict(
    beam_shape="box",
    beam_width=0.5,                          # mm
    description="Footprint correction, low-angle trimming, normalisation to 1.",
    sample_size="52 x 11",                   # mm
    trimmed_points=10,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iso(t: str) -> str:
    """SPEC '#D' date ('Mon May 06 13:50:50 2024') -> ISO '2024-05-06T13:50:50'.
    Already-ISO strings pass through unchanged."""
    for fmt in ("%a %b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(t.strip(), fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return t


def _param_structured_dtype(material_names: Optional[Sequence[str]]) -> np.dtype:
    fields = [("Predicted", "f8"), ("Polished", "f8")]
    if material_names:
        fields += [(str(m), "f8") for m in material_names]
    return np.dtype(fields)


# ─────────────────────────────────────────────────────────────────────────────
# main writer
# ─────────────────────────────────────────────────────────────────────────────

def write_reflectorch_nexus(
        output_file: Union[str, Path],
        *,
        # ── processed data ──────────────────────────────────────────────────
        q: np.ndarray,                       # (N_q,)
        R: np.ndarray,                       # (N_scans, N_q)
        dR: np.ndarray,                      # (N_scans, N_q)
        dq: float = 0.0,
        # ── inference results ───────────────────────────────────────────────
        predicted: Dict[str, np.ndarray],    # {param_name: (N_scans,)}
        polished: Dict[str, np.ndarray],     # {param_name: (N_scans,)}
        prior_bounds: Dict[str, tuple],      # {param_name: (lower, upper)}
        coord: np.ndarray,                   # (N_scans,) sample coordinate [mm]
        model_yaml: str = "",                # raw YAML of the model config
        reflectorch_version: str = None,
        param_specs: List[dict] = None,
        # ── composition columns (optional) ──────────────────────────────────
        material_names: Optional[Sequence[str]] = ("C60", "DIP"),
        fractions: Optional[np.ndarray] = None,   # (N_scans, n_materials), rows aligned with R
        volume_ratio_convention: str = (
            "C60 + DIP = 1.0. C60=0, DIP=1 at most-negative coord (pure DIP); "
            "C60=0.5, DIP=0.5 at centre (1:1); C60=1, DIP=0 at most-positive "
            "coord (pure C60)."),
        # ── raw data (optional but part of the curated schema) ──────────────
        raw_intensity: Optional[np.ndarray] = None,   # (N_scans, N_q) Specular counts
        two_theta: Optional[np.ndarray] = None,       # (N_scans, N_q) Theta+TwoTheta [deg]
        scan_numbers: Optional[np.ndarray] = None,    # (N_scans,)
        timestamps: Optional[Sequence[str]] = None,   # (N_scans,) SPEC '#D' strings
        y_motor: Optional[np.ndarray] = None,         # (N_scans,) [mm]
        monitor: int = 1,
        # ── file-level metadata ─────────────────────────────────────────────
        filename: str = "",
        title: str = "XRR Multiscan of C60:DIP Gradient Thin Film",
        definition: str = "NXxrd",
        start_time: str = None,              # defaults to first timestamp
        end_time: str = None,                # defaults to last timestamp
        sample_info: dict = None,
        user_info: dict = None,
        instrument_info: dict = None,
        footprint_info: dict = None,
        entry_idx: int = 0,
) -> Path:
    """Write reflectorch inference results to a NeXus file with the exact
    curated multiscan schema.

    ``predicted`` / ``polished`` / ``prior_bounds`` are keyed by the parameter
    names of ``param_specs`` (default: thick, rough, SLD, Si_rough, r_scale,
    q_shift). Missing polished values may be NaN arrays.
    """
    output_file = Path(output_file)
    param_specs = param_specs if param_specs is not None else PARAM_SPECS
    reflectorch_version = reflectorch_version or REFLECTORCH_VERSION_DEFAULT

    smpl = {**DEFAULT_SAMPLE_INFO, **(sample_info or {})}
    user = {**DEFAULT_USER_INFO, **(user_info or {})}
    inst = {**DEFAULT_INSTRUMENT_INFO, **(instrument_info or {})}
    fp = {**DEFAULT_FOOTPRINT_INFO, **(footprint_info or {})}

    q = np.asarray(q, dtype=float)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    dR = np.atleast_2d(np.asarray(dR, dtype=float))
    coord = np.asarray(coord, dtype=float)
    n_scans, n_q = R.shape

    if timestamps is not None:
        if start_time is None:
            start_time = _iso(str(timestamps[0]))
        if end_time is None:
            end_time = _iso(str(timestamps[-1]))
    start_time = start_time or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_time or start_time

    with h5py.File(output_file, "w") as f:
        entry_name = f"entry_{entry_idx:04d}"
        f.attrs["default"] = entry_name
        entry = f.create_group(entry_name)
        entry.attrs.update({"NX_class": "NXentry", "default": "data"})

        entry.create_dataset("definition", data=definition)
        entry.create_dataset("title", data=title)
        entry.create_dataset("start_time", data=start_time)
        entry.create_dataset("end_time", data=end_time)

        # ═════════════════════════ data ═════════════════════════════════════
        data = entry.create_group("data")
        data.attrs.update({"NX_class": "NXcollection", "default": "processed_data"})
        data.create_dataset("filename", data=str(filename))

        # ── data/processed_data ──────────────────────────────────────────────
        proc = data.create_group("processed_data")
        proc.attrs.update({
            "NX_class": "NXdata",
            "axes": np.array([".", "q"], dtype=object),
            "interpretation": "spectrum",
            "q_indices": np.array([1], dtype=np.uint32),
            "signal": "R",
        })
        ds = proc.create_dataset("q", data=q)
        ds.attrs.update({"long_name": "Momentum transfer Q", "units": "1/angstrom"})
        ds = proc.create_dataset("R", data=R)
        ds.attrs["long_name"] = "Reflectivity (footprint corrected, normalised)"
        ds = proc.create_dataset("dR", data=dR)
        ds.attrs["long_name"] = "Errors (Poisson, propagated)"
        ds = proc.create_dataset("dq", data=float(dq))
        ds.attrs.update({"long_name": "Q-shift correction (dq)", "units": "1/angstrom"})

        if material_names and fractions is not None:
            fractions = np.atleast_2d(np.asarray(fractions, dtype=float))
            vr_dt = np.dtype([(str(m), "f8") for m in material_names])
            vr = np.zeros(n_scans, dtype=vr_dt)
            # stored in ascending-coord order (as in the reference file)
            order = np.argsort(coord)
            for j, m in enumerate(material_names):
                vr[str(m)] = fractions[order, j]
            ds = proc.create_dataset("volume_ratio", data=vr)
            ds.attrs.update({
                "convention": volume_ratio_convention,
                "long_name": f"{material_names[0]}:{material_names[1]} volume ratio"
                             if len(material_names) == 2 else "volume ratio",
            })

        # ── data/raw_data ────────────────────────────────────────────────────
        raw = data.create_group("raw_data")
        raw.attrs.update({
            "NX_class": "NXdata",
            "axes": np.array([".", "pixels"], dtype=object),
            "pixels_indices": np.array([1], dtype=np.uint32),
            "signal": "Intensity",
        })
        if raw_intensity is not None:
            ds = raw.create_dataset("Intensity",
                                    data=np.atleast_2d(np.asarray(raw_intensity, float)))
            ds.attrs["long_name"] = ("Specular intensity (raw counts, "
                                     "Specular column from SPEC)")
        ds = raw.create_dataset("monitor", data=np.int64(monitor))
        ds.attrs["long_name"] = "Monitor counts"
        ds = raw.create_dataset("pixels", data=np.arange(n_q, dtype=np.int32))
        ds.attrs.update({"long_name": "Pixel / Point index", "units": "pixel"})
        if scan_numbers is not None:
            ds = raw.create_dataset("scan_numbers",
                                    data=np.asarray(scan_numbers, dtype=np.int64))
            ds.attrs["long_name"] = "SPEC scan numbers (low-q scan of each pair)"
        if timestamps is not None:
            ts = np.array([str(t).encode("utf-8") for t in timestamps], dtype="S24")
            ds = raw.create_dataset("timestamps", data=ts)
            ds.attrs["long_name"] = "Scan start date/time (SPEC #D)"
        if two_theta is not None:
            ds = raw.create_dataset("two_theta",
                                    data=np.atleast_2d(np.asarray(two_theta, float)))
            ds.attrs.update({"long_name": "Total scattering angle (Theta + Two Theta)",
                             "units": "degrees"})
        if y_motor is not None:
            ds = raw.create_dataset("y_motor", data=np.asarray(y_motor, dtype=float))
            ds.attrs.update({"long_name": "Y motor position", "units": "mm"})

        # ── data/analysis ────────────────────────────────────────────────────
        ana = data.create_group("analysis")
        ana.attrs.update({
            "NX_class": "NXcollection",
            "description": "Reflectorch inference results (Predicted & Polished)",
        })
        dt = _param_structured_dtype(material_names if fractions is not None else None)
        for spec_ in param_specs:
            name = spec_["name"]
            if name not in predicted:
                raise KeyError(f"predicted values for parameter '{name}' missing "
                               f"(keys given: {list(predicted)})")
            arr = np.zeros(n_scans, dtype=dt)
            arr["Predicted"] = np.asarray(predicted[name], dtype=float)
            arr["Polished"] = (np.asarray(polished[name], dtype=float)
                               if polished and name in polished
                               else np.full(n_scans, np.nan))
            if fractions is not None and material_names:
                for j, m in enumerate(material_names):
                    arr[str(m)] = fractions[:, j]
            ds = ana.create_dataset(name, data=arr)
            ds.attrs.update({"long_name": spec_["long_name"], "units": spec_["units"]})
        ds = ana.create_dataset("coord", data=coord)
        ds.attrs.update({"long_name": "Sample Coordinate", "units": "mm"})

        # ═════════════════════════ instrument ═══════════════════════════════
        ig = entry.create_group("instrument")
        ig.attrs["NX_class"] = "NXinstrument"
        ig.create_dataset("name", data=str(inst["name"]))
        ds = ig.create_dataset("angle_of_incidence",
                               data=float(inst["angle_of_incidence"]))
        ds.attrs.update({"long_name": "Nominal angle of incidence",
                         "units": "degrees"})

        det = ig.create_group("detector")
        det.attrs.update({"NX_class": "NXdetector",
                          "description": inst["detector_description"]})
        ds = det.create_dataset("count_time", data=float(inst["count_time"]))
        ds.attrs.update({"long_name": "Base exposure time per point",
                         "units": "seconds"})

        mono = ig.create_group("monochromator")
        mono.attrs["NX_class"] = "NXmonochromator"
        mono.create_dataset("name", data=np.bytes_(inst["monochromator_name"]))
        ds = mono.create_dataset("wavelength", data=float(inst["wavelength"]))
        ds.attrs.update({"long_name": "Incident X-ray wavelength "
                                      "(Cu K-alpha equivalent)",
                         "units": "m"})  # NB: attr kept as in the curated
        #                                    reference file (value is Angstrom)

        src = ig.create_group("source")
        src.attrs["NX_class"] = "NXsource"
        src.create_dataset("name", data=str(inst["source_name"]))
        src.create_dataset("probe", data=str(inst["source_probe"]))
        src.create_dataset("type", data=str(inst["source_type"]))
        notes = src.create_group("notes")
        notes.attrs.update({"NX_class": "NXnote",
                            "description": "Proposal and beamtime information"})
        for key in ("DOI", "beamtime_id", "description", "proposal_id", "title"):
            notes.create_dataset(key, data=str(inst["notes"].get(key, "")))

        # ═════════════════════════ process ══════════════════════════════════
        pg = entry.create_group("process")
        pg.attrs["NX_class"] = "NXprocess"

        # footprint correction (data reduction provenance)
        fpg = pg.create_group("footprint_correction")
        fpg.attrs["NX_class"] = "NXprocess"
        fpg.create_dataset("beam_shape", data=np.bytes_(fp["beam_shape"]))
        ds = fpg.create_dataset("beam_width", data=float(fp["beam_width"]))
        ds.attrs["units"] = "mm"
        fpg.create_dataset("description", data=np.bytes_(fp["description"]))
        ds = fpg.create_dataset("sample_size", data=np.bytes_(fp["sample_size"]))
        ds.attrs["units"] = "mm"
        ds = fpg.create_dataset("trimmed_points", data=np.int64(fp["trimmed_points"]))
        ds.attrs["description"] = "Number of low-angle points removed"

        # reflectorch provenance
        rg = pg.create_group("reflectorch")
        rg.attrs["NX_class"] = "NXprocess"
        rg.create_dataset("program", data=np.bytes_("reflectorch"))
        ds = rg.create_dataset("version", data=np.bytes_(reflectorch_version))
        ds.attrs["description"] = "Reflectorch version used for inference"
        ds = rg.create_dataset("model", data=str(model_yaml))
        ds.attrs.update({"NX_class": "NXnote",
                         "description": "Reflectorch model configuration (raw YAML)",
                         "format": "yaml"})

        ipg = rg.create_group("input_parameters")
        ipg.attrs.update({"NX_class": "NXcollection",
                          "description": ("Fitting input bounds used in "
                                          "reflectorch analysis")})
        bg = ipg.create_group("bounds")
        bg.attrs.update({"NX_class": "NXcollection",
                         "units_SLD": "1e-6 Ang^-2",
                         "units_roughness": "Angstrom",
                         "units_thickness": "Angstrom"})
        for spec_ in param_specs:
            name = spec_["name"]
            if name not in prior_bounds:
                raise KeyError(f"prior bounds for parameter '{name}' missing")
            lo, hi = prior_bounds[name]
            for suffix, val in (("lower", lo), ("upper", hi)):
                ds = bg.create_dataset(f"{spec_['bound_name']}_{suffix}",
                                       data=float(val))
                if spec_.get("bound_units"):
                    ds.attrs["units"] = spec_["bound_units"]

        # ═════════════════════════ sample ═══════════════════════════════════
        sg = entry.create_group("sample")
        sg.attrs["NX_class"] = "NXsample"
        sg.create_dataset("name", data=str(smpl["name"]))
        sg.create_dataset("description", data=str(smpl["description"]))
        for dim in ("height", "length", "width"):
            ds = sg.create_dataset(dim, data=float(smpl[dim]))
            ds.attrs["units"] = "mm"
        hist = sg.create_group("sample_history")
        hist.attrs["NX_class"] = "NXnote"
        hist.create_dataset("data", data=str(smpl.get("sample_history", "")))
        struct = sg.create_group("structure")
        struct.attrs["NX_class"] = "NXcollection"
        struct.create_dataset("stack", data=str(smpl["stack"]))
        mats = struct.create_group("materials")
        mats.attrs["NX_class"] = "NXcollection"
        for m in smpl["materials"]:
            mg = mats.create_group(m["group"])
            mg.attrs["NX_class"] = "NXsample_component"
            mg.create_dataset("chemical_formula", data=str(m["chemical_formula"]))
            mg.create_dataset("description", data=str(m["description"]))
            mg.create_dataset("name", data=str(m["name"]))
            ds = mg.create_dataset("thickness", data=float(m["thickness"]))
            ds.attrs.update({"type": "NX_FLOAT", "units": "meters"})

        # ═════════════════════════ user ═════════════════════════════════════
        ug = entry.create_group("user")
        ug.attrs["NX_class"] = "NXuser"
        for key in ("ORCID", "affiliation", "email", "name", "role",
                    "telephone_number"):
            ug.create_dataset(key, data=str(user.get(key, "")))

    print(f"NeXus file written -> {output_file.resolve()} [{entry_name}] "
          f"({n_scans} scans x {n_q} q-points, "
          f"{len(param_specs)} parameters)")
    return output_file
