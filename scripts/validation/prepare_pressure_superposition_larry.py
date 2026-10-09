#!/usr/bin/env python3
"""Prepare the existing synthetic pressure test on Larry's transient network.

Install at scripts/validation/prepare_pressure_superposition_larry.py.
This imports the user's existing validate_comsol_transient.py WITHOUT running
its main(). It reuses its station loader, validated vertical mapping and fixed
benchmark constants. It does not change or run the analytical physics, refit the
benchmark, or infer pressure from the COMSOL strain observations.

Only new configuration/provenance files are written. Raw data and demo_001 are
left unchanged. The synthetic pressure truth, prior and noise remain those in
the starter template; they are not measured uncertainties or field results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional, Sequence

import numpy as np


def resolve_input(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Keep inputs/outputs inside the repository for portability: {path}") from exc


def source_record(root: Path, path: Path) -> dict:
    return {"path": relative_path(root, path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_validation_module(root: Path, path: Path) -> ModuleType:
    # Execute in a fresh CLI process; reuse the actual checkout, not site-packages.
    for directory in (root, root / "src"):
        sys.path.insert(0, str(directory))
    name = "_avant_larry_validation_for_pressure_preparation"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load validation script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    required = (
        "load_strain_dataset", "load_station_metadata", "validate_vertical_mapping",
        "DEFAULT_STRAIN_FILE", "DEFAULT_STATIONS_FILE", "COMSOL_SURFACE_Z_M",
        "A", "B", "C", "H", "REFERENCE_X0", "REFERENCE_Y0", "THETA_DEG",
        "E", "ALPHA", "NU", "REFERENCE_PRESSURE_PA", "REPO_ROOT",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Validation script is missing required names: {missing}")
    if Path(module.REPO_ROOT).resolve() != root:
        raise ValueError("The validation script must belong to the selected repository.")
    return module


def prepare(
    root: Path,
    output_dir: Path,
    *,
    validation_script: Path = Path("scripts/validation/validate_comsol_transient.py"),
    template: Path = Path("configs/pressure_superposition_demo.json"),
    strain_file: Optional[Path] = None,
    stations_file: Optional[Path] = None,
    holdout_stations: Sequence[str] = ("S07", "S08"),
) -> dict:
    """Write a portable configuration using the existing validated data interface."""
    root = root.resolve()
    if not (root / "src" / "avant_model").is_dir():
        raise ValueError(f"Not an AVANT repository root: {root}")
    out = (output_dir if output_dir.is_absolute() else root / output_dir).resolve()
    relative_path(root, out)
    if out.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {out}")
    validation_path = resolve_input(root, validation_script)
    template_path = resolve_input(root, template)
    v = load_validation_module(root, validation_path)
    strain_path = resolve_input(root, strain_file if strain_file is not None else Path(v.DEFAULT_STRAIN_FILE))
    station_path = resolve_input(root, stations_file if stations_file is not None else Path(v.DEFAULT_STATIONS_FILE))

    # Read the COMSOL strain file only to establish station/channel metadata.
    # No observed strain values enter the synthetic pressure inversion.
    strain, time_s, names, components = v.load_strain_dataset(strain_path)
    raw = v.load_station_metadata(station_path, names)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Station names must be unique and nonempty.")
    # Reuse the established conversion and its existing consistency checks.
    depth = np.asarray(v.validate_vertical_mapping(raw), dtype=float)
    if depth.shape != (len(raw),) or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("The existing mapping did not return finite, nonnegative depths.")
    converted = raw.copy(deep=True)
    converted["z"] = depth

    cfg = json.loads(template_path.read_text(encoding="utf-8-sig"))
    cfg.pop("stations", None)
    cfg.pop("station_file", None)
    missing_components = set(cfg["components"]) - set(components)
    if missing_components:
        raise ValueError(f"Template channels absent from the COMSOL dataset: {sorted(missing_components)}")
    holdout = tuple(str(name).strip() for name in holdout_stations)
    if len(set(holdout)) != len(holdout) or not set(holdout) <= set(names):
        raise ValueError("Holdout stations must be unique names present in this dataset.")
    if len(holdout) >= len(names):
        raise ValueError("At least one station must remain for inference.")

    cfg["description"] = (
        "Same-model synthetic pressure recovery on Larry's COMSOL transient station network. "
        "Geometry, material properties and vertical conversion are reused from the existing "
        "transient validation script. Synthetic truth/prior/noise come from the starter, "
        "not from measured pressure or strain inversion."
    )
    cfg["geometry"] = {"a": float(v.A), "b": float(v.B), "c": float(v.C),
                       "h": float(v.H), "x0_prime": float(v.REFERENCE_X0),
                       "y0_prime": float(v.REFERENCE_Y0), "theta_deg": float(v.THETA_DEG)}
    cfg["elastic"] = {"E": float(v.E), "nu": float(v.NU), "alpha": float(v.ALPHA)}
    cfg["verification_pressure_pa"] = float(v.REFERENCE_PRESSURE_PA)
    cfg["holdout_stations"] = list(holdout)
    cfg["station_file"] = relative_path(root, out / "stations_analytical_depth.csv")
    inputs = {"validation_script": source_record(root, validation_path),
              "template": source_record(root, template_path),
              "raw_station_metadata": source_record(root, station_path),
              "strain_file_for_metadata_only": source_record(root, strain_path)}
    cfg["preparation_provenance"] = {
        "inputs": inputs,
        "station_z_in_this_configuration": "positive-downward depth, already converted; do not convert again",
        "comsol_surface_z_m": float(v.COMSOL_SURFACE_Z_M),
        "mapping_function": "validate_comsol_transient.validate_vertical_mapping",
        "synthetic_truth_prior_noise_seed": "Inherited unchanged from the starter template",
    }
    manifest = {
        "scope": "Configuration preparation only; no analytical calculation, fitting or observed-data inversion",
        "inputs": inputs,
        "strain_rows_read_for_metadata": int(len(strain)),
        "strain_time_range_s": [float(time_s[0]), float(time_s[-1])],
        "station_names": list(names), "available_strain_components": list(components),
        "geometry": cfg["geometry"], "elastic": cfg["elastic"],
        "surface_comsol_z_m": float(v.COMSOL_SURFACE_Z_M),
        "raw_comsol_z_m": raw["z"].to_numpy(dtype=float).tolist(),
        "analytical_depth_m": depth.tolist(),
        "source_dataset_files_modified": False,
        "raw_coordinate_snapshot": "stations_comsol_snapshot.csv",
        "converted_coordinate_snapshot": "stations_analytical_depth.csv",
    }
    # Serialize before reserving the new directory. Never overwrite prior work.
    cfg_text = json.dumps(cfg, indent=2, allow_nan=False) + "\n"
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    out.mkdir(parents=True, exist_ok=False)
    raw.to_csv(out / "stations_comsol_snapshot.csv", index=False)
    converted.to_csv(out / "stations_analytical_depth.csv", index=False)
    (out / "configuration.json").write_text(cfg_text, encoding="utf-8")
    (out / "preparation_manifest.json").write_text(manifest_text, encoding="utf-8")
    return {"configuration": out / "configuration.json", "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("configs/pressure_superposition/larry_network_001"))
    parser.add_argument("--validation-script", type=Path, default=Path("scripts/validation/validate_comsol_transient.py"))
    parser.add_argument("--template", type=Path, default=Path("configs/pressure_superposition_demo.json"))
    parser.add_argument("--strain-file", type=Path, default=None)
    parser.add_argument("--stations-file", type=Path, default=None)
    parser.add_argument("--holdout-stations", nargs="*", default=["S07", "S08"])
    args = parser.parse_args()
    result = prepare(args.repo_root, args.output_dir, validation_script=args.validation_script,
                     template=args.template, strain_file=args.strain_file,
                     stations_file=args.stations_file, holdout_stations=args.holdout_stations)
    m = result["manifest"]
    print("Prepared using the existing transient validation loader and mapping.")
    for name, raw_z, depth in zip(m["station_names"], m["raw_comsol_z_m"], m["analytical_depth_m"]):
        print(f"  {name}: COMSOL z={raw_z:g} m -> analytical depth={depth:g} m")
    print(f"Geometry: {m['geometry']}")
    print(f"Elastic properties: {m['elastic']}")
    print(f"Configuration: {result['configuration']}")
    print("Preparation only. The next run is a synthetic recoverability test, not observed-data inversion.")


if __name__ == "__main__":
    main()
