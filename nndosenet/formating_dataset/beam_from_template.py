"""
Beam Generation from Templates
===============================
Given a beam template and plan parameters (isocenter, gantry angles, arc type),
shift/rotate the template and resample it to the dose grid.

Algorithm:
1. Select template: Full Arc → full_arc template; Half Arc → half_arc; Static → single_beam
2. Shift: Translate template isocenter (at origin) to the plan's real isocenter in patient coords
3. Rotate (if not full arc): Rotate around the isocenter to match the beam's gantry angle
4. Resample: Resample to the dose grid (dose spacing, origin, direction, size) using nearest-neighbor
5. Sum: If multiple beams per plan, sum all beam contributions

All operations use nearest-neighbor interpolation to keep sharp integer edges.
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join

from nndosenet.formating_dataset.beam_templates import (
    _template_path,
    ensure_templates_exist,
    SPACING_MM,
)

# Module-level cache: template name → sitk.Image (loaded once, reused)
_template_cache = {}
_templates_checked = False


def _load_template(arc_type):
    """Load the appropriate template based on arc type string (cached).

    Only two categories:
    - "Static" → single_beam.nii.gz (rotated to match gantry angle)
    - Everything else (Full Arc, any arc) → full_arc.nii.gz (shifted only)

    Templates are loaded once and cached in memory for the process lifetime.
    """
    global _templates_checked
    if not _templates_checked:
        ensure_templates_exist()
        _templates_checked = True

    arc_lower = arc_type.lower()
    if "static" in arc_lower:
        name = "single_beam.nii.gz"
    else:
        name = "full_arc.nii.gz"

    if name not in _template_cache:
        path = _template_path(name)
        print(f"Loading beam template: {name} ...")
        _template_cache[name] = sitk.ReadImage(path)
        print(f"  Loaded {name}: size={_template_cache[name].GetSize()}")

    return _template_cache[name]


def _shift_and_rotate_template(template_img, isocenter_mm, gantry_start_deg, arc_type):
    """
    Apply spatial transform so that the template's isocenter (at physical 0,0,0)
    aligns with the plan's isocenter in patient (dose) coordinates.

    All templates have isocenter at physical (0, 0, 0).

    Two modes:
    - Static: shift to plan isocenter + rotate around Z by gantry angle
    - Full Arc (or any arc): shift to plan isocenter only (no rotation)

    Resample maps: output_phys (dose grid) → input_phys (template).
    We want: template_phys = R(-gantry) * (patient_phys - plan_iso)
    For full arc (no rotation): template_phys = patient_phys - plan_iso
    """
    iso = (float(isocenter_mm[0]), float(isocenter_mm[1]), float(isocenter_mm[2]))
    arc_lower = arc_type.lower()

    transform = sitk.Euler3DTransform()

    if "static" in arc_lower:
        # Static beam: shift + rotate around Z to match gantry angle
        # Euler3DTransform: T(x) = R*(x - center) + center + translation
        # We want: T(P) = R(-angle) * (P - iso)
        # So: center = iso, R = R(-angle), and center + translation = (0,0,0) → translation = -iso
        transform.SetCenter(iso)
        angle_rad = np.deg2rad(-float(gantry_start_deg))
        transform.SetRotation(0.0, 0.0, angle_rad)
        transform.SetTranslation((-iso[0], -iso[1], -iso[2]))
    else:
        # Full Arc / any arc: shift only, no rotation
        # T(P) = P - iso → translation = -iso
        transform.SetTranslation((-iso[0], -iso[1], -iso[2]))

    return transform


def generate_beam_for_plan(
    dose_image,
    beam_names,
    isocenters,
    gantry_angles,
    couch_angles,
    collimator_angles,
    arc_types,
):
    """
    Generate the beam volume for a single plan by processing each beam
    with the template approach.

    Parameters:
        dose_image (sitk.Image): The dose NIfTI for this plan (defines output grid).
        beam_names (list[str]): Names of treatment beams.
        isocenters (list[list[float]]): Per-beam isocenter [x,y,z] in mm.
        gantry_angles (list[list[float]]): Per-beam [start, end] gantry angles.
        couch_angles (list[float]): Per-beam couch angle (unused for now, assumed ~0).
        collimator_angles (list[float]): Per-beam collimator angle (unused for now).
        arc_types (list[str]): Per-beam arc type: "Full Arc", "Half Arc", "Static", etc.

    Returns:
        sitk.Image: Beam volume on the dose grid (uint16). Sum of all beams.
    """
    # Output grid from dose
    out_size = dose_image.GetSize()
    out_spacing = dose_image.GetSpacing()
    out_origin = dose_image.GetOrigin()
    out_direction = dose_image.GetDirection()

    # Accumulator
    acc = np.zeros(
        (out_size[2], out_size[1], out_size[0]),  # z, y, x
        dtype=np.float64,
    )

    for i, beam_name in enumerate(beam_names):
        iso = tuple(isocenters[i])
        gantry = gantry_angles[i]
        arc_type = arc_types[i]

        # For static beams, gantry_start = gantry[0]
        # For arcs, gantry_start = gantry[0] (the starting angle of the sweep)
        gantry_start = float(gantry[0])

        # Load appropriate template
        template = _load_template(arc_type)

        # Build transform
        transform = _shift_and_rotate_template(template, iso, gantry_start, arc_type)

        # Resample template onto dose grid
        resampled = sitk.Resample(
            template,
            out_size,
            transform,
            sitk.sitkNearestNeighbor,
            out_origin,
            out_spacing,
            out_direction,
            0,  # default value
            sitk.sitkUInt16,
        )

        acc += sitk.GetArrayFromImage(resampled).astype(np.float64)

    # Convert to uint16
    acc = np.clip(acc, 0, 65535).astype(np.uint16)
    result = sitk.GetImageFromArray(acc)
    result.SetSpacing(out_spacing)
    result.SetOrigin(out_origin)
    result.SetDirection(out_direction)

    return result


def generate_beam_for_plan_from_index(dose_image, plan_row):
    """
    Convenience wrapper: generate beam from a row of the DICOM index CSV.

    Parameters:
        dose_image (sitk.Image): The dose NIfTI for this plan.
        plan_row (dict or pd.Series): A row from the DICOM index CSV with
            BeamNames, Isocenters, GantryAngles, CouchAngles, CollimatorAngles, ArcTypes columns.
            GantryAngles are [min_angle, max_angle] per beam.
            ArcTypes are classified from the angular range (max - min):
              >350° → "Full Arc", >170° → "Half Arc", >5° → "Arc (...)", else "Static".

    Returns:
        sitk.Image: Beam volume on the dose grid (uint16).
    """
    beam_names = json.loads(plan_row["BeamNames"]) if isinstance(plan_row["BeamNames"], str) else plan_row["BeamNames"]
    isocenters = json.loads(plan_row["Isocenters"]) if isinstance(plan_row["Isocenters"], str) else plan_row["Isocenters"]
    gantry_angles = json.loads(plan_row["GantryAngles"]) if isinstance(plan_row["GantryAngles"], str) else plan_row["GantryAngles"]
    couch_angles = json.loads(plan_row["CouchAngles"]) if isinstance(plan_row["CouchAngles"], str) else plan_row["CouchAngles"]
    collimator_angles = json.loads(plan_row["CollimatorAngles"]) if isinstance(plan_row["CollimatorAngles"], str) else plan_row["CollimatorAngles"]
    arc_types = json.loads(plan_row["ArcTypes"]) if isinstance(plan_row["ArcTypes"], str) else plan_row["ArcTypes"]

    return generate_beam_for_plan(
        dose_image=dose_image,
        beam_names=beam_names,
        isocenters=isocenters,
        gantry_angles=gantry_angles,
        couch_angles=couch_angles,
        collimator_angles=collimator_angles,
        arc_types=arc_types,
    )
