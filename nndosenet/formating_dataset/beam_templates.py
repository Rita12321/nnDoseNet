"""
Beam Template Generation
========================
Pre-compute three beam templates at 3mm isotropic spacing:
  1. single_beam.nii.gz  — Pyramid beam, 200cm height, 40x40 field at isocenter (100cm depth)
  2. half_arc.nii.gz     — 90 rotations of single_beam (0°–178°, 2° step), summed
  3. full_arc.nii.gz     — 180 rotations of single_beam (0°–358°, 2° step), summed

Templates are saved in the same directory as this module.
All rotations use nearest-neighbor interpolation to keep sharp integer edges.
"""

import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# Template directory = same folder as this .py file
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
SPACING_MM = 3.0  # 3mm isotropic

# All templates have isocenter at physical (0, 0, 0):
#   single_beam: origin_y = -1000, isocenter at 1000mm from source → physical Y = -1000 + 1000 = 0
#   half_arc / full_arc: canvas centered at (0, 0, 0)
TEMPLATE_ISO_SINGLE_BEAM_MM = (0.0, 0.0, 0.0)
TEMPLATE_ISO_ARC_MM = (0.0, 0.0, 0.0)


def _template_path(name):
    return os.path.join(TEMPLATE_DIR, name)


def generate_single_beam(overwrite=False):
    """
    Generate a solid pyramid beam template.

    Geometry (in the template coordinate system):
        - The beam travels along the Y-axis (anterior-posterior in IEC).
        - Apex at Y=0 (source), base at Y=2000mm (200cm depth).
        - Isocenter at Y=1000mm (100cm depth), where the field is 400x400mm (40x40cm).
        - The beam diverges linearly from the source:
            At depth d (Y=d), field width = 400 * (d / 1000) mm.

    The image is centered so that the isocenter is at the physical center.
    Spacing: 3mm isotropic. Values: uint8, 1 inside beam, 0 outside.
    """
    out_path = _template_path("single_beam.nii.gz")
    if os.path.isfile(out_path) and not overwrite:
        print(f"single_beam.nii.gz already exists. Skipping. Use overwrite=True to regenerate.")
        return out_path

    sp = SPACING_MM
    # Beam height 2000mm along Y. Isocenter at Y=1000mm.
    # Max field at bottom (Y=2000): 400 * (2000/1000) = 800mm in X and Z.
    # We need to cover ±400mm in X and Z from center.
    # Plus some margin for rotation later.

    # Image dimensions (in voxels)
    # Y: 0 to 2000mm -> 2000/3 ≈ 667 voxels
    # X, Z: -400 to +400mm -> 800/3 ≈ 267 voxels
    ny = int(np.ceil(2000.0 / sp))  # 667
    nx = int(np.ceil(800.0 / sp))    # 267
    nz = nx                            # 267

    # Ensure odd dimensions for clean centering
    if nx % 2 == 0: nx += 1
    if ny % 2 == 0: ny += 1
    if nz % 2 == 0: nz += 1

    # Physical origin: place isocenter at center of image
    # Isocenter is at Y=1000mm from source (Y=0).
    # Image Y spans [0, ny*sp), center of Y-dim is at (ny-1)/2 * sp.
    # We want the isocenter at the center. The source is at -1000mm from iso.
    # Origin in physical coordinates:
    origin_x = -((nx - 1) / 2.0) * sp
    origin_z = -((nz - 1) / 2.0) * sp
    origin_y = -1000.0  # Source at Y=0 in beam coords, iso at Y=1000

    arr = np.zeros((nz, ny, nx), dtype=np.uint8)  # SimpleITK: (z, y, x)

    # Fill the pyramid: for each Y (depth), compute the field half-width
    for iy in range(ny):
        y_mm = iy * sp  # distance from source (0 to ~2000)
        if y_mm < 1.0:  # skip source point itself
            continue
        # Half-width at this depth: 200 * (y_mm / 1000)
        half_width = 200.0 * (y_mm / 1000.0)  # 200mm half at isocenter (=40x40 field)

        # Convert to voxel indices
        hw_voxels = half_width / sp
        cx = (nx - 1) / 2.0
        cz = (nz - 1) / 2.0

        x_min = max(0, int(np.floor(cx - hw_voxels)))
        x_max = min(nx - 1, int(np.ceil(cx + hw_voxels)))
        z_min = max(0, int(np.floor(cz - hw_voxels)))
        z_max = min(nz - 1, int(np.ceil(cz + hw_voxels)))

        arr[z_min:z_max+1, iy, x_min:x_max+1] = 1

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((sp, sp, sp))
    img.SetOrigin((origin_x, origin_y, origin_z))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # Identity

    sitk.WriteImage(img, out_path)
    print(f"Generated single_beam.nii.gz: shape={arr.shape}, spacing={sp}mm, "
          f"max={arr.max()}, nonzero={np.count_nonzero(arr)}")
    return out_path


def _rotate_image_z(image, angle_deg, center_mm):
    """
    Rotate a SimpleITK image around the Z-axis by angle_deg degrees,
    around a physical center point. Uses nearest-neighbor interpolation
    to keep sharp integer edges (no float artifacts).

    Parameters:
        image: sitk.Image
        angle_deg: rotation angle in degrees (counter-clockwise around Z)
        center_mm: (x, y, z) physical center of rotation in mm

    Returns:
        sitk.Image: rotated image (same grid as input)
    """
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_mm)
    # Euler3D: rotations around X, Y, Z axes
    angle_rad = np.deg2rad(angle_deg)
    transform.SetRotation(0.0, 0.0, angle_rad)  # Only Z-axis rotation

    rotated = sitk.Resample(
        image,
        image,  # reference image (same grid)
        transform,
        sitk.sitkNearestNeighbor,
        0,  # default pixel value
        image.GetPixelID(),
    )
    return rotated


def generate_half_arc(overwrite=False):
    """
    Generate half_arc.nii.gz by rotating single_beam 90 times
    (0°, 2°, 4°, ..., 178°) around the Z-axis at the isocenter.
    Sum all rotations -> max value ≈ 90.
    """
    out_path = _template_path("half_arc.nii.gz")
    if os.path.isfile(out_path) and not overwrite:
        print(f"half_arc.nii.gz already exists. Skipping.")
        return out_path

    beam_path = _template_path("single_beam.nii.gz")
    if not os.path.isfile(beam_path):
        generate_single_beam(overwrite=False)

    beam_img = sitk.ReadImage(beam_path)

    # We need a larger canvas to hold the rotated beams.
    # The single beam is ~267x667x267 voxels. After rotation, we need
    # enough space for the sweep. The max extent is ~2000mm in any direction
    # from the isocenter. Create a cube large enough.
    # The isocenter is at the physical center of the beam image (Y=0 in physical coords,
    # since origin_y = -1000 and iso is at Y=1000mm from source = Y=0 physical).
    # Wait, let me recalculate:
    # origin_y = -1000. iso is at index ny/2 -> physical y = -1000 + ny/2 * sp.
    # ny ≈ 667, so physical y of iso = -1000 + 333*3 = -1000 + 999 = -1. Close to 0.
    # So isocenter is approximately at physical (0, 0, 0). Good.

    # For the arc, the beam sweeps around Z axis. We need a canvas that fits
    # the beam in all orientations. Max radius from isocenter: ~1000mm (half the beam length).
    # Plus field half-width at base: 400mm. Total: ~1400mm from center.
    # At 3mm: 1400/3 ≈ 467 voxels from center => 935 total per axis.
    canvas_half = int(np.ceil(1400.0 / SPACING_MM))
    canvas_size = 2 * canvas_half + 1  # odd for clean centering

    # Create accumulator
    acc = np.zeros((canvas_size, canvas_size, canvas_size), dtype=np.uint16)

    # Create reference image for resampling
    ref = sitk.Image(canvas_size, canvas_size, canvas_size, sitk.sitkUInt8)
    ref.SetSpacing((SPACING_MM, SPACING_MM, SPACING_MM))
    origin = (-canvas_half * SPACING_MM, -canvas_half * SPACING_MM, -canvas_half * SPACING_MM)
    ref.SetOrigin(origin)
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # Isocenter at physical (0, 0, 0)
    center_mm = (0.0, 0.0, 0.0)

    angles = np.arange(0, 180, 2)  # 0, 2, 4, ..., 178 -> 90 angles
    print(f"Generating half_arc: {len(angles)} rotations on canvas {canvas_size}^3...")

    for angle in tqdm(angles, desc="Half Arc Rotations"):
        # Rotate the beam
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center_mm)
        transform.SetRotation(0.0, 0.0, np.deg2rad(float(angle)))

        rotated = sitk.Resample(
            beam_img,
            ref,  # resample onto the larger canvas
            transform,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        acc += sitk.GetArrayFromImage(rotated).astype(np.uint16)

    result = sitk.GetImageFromArray(acc)
    result.CopyInformation(ref)

    sitk.WriteImage(result, out_path)
    print(f"Generated half_arc.nii.gz: shape={acc.shape}, max={acc.max()}, "
          f"nonzero={np.count_nonzero(acc)}")
    return out_path


def generate_full_arc(overwrite=False):
    """
    Generate full_arc.nii.gz by rotating single_beam 180 times
    (0°, 2°, 4°, ..., 358°) around the Z-axis at the isocenter.
    Sum all rotations -> max value ≈ 180.
    """
    out_path = _template_path("full_arc.nii.gz")
    if os.path.isfile(out_path) and not overwrite:
        print(f"full_arc.nii.gz already exists. Skipping.")
        return out_path

    beam_path = _template_path("single_beam.nii.gz")
    if not os.path.isfile(beam_path):
        generate_single_beam(overwrite=False)

    beam_img = sitk.ReadImage(beam_path)

    # Same canvas as half arc
    canvas_half = int(np.ceil(1400.0 / SPACING_MM))
    canvas_size = 2 * canvas_half + 1

    acc = np.zeros((canvas_size, canvas_size, canvas_size), dtype=np.uint16)

    ref = sitk.Image(canvas_size, canvas_size, canvas_size, sitk.sitkUInt8)
    ref.SetSpacing((SPACING_MM, SPACING_MM, SPACING_MM))
    origin = (-canvas_half * SPACING_MM, -canvas_half * SPACING_MM, -canvas_half * SPACING_MM)
    ref.SetOrigin(origin)
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    center_mm = (0.0, 0.0, 0.0)

    angles = np.arange(0, 360, 2)  # 0, 2, 4, ..., 358 -> 180 angles
    print(f"Generating full_arc: {len(angles)} rotations on canvas {canvas_size}^3...")

    for angle in tqdm(angles, desc="Full Arc Rotations"):
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center_mm)
        transform.SetRotation(0.0, 0.0, np.deg2rad(float(angle)))

        rotated = sitk.Resample(
            beam_img,
            ref,
            transform,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        acc += sitk.GetArrayFromImage(rotated).astype(np.uint16)

    result = sitk.GetImageFromArray(acc)
    result.CopyInformation(ref)

    sitk.WriteImage(result, out_path)
    print(f"Generated full_arc.nii.gz: shape={acc.shape}, max={acc.max()}, "
          f"nonzero={np.count_nonzero(acc)}")
    return out_path


def generate_beam_templates(overwrite=False):
    """
    Generate all three beam templates. Skips any that already exist unless overwrite=True.
    """
    print("=" * 60)
    print("Beam Template Generation")
    print(f"  Spacing: {SPACING_MM} mm isotropic")
    print(f"  Output directory: {TEMPLATE_DIR}")
    print("=" * 60)

    generate_single_beam(overwrite=overwrite)
    generate_half_arc(overwrite=overwrite)
    generate_full_arc(overwrite=overwrite)

    print("\nAll templates generated successfully!")


def ensure_templates_exist():
    """Check if all templates exist; generate missing ones."""
    for name in ["single_beam.nii.gz", "half_arc.nii.gz", "full_arc.nii.gz"]:
        if not os.path.isfile(_template_path(name)):
            print(f"Template {name} not found. Generating...")
            generate_beam_templates(overwrite=False)
            return  # generate_beam_templates handles all three
    print("All beam templates present.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate beam templates (single_beam, half_arc, full_arc)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if files exist")
    parser.add_argument("--spacing", type=float, default=3.0, help="Isotropic spacing in mm (default 3)")
    args = parser.parse_args()

    if args.spacing != SPACING_MM:
        SPACING_MM = args.spacing
        print(f"Using custom spacing: {SPACING_MM} mm")

    generate_beam_templates(overwrite=args.overwrite)
