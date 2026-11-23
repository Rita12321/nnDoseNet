import logging
import os
import shutil
import json
import pydicom
import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from scipy.ndimage import affine_transform
from itertools import product
join = os.path.join

class Beam():
    def __init__(self, beam:pydicom.dataset.Dataset, image_orientation, image_orientation_patient, spacing, static_beam=False, use_field_only=False):
        self.beam = beam
        self.delivery_type = beam.TreatmentDeliveryType
        self.SAD = float(beam.SourceAxisDistance)
        # self.nominal_beam_energy = float(beam.ControlPointSequence[0].NominalBeamEnergy)
        self.points_angle = None
        self.iso_center = tuple([float(coord) for coord in beam.ControlPointSequence[0].IsocenterPosition])

        self.spacing = tuple(spacing)
        # print("image_origin",image_orientation)
        # print("image_origin_patient",image_orientation_patient)
        self.arr_iso_center = tuple(self.get_arr_iso_center(self.iso_center, image_orientation, image_orientation_patient, spacing))
        self.arr_SAD =int(np.round(self.SAD/spacing[1]))
        self.static_beam = static_beam
        self.use_field_only = use_field_only


    def get_beam_points_return_combination (self,beam):
        points_angle = []
        # Enforce field-based (MLC/jaw) cone when use_field_only=True, even for VMAT
        field_flag = self.use_field_only or (self.static_beam and (len(beam.ControlPointSequence)<20))
        # print(field_flag,self.static_beam, (len(beam.ControlPointSequence)<20))
        for bp in beam.ControlPointSequence:

            ga = int(float(bp.GantryAngle)) if 'GantryAngle' in bp else \
                int(float(beam.ControlPointSequence[0].GantryAngle))
            ga = 1*ga if beam.ControlPointSequence[0].GantryRotationDirection == 'CW' else 360-1*ga

            ta = int(float(bp.PatientSupportAngle)) if 'PatientSupportAngle' in bp else \
                int(float(beam.ControlPointSequence[0].PatientSupportAngle))
            ta = 1*ta if beam.ControlPointSequence[0].PatientSupportRotationDirection == 'CW' else 360-1*ta

            ca = int(float(bp.BeamLimitingDeviceAngle)) if 'BeamLimitingDeviceAngle' in bp else \
                int(float(beam.ControlPointSequence[0].BeamLimitingDeviceAngle))
            ca = 1*ca if beam.ControlPointSequence[0].BeamLimitingDeviceAngle == 'CW' else 360-1*ca

            field = [None,None,None,None]
            if field_flag:
                # field = [0,0,0,0]
                if 'BeamLimitingDevicePositionSequence'in bp:
                    for MLC_pos in bp.BeamLimitingDevicePositionSequence:
                        if MLC_pos.RTBeamLimitingDeviceType == 'ASYMX':
                            field[0],field[1] = MLC_pos.LeafJawPositions
                        elif MLC_pos.RTBeamLimitingDeviceType == 'ASYMY':
                            field[2],field[3] = MLC_pos.LeafJawPositions
                if None in field:
                    for MLC_pos in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence:
                        if MLC_pos.RTBeamLimitingDeviceType == 'ASYMX':
                            field[0],field[1] = MLC_pos.LeafJawPositions
                        elif MLC_pos.RTBeamLimitingDeviceType == 'ASYMY':
                            field[2],field[3] = MLC_pos.LeafJawPositions
            #     points_angle.append((self.arr_SAD,self.SAD,self.arr_iso_center,self.iso_center,self.spacing,(0,ta%360,ga%360),tuple(field)))
            # else:

            # (ca%360,ta%360,ga%360)
            points_angle.append((self.arr_SAD,self.SAD,self.arr_iso_center,self.iso_center,self.spacing,(ca%360,ta%360,ga%360),tuple(field)))
        return set(points_angle)


    def get_arr_iso_center(self,iso_center, image_orientation, image_orientation_patient, spacing):
        spacing = self.spacing if spacing is None else spacing
        arr_center = [iso_center[i]-image_orientation_patient[i] for i in range(0,len(iso_center))]
        arr_center = np.matmul(np.linalg.inv(image_orientation), arr_center)
        # print("image",image_orientation_patient)
        # print("After",arr_center)
        arr_center= arr_center// spacing
        return arr_center

# def smash(binary_mask:np.ndarray,axis=0):
#     presence_mask = np.any(binary_mask.astype(bool),axis=axis)
#     #print(np.sum(presence_mask))
#     return presence_mask



def get_exact_sin_cos(angle):
    """Returns exact sine and cosine values for common angles (0°, 90°, 180°, 270°)"""
    angle = (360+angle)% 360  # Normalize to [0, 360)
    lookup = {
        0: (1, 0),
        90: (0, 1),
        180: (-1, 0),
        270: (0, -1),
    }
    if angle in lookup:
        return lookup[angle]  # Exact values
    else:
        rad = np.radians(angle)
        return np.cos(rad), np.sin(rad)  # Use standard functions for non-common angles

def rotation_matrix_3d(angle_z: float, angle_x: float) -> np.ndarray:
    """
    Compute the 3D rotation matrix for rotations around x-axis and z-axis in (z, y, x) coordinate system.

    Parameters:
        angle_x (float): Rotation angle around the x-axis (in radians).
        angle_z (float): Rotation angle around the z-axis (in radians).

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    # Rotation matrix around the x-axis
    cos_x,sin_x = get_exact_sin_cos(angle_x)
    cos_z,sin_z = get_exact_sin_cos(angle_z)
    Rz = np.array([
        [1, 0, 0],
        [0, cos_z, -sin_z],
        [0, sin_z, cos_z]
    ])

    # Rotation matrix around the z-axis
    Rx = np.array([
        [cos_x, -sin_x, 0],
        [sin_x, cos_x, 0],
        [0, 0, 1]
    ])

    # Combined rotation (Rz * Rx) since the order of transformations matters
    R = Rx @ Rz
    return R



# def apply_inverse_rotation(R: np.ndarray, point: np.ndarray) -> np.ndarray:
#     """Apply the inverse of a rotation matrix to a point."""
#     R_inv = np.linalg.inv(R)  # Compute inverse rotation matrix
#     return R @ point
def line_function(p1: np.ndarray, p2: np.ndarray,y) -> np.ndarray:
    """
    Compute (z, x) coordinates given y, assuming a line passes through p1 and p2.
    Parameters:
        p1 (np.ndarray): First point (z1, y1, x1).
        p2 (np.ndarray): Second point (z2, y2, x2).
        y (float): Desired y-coordinate.

    Returns:
        np.ndarray: Corresponding (z, x) point on the line.
    """
    # Extract components
    z1, y1, x1 = p1
    z2, y2, x2 = p2

    # Compute t based on y
    if y1 == y2:
        return[]
        raise ValueError("The two points have the same y-coordinate, cannot interpolate.")

    t = (y - y1) / (y2 - y1)  # Solve for t in the line equation

    # Compute corresponding (z, x)
    z = (1 - t) * z1 + t * z2
    x = (1 - t) * x1 + t * x2
    # print(t, (z,y,x), p2 )
    return [z,y,x]


def find_3d_coord_with_beam_field(beam_field, arr_dim=None, iso_center=None, SAD=None, angle=None,spacing=None, origin=None):
    """ Finds 3D voxel indices for a rotated projection along a tilted y-axis """

    [D, H, W] = arr_dim
    # print(arr_dim)
    source = np.array([0, -SAD, 0])
    angle_y,angle_z, angle_x = angle

    R = np.linalg.inv(rotation_matrix_3d(-angle_x, -angle_z))
    source_transformed = R @ source
    rotation_matrix = None
    # angle_y = (360-angle_y)%360
    if angle_y != 0:
        if angle_y == 90:
            rotation_matrix = np.array([[0, -1], [1, 0]])
        elif angle_y == 180:
            rotation_matrix = np.array([[-1, 0], [0, -1]])
        elif angle_y == 270:
            rotation_matrix = np.array([[0, 1], [-1, 0]])
        else:
            # Compute the general rotation matrix using sine and cosine
            theta_rad = np.deg2rad(angle_y)
            rotation_matrix = np.array([
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad),  np.cos(theta_rad)]
            ])
    xs =np.arange(beam_field[0],beam_field[1],spacing[0])
    zs =np.arange(beam_field[2],beam_field[3],spacing[2])
    points=[]
    for zz in zs:
        for xx in xs:
            if rotation_matrix is not None:
                z,x = rotation_matrix @ (zz, xx)
            else:
                z,x = zz, xx
            point_transformed = R @ (z,0, x)

            for y in np.arange(np.floor(-iso_center[1]),np.ceil(-iso_center[1]+(H+1)*spacing[1]),spacing[1]):
                zt, yt, xt= line_function(p1 = source_transformed,p2 = point_transformed, y = y)
                points.append([zt+iso_center[2], yt+iso_center[1], xt+iso_center[0]])
            # print("point_trans",source_transformed,point_transformed, iso_center)
    return points

# def rotate_3d_image_sitk(image, angles_degrees, center = None, ref_image = None):
#     # Create composite transform for all rotations
#     interpolator = sitk.sitkLinear#sitk.sitkNearestNeighbor
#     transform = sitk.Euler3DTransform()
#     transform.SetCenter(center)
#     resampled_flag = False
#     # Apply rotations in specified order
#     for r in [2,1]: #x,yz
#         if angles_degrees[r]!=0:
#             r_theta = [0,0,0]
#             r_theta[r] = np.deg2rad(angles_degrees[r])
#             transform.SetRotation(r_theta[0],r_theta[1],r_theta[2])

#             # Resample with reference image parameters to ensure matching size
#             image = sitk.Resample(image, referenceImage=ref_image, transform = transform, interpolator=interpolator, defaultPixelValue=0)
#             resampled_flag = True
#             # Reset transform for next rotation
#             transform = sitk.Euler3DTransform()
#             transform.SetCenter(center)
#     if resampled_flag is False:
#         image = sitk.Resample(image, referenceImage=ref_image, transform=transform, interpolator=interpolator, defaultPixelValue=0)
#     rotated_image_np = sitk.GetArrayFromImage(image)
#     return rotated_image_np, image
def find_3d_coord(x_z_intersection, arr_dim=None, iso_center=None, SAD=None, angle=None,spacing=None, origin=None):
    """ Finds 3D voxel indices for a rotated projection along a tilted y-axis """
    [D, H, W] = arr_dim
    # print(arr_dim)
    source = np.array([0, -SAD, 0])
    _,angle_z, angle_x = angle

    R = np.linalg.inv(rotation_matrix_3d(-angle_x, -angle_z))
    source_transformed = R @ source

    points=[]
    for z,x in x_z_intersection:

        # print("center",(iso_center[2],iso_center[0]))
        # print("point",(zz,0, xx),(z,0, x))
        point_transformed = R @ (z,0, x)

        for y in np.arange(np.floor(-iso_center[1]),np.ceil(-iso_center[1]+(H+1)*spacing[1]),spacing[1]):
            zt, yt, xt= line_function(p1 = source_transformed,p2 = point_transformed, y = y)
            points.append([zt+iso_center[2], yt+iso_center[1], xt+iso_center[0]])
        # print("point_trans",source_transformed,point_transformed, iso_center)
    return points


def get_rot_target_coordinate(target_img:sitk.Image,angle,iso_center,axis=0):
    target_array = sitk.GetArrayFromImage(target_img)
    nonzero_coords = np.argwhere(target_array > 0)
    origin = target_img.GetOrigin()
    spacing = target_img.GetSpacing()
    _,angle_x, angle_z = angle
    R = rotation_matrix_3d(angle_z=-angle_z, angle_x=-angle_x)
    rot_coord = []
    for zz,yy,xx in  nonzero_coords:
        zz = zz*spacing[2]-iso_center[2]
        yy = yy*spacing[1]-iso_center[1]
        xx = xx*spacing[0]-iso_center[0]
        z, y, x =R @ (zz,yy,xx)
        rot_coord.append((z,x))

    return list(set(rot_coord))
def fill_image_with_coord(rf_image:sitk.Image, all_3d_coord, value = 1):
    image_size = rf_image.GetSize()
    origin = rf_image.GetOrigin()
    spacing = rf_image.GetSpacing()
    # print(origin)
    sample_image = sitk.Image(image_size, sitk.sitkUInt8)
    sample_image.CopyInformation(rf_image)

    for point in all_3d_coord:
        index_float = [(point[0] ) / spacing[2],(point[1]) / spacing[1],(point[2] ) / spacing[0]]
        # print(point,index)
        # index = [int((point[0] - origin[2]) / spacing[2]),int((point[1] - origin[1]) / spacing[1]),int((point[2] - origin[0]) / spacing[0])]

        indexs = [tuple(funcs) for funcs in product(*[(np.floor(x), np.ceil(x)) for x in index_float])]
        indexs = list(set(indexs))
        for index in indexs:
            if (0 <= index[0] < image_size[2])and(0 <= index[1] < image_size[1])and(0 <= index[2] < image_size[0]):#all(0 <= index[i] < image_size[i] for i in range(3)):  # Ensure inside bounds
                # print("insert")
                sample_image[int(index[2]), int(index[1]), int(index[0])] = value  # Mark point in the image

    return sample_image

def get_beam(target_img:sitk.Image, angle, iso_center = None, arr_center = None, spacing = None, SAD = None,arr_SAD = None, beam_field = None):
    # print("iso_center", iso_center)
    # print("arr_center", arr_center)
    arr_center= [iso_center[i]-target_img.GetOrigin()[i] for i in range(3)]
    # print(angle)
    D,H,W = sitk.GetArrayFromImage(target_img).shape
    if None in beam_field:

        target_coordinate = get_rot_target_coordinate(target_img, angle, arr_center,axis=1)
        # print(np.nonzero(mask_arr.flatten()))
        all_3d_coord = find_3d_coord(target_coordinate, arr_dim = (D, H, W) , iso_center = arr_center,  SAD = SAD, angle=angle, spacing = spacing, origin = target_img.GetOrigin())
    else:
        all_3d_coord = find_3d_coord_with_beam_field(beam_field, arr_dim = (D, H, W) , iso_center = arr_center,  SAD = SAD, angle=angle, spacing = spacing, origin = target_img.GetOrigin())


    # all_3d_coord = find_3d_coord(mask_arr, arr_dim = (D, H, W) , iso_center = arr_center,  SAD = SAD, angle=angle, spacing = spacing, origin = target_img.GetOrigin())
    # print(arr_center)
    cone_image = fill_image_with_coord(rf_image=target_img , all_3d_coord=all_3d_coord, value = 1)

    return cone_image


def process_control_point(target_img, beam_info):
    try:
        arr_SAD, SAD, arr_iso_center,iso_center,spacing,angle, beam_field = beam_info
    except Exception as exc:
        logging.error(f'Parallel processing did beam info : {exc}')
    logging.info(f'... running control point # {iso_center} {SAD} {angle},{beam_field})') ##DEBUG
    cone_image = get_beam(target_img = target_img, angle = angle, iso_center = iso_center, arr_center = arr_iso_center, spacing = spacing, SAD = SAD,arr_SAD = arr_SAD,beam_field=beam_field)

    return cone_image


def get_beams_image_pool(target_img,beam_infos,num_process):
    from multiprocessing import Pool#, cpu_count
    beams_img=None

    with Pool(processes=num_process) as pool:
        beam_args = [(target_img, beam_info)
                    for beam_info in beam_infos]


        results = pool.starmap(process_control_point, beam_args)

        # Combine results
        for cone_image in results:
            if beams_img is None:
                beams_img = cone_image
            else:
                beams_img = sitk.Add(beams_img, cone_image)

    return beams_img

def move_all_files_and_folders(src_dir, dest_dir):
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory '{src_dir}' does not exist.")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # Create destination directory if it doesn't exist

    # Iterate through all items in the source directory
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        # Move the file or folder
        shutil.move(src_path, dest_path)
        print(f"Moved: {src_path} -> {dest_path}")

def resample_to_high_res(image):
    """
    Resample the input image to have an isotropic spacing of (1,1,1)
    with a larger image size.

    Parameters:
        image (sitk.Image): The input SimpleITK image.

    Returns:
        sitk.Image: The resampled image with spacing (1,1,1).
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = (2.0, 2.0, 2.0)

    # Compute new size while preserving the physical dimensions
    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    return resampler.Execute(image)


def resample_back(image, ref_image:sitk.Image):
    """
    Resample the image back to its original spacing and size.

    Parameters:
        image (sitk.Image): The resampled high-resolution image.
        original_spacing (tuple): The original spacing before resampling.
        original_size (tuple): The original size before resampling.

    Returns:
        sitk.Image: The image resampled back to the original spacing and size.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ref_image.GetSpacing())
    resampler.SetSize(ref_image.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(ref_image.GetOrigin())
    resampler.SetOutputDirection(ref_image.GetDirection())

    return resampler.Execute(image)


def _find_rp_path_by_uid(patient_folder, ref_uid):
    """Return path to RP DICOM file whose SOPInstanceUID matches ref_uid, or None."""
    rp_dir = join(patient_folder, "Dicom", "RP")
    if not ref_uid or not os.path.isdir(rp_dir):
        return None
    for f in os.listdir(rp_dir):
        if not f.lower().endswith((".dcm", ".dicom")):
            continue
        path = join(rp_dir, f)
        try:
            ds = pydicom.dcmread(path)
            if getattr(ds, "SOPInstanceUID", None) == ref_uid:
                return path
        except Exception:
            continue
    return None


def _get_isocenter_from_plan(plan):
    """Return isocenter (x, y, z) in mm from first treatment beam, first control point."""
    for beam in getattr(plan, "BeamSequence", []):
        if getattr(beam, "TreatmentDeliveryType", None) != "TREATMENT":
            continue
        seq = getattr(beam, "ControlPointSequence", None)
        if seq and len(seq) > 0:
            pos = getattr(seq[0], "IsocenterPosition", None)
            if pos is not None:
                return tuple(float(c) for c in pos)
    return None


def _get_isocenter_from_beam(beam):
    """Return isocenter (x, y, z) in mm from this beam's first control point."""
    seq = getattr(beam, "ControlPointSequence", None)
    if seq and len(seq) > 0:
        pos = getattr(seq[0], "IsocenterPosition", None)
        if pos is not None:
            return tuple(float(c) for c in pos)
    return None



def generate_beam_cone_pytorch(target_image: sitk.Image, iso_center_mm, sad_mm=1000.0, field_size_mm=400.0, gantry_angles=[0.0], device='cuda'):
    """
    Generates a cone beam mask using PyTorch with chunking to manage VRAM usage.
    """
    # 1. Setup
    size = target_image.GetSize() # (W, H, D) -> x, y, z indices
    spacing = target_image.GetSpacing()
    origin = target_image.GetOrigin()
    direction = target_image.GetDirection()

    nx, ny, nz = size # W, H, D

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    try:
        # Pre-calculate constants on GPU
        d = torch.tensor(direction, device=device, dtype=torch.float64).view(3, 3)
        o = torch.tensor(origin, device=device, dtype=torch.float64)
        s = torch.tensor(spacing, device=device, dtype=torch.float64)

        # Source position: AP Beam (Anterior to Posterior)
        # In DICOM/ITK LPS coordinates: +Y is Posterior, -Y is Anterior.
        # User wants "Front to Back" (Anterior -> Posterior).
        # So Source is at -Y side relative to Isocenter.
        source_x = iso_center_mm[0]
        source_y = iso_center_mm[1] - sad_mm  # Source is Anterior (at Y = iso_y - SAD)
        source_z = iso_center_mm[2]

        # Prepare output array (CPU)
        final_mask_np = np.zeros(size[::-1], dtype=np.uint8) # (D, H, W) -> (z, y, x) matching sitk

        # Pre-compute rotation matrices for all angles
        # IEC Gantry Scale: 0=Ant (-Y), 90=Left (+X), 180=Post (+Y), 270=Right (-X)
        # We rotate Voxel P by -Theta to map P to the Static Frame (Angle 0).
        # Static Frame: Source at (0, -SAD). Beam points +Y.
        # Transformation derived:
        # u' = u cos(g) + v sin(g)
        # v' = -u sin(g) + v cos(g)
        # (Where u=x-iso_x, v=y-iso_y)

        angles_rad = np.deg2rad(gantry_angles)
        cos_a = torch.tensor(np.cos(angles_rad), device=device, dtype=torch.float64).view(1, -1) # (1, N)
        sin_a = torch.tensor(np.sin(angles_rad), device=device, dtype=torch.float64).view(1, -1) # (1, N)

        # Chunking configuration
        chunk_size = 32

        # Create X and Y ranges once
        w_range = torch.arange(nx, device=device, dtype=torch.float64) # x
        h_range = torch.arange(ny, device=device, dtype=torch.float64) # y
        # Broadcast ready
        gx = w_range.view(1, 1, -1)
        gy = h_range.view(1, -1, 1)

        half_field = field_size_mm / 2.0

        for z_start in range(0, nz, chunk_size):
            z_end = min(z_start + chunk_size, nz)
            current_chunk_size = z_end - z_start

            # Create Z range for this chunk
            d_range_chunk = torch.arange(z_start, z_end, device=device, dtype=torch.float64)
            gz = d_range_chunk.view(-1, 1, 1)

            # Calculate Physical Coordinates P(x,y,z)
            x_term = gx * s[0]
            y_term = gy * s[1]
            z_term = gz * s[2]

            px = o[0] + d[0,0]*x_term + d[0,1]*y_term + d[0,2]*z_term
            py = o[1] + d[1,0]*x_term + d[1,1]*y_term + d[1,2]*z_term
            pz = o[2] + d[2,0]*x_term + d[2,1]*y_term + d[2,2]*z_term

            # Deltas from Source
            # Relative to Isocenter
            u = px - iso_center_mm[0]
            v = py - iso_center_mm[1]
            # w = pz - iso_center_mm[2] # Z is invariant to Gantry rotation (usually)

            # Reshape for broadcasting (K, 1)
            u = u.view(-1, 1)
            v = v.view(-1, 1)

            # Batch over angles to avoid OOM
            # u_rot size: (K, N). K=8M, N=360 -> 3B floats -> 12GB. Too big.
            # Split N into batches of 32
            angle_batch_size = 32
            num_angles = cos_a.shape[1]

            mask_chunk_all = torch.zeros(u.shape[0], dtype=torch.bool, device=device)

            for i in range(0, num_angles, angle_batch_size):
                end_i = min(i + angle_batch_size, num_angles)
                cos_batch = cos_a[:, i:end_i]
                sin_batch = sin_a[:, i:end_i]

                # Apply Rotation (Vectorized over batch angles)
                u_rot = u * cos_batch + v * sin_batch
                v_rot = -u * sin_batch + v * cos_batch

                dy_local = v_rot + sad_mm
                dx_local = u_rot
                # dz_local is same for all angles, but needs broadcasting
                # dz_local = pz - iso_z
                # pz is (K,). dz_local is (K,). Broadcase to (K, Batch)

                # Avoid div 0
                dy_local = torch.where(dy_local == 0, torch.tensor(1e-6, device=device), dy_local)

                t = sad_mm / dy_local

                # Check field
                # Need dz relative to PZ of voxel.
                # pz is shape (K). dz = pz - iso_z
                dz_val = (pz - iso_center_mm[2]).view(-1, 1)

                mask_batch = (torch.abs(t * dx_local) <= half_field) & \
                             (torch.abs(t * dz_val) <= half_field) & \
                             (t > 0)

                # OR accumulation
                mask_chunk_all |= mask_batch.any(dim=1)

                del u_rot, v_rot, dy_local, dx_local, t, mask_batch

                # Reshape back to (Chunk, H_roi, W_roi)
                # NOTE: We need to place this into the larger final_mask_np (D, H, W)
                # final_mask_np indices: [z, y, x]

                chunk_res = mask_chunk_all.cpu().numpy().reshape(current_chunk_size, y_end_proc - y_start_proc, x_end_proc - x_start_proc).astype(np.uint8)

                final_mask_np[z_start:z_end, y_start_proc:y_end_proc, x_start_proc:x_end_proc] = chunk_res

                del mask_chunk_all

        out = sitk.GetImageFromArray(final_mask_np)
        out.CopyInformation(target_image)
        return out

    except Exception as e:
        logging.error(f"PyTorch beam generation failed: {e}. Falling back to NumPy.")
        return generate_beam_cone_numpy_fallback(target_image, iso_center_mm, sad_mm, field_size_mm, gantry_angles)


    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    try:
        # Tensors
        # Direction Matrix D: Columns are basis vectors of Image axes in Patient LPS
        # D = [X_i, Y_i, Z_i]
        d = torch.tensor(direction, device=device, dtype=torch.float64).view(3, 3)
        o = torch.tensor(origin, device=device, dtype=torch.float64)
        s = torch.tensor(spacing, device=device, dtype=torch.float64)

        # Ranges
        # If 'bbox' is provided, use it to restrict Z range (and potentially X/Y)
        # bbox format: [x_min, x_max, y_min, y_max, z_min, z_max] (indices)
        # defaulting to full image

        # NOTE: For VMAT, the beam moves, so XY bounds might be risky if aggressive?
        # But Z-range of Dose is usually safe (beam doesn't go where dose is 0).
        # Let's stick to safe Z-restriction for now, or full bbox if confident.
        # User said "chunking or cropping based on background(zeros) of dose file"

        z_min, z_max = 0, nz
        x_min, x_max = 0, nx
        y_min, y_max = 0, ny

        # Check if bbox is passed in kwargs (assuming we update signature or use dynamic args)
        # Actually I need to update signature. Let's do it below.

    except Exception as e:
         logging.error(f"PyTorch setup failed: {e}")
         return None

def generate_beam_cone_pytorch_iec(target_image: sitk.Image, iso_center_mm, sad_mm=1000.0, field_size_mm=400.0,
                                   angles_deg_list=[(0., 0., 0.)], device='cuda', dose_bbox=None):
    """
    Generates beam mask using full IEC 61217 transformations.
    angles_deg_list: List of (gantry, couch, collimator) tuples in degrees.
    dose_bbox: Optional [x_min, x_max, y_min, y_max, z_min, z_max] indices of non-zero dose.
    """
    # 1. Setup Image Geometry (LPS)
    size = target_image.GetSize()
    spacing = target_image.GetSpacing()
    origin = target_image.GetOrigin()
    direction = target_image.GetDirection()

    nx, ny, nz = size

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    try:
        # Prepare output
        final_mask_np = np.zeros(size[::-1], dtype=np.uint8) # (z, y, x)

        # Determine Processing Bounds
        z_start_proc, z_end_proc = 0, nz
        y_start_proc, y_end_proc = 0, ny
        x_start_proc, x_end_proc = 0, nx

        if dose_bbox is not None:
            # Add a small safety margin?
            # Dose is usually tight. Beam edges might have low dose < threshold?
            # User said "based on background(zeros)".
            # Be safe: Expand by 5 voxels?
            margin = 5
            x_start_proc = max(0, dose_bbox[0] - margin)
            x_end_proc = min(nx, dose_bbox[1] + margin)
            y_start_proc = max(0, dose_bbox[2] - margin)
            y_end_proc = min(ny, dose_bbox[3] + margin)
            z_start_proc = max(0, dose_bbox[4] - margin)
            z_end_proc = min(nz, dose_bbox[5] + margin)

            logging.info(f"Optimizing Beam Gen: Processing only Dose ROI: X[{x_start_proc}:{x_end_proc}], Y[{y_start_proc}:{y_end_proc}], Z[{z_start_proc}:{z_end_proc}]")

        # Tensors
        d = torch.tensor(direction, device=device, dtype=torch.float64).view(3, 3)
        o = torch.tensor(origin, device=device, dtype=torch.float64)
        s = torch.tensor(spacing, device=device, dtype=torch.float64)

        # Pre-compute rotation matrices for all N control points
        # angles_deg_list is (N, 3): Gantry, Couch, Collimator
        # Convert to Radians
        angles_rad = np.deg2rad(angles_deg_list)
        ag = torch.tensor(angles_rad[:, 0], device=device, dtype=torch.float64) # Gantry
        asup = torch.tensor(angles_rad[:, 1], device=device, dtype=torch.float64) # Couch (Support)
        ac = torch.tensor(angles_rad[:, 2], device=device, dtype=torch.float64) # Collimator

        inv_asup = asup # Positive (Forward) Rotation P_patient -> P_fixed
        inv_ag = ag     # Positive (Reverse from -ag?) No, user says G90 looks like 270 (-90). So we need +ag.
        inv_ac = ac     # Match Gantry convention.

        # Grid Setup - Restricted to ROI
        w_range = torch.arange(x_start_proc, x_end_proc, device=device, dtype=torch.float64)
        h_range = torch.arange(y_start_proc, y_end_proc, device=device, dtype=torch.float64)

        gx = w_range.view(1, 1, -1)
        gy = h_range.view(1, -1, 1)

        iso = torch.tensor(iso_center_mm, device=device, dtype=torch.float64)

        # Batching
        chunk_size = 32 # Z-slices per chunk
        angle_batch_size = 32

        # Use uint16 to support >255 sums (e.g. 180 control points -> 180)
        final_mask_np = np.zeros(size[::-1], dtype=np.uint16) # (z, y, x)

        # Iterate only through Z ROI
        for z_start in range(z_start_proc, z_end_proc, chunk_size):
            z_end = min(z_start + chunk_size, z_end_proc)
            current_chunk_size = z_end - z_start

            d_range_chunk = torch.arange(z_start, z_end, device=device, dtype=torch.float64)
            gz = d_range_chunk.view(-1, 1, 1)

            # P_lps (Physical Coords)
            x_term = gx * s[0]
            y_term = gy * s[1]
            z_term = gz * s[2]

            px_lps = o[0] + d[0,0]*x_term + d[0,1]*y_term + d[0,2]*z_term
            py_lps = o[1] + d[1,0]*x_term + d[1,1]*y_term + d[1,2]*z_term
            pz_lps = o[2] + d[2,0]*x_term + d[2,1]*y_term + d[2,2]*z_term

            # 1. Translate to Iso
            vx_lps = px_lps - iso[0]
            vy_lps = py_lps - iso[1]
            vz_lps = pz_lps - iso[2]

            # 2. LPS -> IEC Fixed
            # X_f = -X_lps (LPS X is Left, IEC X is Right)
            # Y_f = Z_lps (Head)
            # Z_f = -Y_lps (Post -> Ant)
            vx_f = -vx_lps
            vy_f = vz_lps
            vz_f = -vy_lps

            # Flatten for batching
            # Shape (K) where K = chunk * H * W
            flat_dim = vx_f.shape[0] * vx_f.shape[1] * vx_f.shape[2]
            vx_f = vx_f.view(-1, 1) # (K, 1)
            vy_f = vy_f.view(-1, 1)
            vz_f = vz_f.view(-1, 1)

            mask_chunk_all = torch.zeros(flat_dim, dtype=torch.int16, device=device)

            num_angles = len(angles_deg_list)

            # Loop over angle batches
            for i in range(0, num_angles, angle_batch_size):
                end_i = min(i + angle_batch_size, num_angles)

                # Get angles for batch (1, Batch)
                b_inv_asup = inv_asup[i:end_i].view(1, -1)
                b_inv_ag = inv_ag[i:end_i].view(1, -1)
                b_inv_ac = inv_ac[i:end_i].view(1, -1)

                # 3. Rotations (Broadcasting K vs Batch)

                # Couch (Z_f rotation)
                # V_s = RotZ(-couch) * V_f
                # x_s = x_f cos - y_f sin
                # y_s = x_f sin + y_f cos
                c_s = torch.cos(b_inv_asup); s_s = torch.sin(b_inv_asup)
                vx_s = vx_f * c_s - vy_f * s_s
                vy_s = vx_f * s_s + vy_f * c_s
                vz_s = vz_f # Z invariant

                # Gantry (Y_s rotation -> Y_g is axis?)
                # Wait. Gantry rotates around Y_f (Longitudinal)?
                # Standard IEC: Gantry rotates around Y coordinate.
                # V_g = RotY(-gantry) * V_s
                # x_g = x_s cos + z_s sin
                # z_g = -x_s sin + z_s cos
                c_g = torch.cos(b_inv_ag); s_g = torch.sin(b_inv_ag)
                vx_g = vx_s * c_g + vz_s * s_g
                vy_g = vy_s
                vz_g = -vx_s * s_g + vz_s * c_g

                # Collimator (Z_g rotation -> Beam Axis?)
                # Beam Axis is Z_g (Standard?). Beam normally points Source(0,0,SAD) to Iso.
                # In BEV, we want Project to Z plane.
                # Inverse Collimator: RotZ(-coll)
                c_c = torch.cos(b_inv_ac); s_c = torch.sin(b_inv_ac)
                vx_b = vx_g * c_c - vy_g * s_c
                vy_b = vx_g * s_c + vy_g * c_c
                vz_b = vz_g

                # 4. Cone Check in Beam Eye View (BEV)
                # In BEV (unrotated IEC):
                # Source is at +Z (SAD)? or -Z?
                # IEC Beam Frame: Origin at Iso. Z axis towards Source?
                # IEC 61217: Z_b axis is from Isocenter towards Source.
                # So Source is at (0, 0, +SAD).
                # Ray from Source(0,0,SAD) to Voxel(vx_b, vy_b, vz_b).
                # Project to Isocenter Plane (Z=0).
                # Intersect Z=0.
                # Ray R(t) = S + t(V-S)
                # z(t) = SAD + t(vz_b - SAD) = 0 => t = SAD / (SAD - vz_b)
                # proj_x = sx + t(vx_b - sx) = 0 + t(vx_b) = vx_b * SAD / (SAD - vz_b)
                # proj_y = sy + t(vy_b - sy) = 0 + t(vy_b) = vy_b * SAD / (SAD - vz_b)

                # Avoid div 0 (SAD ~= vz_b)
                denom = sad_mm - vz_b
                denom = torch.where(denom == 0, torch.tensor(1e-6, device=device), denom)

                scale = sad_mm / denom

                proj_x = vx_b * scale
                proj_y = vy_b * scale

                # Condition:
                # 1. Inside Field at Iso (X, Y)
                # 2. Voxel is "below" source (towards iso).
                # Source at +SAD. Voxel at vz_b.
                # If vz_b < SAD, it is valid?
                # Usually yes. And we want to ensure it's in the "forward" direction of beam.
                # Beam direction is -Z.
                # So we want downstream?
                # Actually simpler: scale > 0 implies t > 0 (if SAD > 0).
                # If vz_b > SAD, scale < 0 (behind source).

                mask_batch = (torch.abs(proj_x) <= field_size_mm/2.0) & \
                             (torch.abs(proj_y) <= field_size_mm/2.0) & \
                             (scale > 0)

                # Sum overlapping beams/control points instead of logical OR
                # Cast boolean to int16 (0 or 1) then sum
                mask_chunk_all += mask_batch.sum(dim=1, dtype=torch.int16)

            # Reshape back to (Chunk, H_roi, W_roi)
            # final_mask_np indices: [z, y, x]
            # Use calculated ROI width/height
            roi_h = y_end_proc - y_start_proc
            roi_w = x_end_proc - x_start_proc

            final_mask_np[z_start:z_end, y_start_proc:y_end_proc, x_start_proc:x_end_proc] = \
                mask_chunk_all.cpu().numpy().reshape(current_chunk_size, roi_h, roi_w).astype(np.uint16)
            del mask_chunk_all

        out = sitk.GetImageFromArray(final_mask_np)
        out.CopyInformation(target_image)
        return out

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"PyTorch IEC generation failed: {e}. Returning empty.")
        # Fallback to simple None or empty for now
        return None

def generate_beam_cone_numpy_fallback(target_image, iso_center_mm, sad_mm=1000.0, field_size_mm=400.0, gantry_angles=[0.0]):
    size = target_image.GetSize()
    spacing = target_image.GetSpacing()
    origin = target_image.GetOrigin()
    direction = target_image.GetDirection()

    nx, ny, nz = size

    w_range = np.arange(nx, dtype=np.float64)
    h_range = np.arange(ny, dtype=np.float64)
    d_range = np.arange(nz, dtype=np.float64)

    gz, gy, gx = np.meshgrid(d_range, h_range, w_range, indexing='ij')

    direction = np.array(direction).reshape(3,3)

    px = origin[0] + direction[0,0]*(gx*spacing[0]) + direction[0,1]*(gy*spacing[1]) + direction[0,2]*(gz*spacing[2])
    py = origin[1] + direction[1,0]*(gx*spacing[0]) + direction[1,1]*(gy*spacing[1]) + direction[1,2]*(gz*spacing[2])
    pz = origin[2] + direction[2,0]*(gx*spacing[0]) + direction[2,1]*(gy*spacing[1]) + direction[2,2]*(gz*spacing[2])

    # Rotation Logic (NumPy)
    u = px - iso_center_mm[0]
    v = py - iso_center_mm[1]
    w = pz - iso_center_mm[2]

    angles_rad = np.deg2rad(gantry_angles)
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    final_mask = np.zeros_like(u, dtype=bool)

    # Iterate angles (NumPy broadcasting over angles might explode mem too, so loop)
    for i in range(len(angles_rad)):
        c = cos_a[i]
        s = sin_a[i]

        # u' = u cos + v sin
        # v' = -u sin + v cos
        u_rot = u * c + v * s
        v_rot = -u * s + v * c

        # Static check
        dy = v_rot + sad_mm
        dx = u_rot
        dz = w

        # t = SAD / dy
        with np.errstate(divide='ignore', invalid='ignore'):
            t = sad_mm / dy
            mask = (np.abs(t * dx) <= field_size_mm/2) & (np.abs(t * dz) <= field_size_mm/2) & (t > 0)

        final_mask |= mask

    out = sitk.GetImageFromArray(final_mask.astype(np.uint8))
    out.CopyInformation(target_image)
    return out

def generate_beam_static_cylinder(target_image, iso_center_mm, radius_mm=200.0, SAD=1000.0):
    # Backward compatibility
    return generate_beam_cone_pytorch(target_image, iso_center_mm, sad_mm=SAD, field_size_mm=radius_mm*2.0)



def _generate_beam_for_one_plan(patient_folder, plan_path, num_process, use_field_only, use_static_cylinder=False, static_cylinder_radius_mm=200.0):
    """Generate beam map for one RP; return (final_beams sitk.Image, target_image for origin/spacing) or (None, None)."""
    try:
        plan = pydicom.dcmread(plan_path)
        ptv_file = join(patient_folder, "Nifti", "combined_target.nii.gz")
        if not os.path.exists(ptv_file):
            return None, None
        target_image_ori = sitk.ReadImage(ptv_file)
        target_array = sitk.GetArrayFromImage(target_image_ori) > 0
        target_array = target_array.astype(np.uint16)
        target_image = sitk.GetImageFromArray(target_array)
        target_image.CopyInformation(target_image_ori)

        # -------------------------------------------------------------
        # Dose-Based Optimization
        # Load dose.nii.gz if available to compute bounding box
        # -------------------------------------------------------------
        dose_bbox = None
        dose_file = join(patient_folder, "Nifti", "dose.nii.gz")
        if os.path.exists(dose_file):
            try:
                dose_img = sitk.ReadImage(dose_file)
                dose_arr = sitk.GetArrayFromImage(dose_img)
                # Find non-zero indices
                # numpy indices are (z, y, x)
                non_zeros = np.nonzero(dose_arr)
                if len(non_zeros[0]) > 0:
                    # Get indices in Dose Space
                    z_min_d, z_max_d = np.min(non_zeros[0]), np.max(non_zeros[0])
                    y_min_d, y_max_d = np.min(non_zeros[1]), np.max(non_zeros[1])
                    x_min_d, x_max_d = np.min(non_zeros[2]), np.max(non_zeros[2])

                    # Convert to Physical Coordinates (LPS)
                    # Point 1: Min Corner (Index -> Physical)
                    p1_phys = dose_img.TransformIndexToPhysicalPoint((int(x_min_d), int(y_min_d), int(z_min_d)))
                    # Point 2: Max Corner (Index -> Physical)
                    p2_phys = dose_img.TransformIndexToPhysicalPoint((int(x_max_d), int(y_max_d), int(z_max_d)))

                    # Compute Physical BBox (Min/Max for all axes)
                    # Because of rotation/direction, p1 might not be min in valid LPS.
                    # We take min/max of physical coords.
                    phys_x = [p1_phys[0], p2_phys[0]]
                    phys_y = [p1_phys[1], p2_phys[1]]
                    phys_z = [p1_phys[2], p2_phys[2]]

                    min_px, max_px = min(phys_x), max(phys_x)
                    min_py, max_py = min(phys_y), max(phys_y)
                    min_pz, max_pz = min(phys_z), max(phys_z)

                    # Margin in mm (e.g. 10mm)
                    margin_mm = 10.0
                    min_px -= margin_mm; max_px += margin_mm
                    min_py -= margin_mm; max_py += margin_mm
                    min_pz -= margin_mm; max_pz += margin_mm

                    # Convert Physical BBox to Target Image Indices
                    # We need to find the index range in target_image that covers this physical box.
                    # Transform Physical Point -> Target Index

                    # Check 8 corners of the physical box to be safe against rotation?
                    # Or simpler: transform min/max point?
                    # Simpler strategy: Transform Physical Min/Max to continuous index, then floor/ceil.

                    # Target Image
                    try:
                         # target_image is already loaded
                         idx_1 = target_image.TransformPhysicalPointToIndex((min_px, min_py, min_pz))
                         idx_2 = target_image.TransformPhysicalPointToIndex((max_px, max_py, max_pz))

                         # Because of direction, idx_1 might be > idx_2. Sort them.
                         x_indices = sorted([idx_1[0], idx_2[0]])
                         y_indices = sorted([idx_1[1], idx_2[1]])
                         z_indices = sorted([idx_1[2], idx_2[2]])

                         x_min, x_max = x_indices[0], x_indices[1]
                         y_min, y_max = y_indices[0], y_indices[1]
                         z_min, z_max = z_indices[0], z_indices[1]

                         # Clamp to image bounds
                         size = target_image.GetSize()
                         x_min = max(0, x_min); x_max = min(size[0], x_max)
                         y_min = max(0, y_min); y_max = min(size[1], y_max)
                         z_min = max(0, z_min); z_max = min(size[2], z_max)

                         # bbox format: [x_min, x_max, y_min, y_max, z_min, z_max]
                         dose_bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
                         logging.info(f"Dose-Based Optimization: Physical BBox mapped to Target Indices: X[{x_min}:{x_max}], Y[{y_min}:{y_max}], Z[{z_min}:{z_max}]")

                    except Exception as e:
                         logging.warning(f"Failed to map physical dose bbox to target image: {e}. Using full volume.")
                         dose_bbox = None

                else:
                    logging.warning("Dose file found but empty. Using full volume.")
            except Exception as e:
                logging.warning(f"Failed to load dose for optimization: {e}. Using full volume.")
        else:
            logging.info("No dose.nii.gz found. Using full volume.")


        # Fast path: static cone from source with square field at isocenter
        if use_static_cylinder:
            iso = _get_isocenter_from_plan(plan)
            if iso is not None:
                # Get SAD from first treatment beam
                SAD = 1000.0  # Default
                # Extract Gantry, Couch, Collimator Angles
                # List of (g, s, c) tuples
                angles_list = []

                # Initialize total beam image accumulator
                total_beam_img = None

                # List to store beam metadata for logging
                beam_infos_log = []

                output_path = join(patient_folder, "Nifti", f"beam_{getattr(plan, 'BeamSequence', [])[0].BeamNumber if getattr(plan, 'BeamSequence', []) else '0'}.nii.gz")
                # We need a proper output path to derive the json log path.
                # The caller usually handles saving, but here we return the image.
                # The caller saves `beam_img` to `beam_{i}.nii.gz`.
                # We can write the JSON log based on that index?
                # The function signature doesn't have the Plan Index.
                # We can write to `beam_info_{SOPInstanceUID}.json` or similar.
                # Let's write to "beam_info.json" in the Nifti folder, appending?
                # Or better: `beam_info_{PlanName}.json`.

                plan_name = getattr(plan, "RTPlanLabel", "Plan")
                # Sanitize
                plan_name = "".join([c for c in plan_name if c.isalnum() or c in (' ','-','_')]).strip()
                log_filename = f"beam_info_{plan_name}.json"
                log_path = join(patient_folder, "Nifti", log_filename)

                for b_idx, beam in enumerate(getattr(plan, "BeamSequence", [])):
                     if getattr(beam, "TreatmentDeliveryType", None) == "TREATMENT":
                         SAD = float(getattr(beam, "SourceAxisDistance", SAD))

                         # Filter for TREATMENT beams only?
                         # Usually we only want the beams contributing to Dose.
                         delivery_type = getattr(beam, "TreatmentDeliveryType", "SETUP")
                         if delivery_type != "TREATMENT":
                             continue

                         # Use this beam's isocenter (plans can have multiple beams with different isocenters)
                         iso_beam = _get_isocenter_from_beam(beam)
                         if iso_beam is None:
                             iso_beam = iso  # fallback to plan's first-beam isocenter
                             logging.warning("Beam %s has no IsocenterPosition; using plan fallback %s", getattr(beam, "BeamName", b_idx), iso_beam)

                         beam_name = getattr(beam, "BeamName", f"Beam_{getattr(beam,'BeamNumber', '?')}")

                         # Beam-level setup
                         # Start with Beam's ControlPoint 0 state + subsequent CPs

                         # State variables
                         current_g = 0.0
                         current_s = 0.0
                         current_c = 0.0

                         beam_angles_list = []

                         for i, cp in enumerate(getattr(beam, "ControlPointSequence", [])):
                             # Update state if attribute exists
                             if "GantryAngle" in cp:
                                 current_g = float(cp.GantryAngle)
                             if "PatientSupportAngle" in cp:
                                 current_s = float(cp.PatientSupportAngle)
                             if "BeamLimitingDeviceAngle" in cp:
                                 current_c = float(cp.BeamLimitingDeviceAngle)

                             beam_angles_list.append((current_g, current_s, current_c))

                         if not beam_angles_list:
                             beam_angles_list = [(0.0, 0.0, 0.0)]

                         # Compute statistics for logging
                         g_vals = [a[0] for a in beam_angles_list]
                         s_vals = [a[1] for a in beam_angles_list]
                         c_vals = [a[2] for a in beam_angles_list]

                         beam_meta = {
                             "BeamName": beam_name,
                             "BeamNumber": getattr(beam, "BeamNumber", b_idx),
                             "DeliveryType": delivery_type,
                             "SAD": SAD,
                             "Isocenter": list(iso_beam),
                             "ControlPoints": len(beam_angles_list),
                             "GantryRange": [min(g_vals), max(g_vals)],
                             "SupportRange": [min(s_vals), max(s_vals)],
                             "CollimatorRange": [min(c_vals), max(c_vals)],
                             "Angles": beam_angles_list  # Full list
                         }
                         beam_infos_log.append(beam_meta)

                         logging.info(f"Generating IEC Beam '{beam_name}': Iso={iso_beam}, SAD={SAD}, ControlPoints={len(beam_angles_list)}")

                         # Generate mask for THIS beam at THIS beam's isocenter
                         # Use binary generation (0 or 1)
                         single_beam_mask = generate_beam_cone_pytorch_iec(
                             target_image,
                             iso_beam,
                             sad_mm=SAD,
                             field_size_mm=static_cylinder_radius_mm*2.0,
                             angles_deg_list=beam_angles_list,
                             dose_bbox=dose_bbox
                         )

                         # Add to accumulation image
                         # sitk.Add requires same type. generate_.. returns UInt8.
                         # We allows summation > 1.

                         if total_beam_img is None:
                             total_beam_img = single_beam_mask
                         else:
                             total_beam_img = sitk.Add(total_beam_img, single_beam_mask)

                # Save Log
                try:
                    with open(log_path, 'w') as f:
                        json.dump(beam_infos_log, f, indent=4)
                    logging.info(f"Saved beam info to {log_path}")
                except Exception as e:
                    logging.error(f"Failed to save beam info log: {e}")

                if total_beam_img is None:
                    # No beams found
                    logging.warning("No valid treatment beams found in plan.")
                    return None, None

                return total_beam_img, target_image_ori
            # fallback to full path if isocenter not found
        ct_file = join(patient_folder, "Dicom", "CT")
        ct_file = join(ct_file, os.listdir(ct_file)[0])
        ct = pydicom.dcmread(ct_file)
        full_size_arr = np.zeros_like(np.asarray(target_array))
        non_zero_z = np.where(np.any(target_array, axis=(1, 2)))[0]
        z_min = max(0, non_zero_z.min() - 10)
        z_max = min(non_zero_z.max() + 10, target_array.shape[0] - 1)
        target_image_cropped = target_image[:, :, z_min:z_max + 1]
        target_image_cropped.SetOrigin([
            target_image.GetOrigin()[0], target_image.GetOrigin()[1],
            target_image.GetOrigin()[2] + z_min * target_image.GetSpacing()[2] * int(target_image.GetDirection()[-1])
        ])
        orientation = np.array(ct.ImageOrientationPatient).reshape(2, 3)
        normal_vector = np.cross(orientation[0], orientation[1])
        orientation = np.vstack([orientation, normal_vector])
        beam_infos = set()
        for beam in plan.BeamSequence:
            if beam.TreatmentDeliveryType != "TREATMENT":
                continue
            beam_class = Beam(
                beam, orientation, ct.ImagePositionPatient, target_image_cropped.GetSpacing(),
                static_beam=beam.BeamType == "STATIC", use_field_only=use_field_only
            )
            beam_infos |= beam_class.get_beam_points_return_combination(beam_class.beam)
        if not beam_infos:
            return None, None
        beams = get_beams_image_pool(target_img=target_image_cropped, beam_infos=beam_infos, num_process=num_process)
        full_size_arr[z_min:z_max + 1, :, :] = sitk.GetArrayFromImage(beams)
        final_beams = sitk.GetImageFromArray(full_size_arr)
        final_beams.SetOrigin(target_image.GetOrigin())
        final_beams.SetSpacing(target_image.GetSpacing())
        final_beams.SetDirection(target_image.GetDirection())
        return final_beams, target_image_ori
    except Exception as e:
        logging.info("Beam gen failed %s: %s", plan_path, e)
        return None, None


def main_process_multi_plan(Main_folder, num_process=5, use_field_only=False, use_static_cylinder=False, static_cylinder_radius_mm=200.0):
    """Generate one beam file per plan (beam_000.nii.gz, beam_001.nii.gz) using dose_manifest.json. n dose = n beam."""
    patients = sorted(os.listdir(Main_folder))
    logging.info(f"Starting beam generation for {len(patients)} patients in {Main_folder}")
    for i, patient in enumerate(patients):
        logging.info(f"Processing patient {i+1}/{len(patients)}: {patient}")
        patient_path = join(Main_folder, patient)
        if not os.path.isdir(patient_path):
            continue
        patient_folder = patient_path
        nifti_folder = join(patient_folder, "Nifti")
        manifest_path = join(nifti_folder, "dose_manifest.json")
        try:
            if os.path.isfile(manifest_path):
                manifest = load_json(manifest_path)
                doses = manifest.get("doses", [])
                for idx, d in enumerate(doses):
                    ref_uid = d.get("referenced_rt_plan_uid")
                    rp_path = _find_rp_path_by_uid(patient_folder, ref_uid) if ref_uid else None
                    if not rp_path and idx == 0:
                        rp_dir = join(patient_folder, "Dicom", "RP")
                        if os.path.isdir(rp_dir):
                            files = sorted([f for f in os.listdir(rp_dir) if f.lower().endswith((".dcm", ".dicom"))])
                            if files:
                                rp_path = join(rp_dir, files[0])
                    if rp_path:
                        beams_img, _ = _generate_beam_for_one_plan(
                            patient_folder, rp_path, num_process, use_field_only,
                            use_static_cylinder=use_static_cylinder, static_cylinder_radius_mm=static_cylinder_radius_mm,
                        )
                        if beams_img is not None:
                            out_path = join(nifti_folder, "beam_%03d.nii.gz" % idx)
                            sitk.WriteImage(beams_img, out_path)
            else:
                rp_dir = join(patient_folder, "Dicom", "RP")
                if not os.path.isdir(rp_dir):
                    continue
                rp_path = join(rp_dir, os.listdir(rp_dir)[0])
                beams_img, _ = _generate_beam_for_one_plan(
                    patient_folder, rp_path, num_process, use_field_only,
                    use_static_cylinder=use_static_cylinder, static_cylinder_radius_mm=static_cylinder_radius_mm,
                )
                if beams_img is not None:
                    sitk.WriteImage(beams_img, join(nifti_folder, "beam.nii.gz"))
                    shutil.copy2(join(nifti_folder, "beam.nii.gz"), join(nifti_folder, "beam_000.nii.gz"))
        except Exception as e:
            logging.info("Multi-plan beam failed %s: %s", patient_folder, e)


def main_process(Main_folder, num_process=5, use_field_only=False):
    for i, patient in enumerate(sorted(os.listdir(Main_folder))):
        patient_path = join(Main_folder, patient)
        if not os.path.isdir(patient_path):
            continue
        try:
            patient_folder = patient_path
            logging.info(patient_folder)

            plan_file = join(patient_folder,"Dicom","RP")
            plan_file = join(plan_file,os.listdir(plan_file)[0])
            if not os.path.exists(plan_file):
                print(os.listdir(patient_folder))
                move_all_files_and_folders(src_dir=patient_folder,dest_dir=patient_folder.replace("Tr", "no_plan"))
                continue #return 0 #on cheha
            plan = pydicom.dcmread(plan_file)

            ct_file = join(patient_folder,"Dicom","CT")
            ct_file = join(ct_file,os.listdir(ct_file)[0])
            if not os.path.exists(ct_file):
                print(os.listdir(patient_folder))
                move_all_files_and_folders(src_dir=patient_folder,dest_dir=patient_folder.replace("Tr", "no_ct"))
                continue #return 0 #on cheha
            ct = pydicom.dcmread(ct_file)

            ptv_file = join(patient_folder,"Nifti","combined_target.nii.gz")
            if not os.path.exists(ptv_file):
                print(os.listdir(patient_folder))
                move_all_files_and_folders(src_dir=patient_folder,dest_dir=patient_folder.replace("Tr", "no_target"))
                continue #return 0 #on cheha
            target_image_ori=sitk.ReadImage(ptv_file)

            target_array = sitk.GetArrayFromImage(target_image_ori)
            target_array = target_array>0
            target_array = target_array.astype(np.uint16)
            target_image = sitk.GetImageFromArray(target_array)
            target_image.CopyInformation(target_image_ori)
            full_size_arr = np.zeros_like(target_array)

            non_zero_z = np.where(np.any(target_array, axis=(1, 2)))[0]
            z_min = max(0, non_zero_z.min() - 10)  # Ensure z_min is not less than 0
            z_max = min(non_zero_z.max() + 10, target_array.shape[0] - 1)  # Ensure z_max is within array bounds

            # print("cropped min max",z_min,z_max)
            target_image_cropped = target_image[:, :, z_min:z_max + 1]


            orientation=np.array(ct.ImageOrientationPatient).reshape(2, 3)
            normal_vector = np.cross(orientation[0], orientation[1])
            orientation = np.vstack([orientation, normal_vector])
            # if len(ct.ImageOrientationPatient) == 6:
            #     orientation =np.concatenate((orientation, np.array([0,0,1])), axis=None)
            target_image_cropped.SetOrigin([target_image.GetOrigin()[0],target_image.GetOrigin()[1],target_image.GetOrigin()[2]+z_min*target_image.GetSpacing()[2]*int(target_image.GetDirection()[-1])])

            # print(target_image_cropped.GetOrigin())
            beams = None
            beam_infos = set()
            # new_patient_origin = [float(ct.ImagePositionPatient[0]),float(ct.ImagePositionPatient[1]),\
            # #     float(ct.ImagePositionPatient[2])+z_min*target_image.GetSpacing()[2]*int(target_image.GetDirection()[-1])]
            # target_image_cropped_before_resmple =target_image_cropped
            # target_image_cropped = resample_to_high_res(target_image_cropped)
            # print(target_image_cropped_before_resmple.GetOrigin(),target_image_cropped.GetOrigin())
            for beam in plan.BeamSequence:

                if beam.TreatmentDeliveryType !="TREATMENT":
                    continue

               # (beam:pydicom.dataset.Dataset, image_orientation, image_orientation_patient, spacing)
                # try:
                # target_image_cropped_before_resample =target_image_cropped
                # target_image_cropped= resample_image(target_image_cropped, ref_image = None)
                # beam_class=Beam(beam,orientation,target_image_cropped.GetOrigin(),target_image_cropped.GetSpacing(),static_beam = beam.BeamType == "STATIC")
                beam_class=Beam(beam,orientation,ct.ImagePositionPatient,target_image_cropped.GetSpacing(),static_beam = beam.BeamType == "STATIC", use_field_only=use_field_only)
                beam_infos = beam_infos| beam_class.get_beam_points_return_combination(beam_class.beam)
                # # logging.info(f'... running total {len(beam.ControlPointSequence)}  control point of {beam.BeamType} # ')
                # break #NOTE: Debug per beam
            # logging.info(f'... running total {len(beam_infos)} control point # ')
            # try: #TODO: change to while loop retrying
            beams = get_beams_image_pool(target_img=target_image_cropped,beam_infos=beam_infos,num_process=num_process)
            # except:
            #     try:
            #         beams = get_beams_image_pool(target_img=target_image_cropped,beam_infos=beam_infos,num_process=int(num_process//2))
            #     except:
            #         move_all_files_and_folders(src_dir=patient_folder,dest_dir=patient_folder.replace("Tr", "Trr"))
                    # FAIL_LIST.append(patient)
                    # logging.info(f'FAIL {patient} ')
            # sigma = 0.1 # Standard deviation of the Gaussian kernel
            # beams = sitk.SmoothingRecursiveGaussian(beams, sigma)
            # beams=resample_back(beams, target_image_cropped_before_resmple)
            full_size_arr[z_min:z_max + 1, :, :] = sitk.GetArrayFromImage(beams)
            final_beams = sitk.GetImageFromArray(full_size_arr)
            final_beams.SetOrigin(target_image.GetOrigin())
            final_beams.SetSpacing(target_image.GetSpacing())
            final_beams.SetDirection(target_image.GetDirection())

            # sitk.WriteImage(beams,join(patient_folder,"Nifti","Beam.nii.gz"))
            sitk.WriteImage(final_beams,join(patient_folder,"Nifti","beam.nii.gz"))
        except Exception as e:
            move_all_files_and_folders(src_dir=patient_folder,dest_dir=patient_folder.replace("Tr", "Trr"))
            logging.info(f'Fail!!!!!!!!!{patient}, error {e}')

    # logging.info(f'These are the cases that FAIL {FAIL_LIST}, please set a much smaller number of process and try again.')


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-nndose_dicom', type=str,default=None,
                        help="nndose_dicom")

    parser.add_argument('-name', type=str,
                        help="dataset_name")
    parser.add_argument('-trts', type=str,default="Tr",
                        help="Tr or Ts")
    parser.add_argument('-np', type=int, default=5,
                        help="num of process")
    parser.add_argument('--use_field_only', action='store_true',
                        help="Enforce field-based (MLC/jaw) cone instead of BEV projection, even for VMAT.")
    args = parser.parse_args()



    if args.nndose_dicom is None:
        nndose_dicom = os.environ.get("nnDose_dicom")
        if not nndose_dicom:
            parser.error("provide -nndose_dicom or set the nnDose_dicom environment variable")
    else:
        nndose_dicom = args.nndose_dicom

    MAIN_DIR= os.path.join(nndose_dicom,args.name,args.trts)
    main_process(Main_folder=MAIN_DIR, num_process=args.np, use_field_only=getattr(args, "use_field_only", False))
