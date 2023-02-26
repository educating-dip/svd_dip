import os
from functools import partial
import odl
from odl.contrib.torch import OperatorModule
import torch
import numpy as np
import h5py
from pydicom.filereader import dcmread
import scipy
from skimage.transform import resize
from .ellipses import EllipsesDataset, DiskDistributedEllipsesDataset, DiskDistributedNoiseMasksDataset, EllipsoidsInBallDataset
from .rectangles import RectanglesDataset
from .pascal_voc import PascalVOCDataset
from .brain import ACRINFMISOBrainDataset
from .lodopab import LoDoPaBGroundTruthDataset
from . import lotus
from . import walnuts
from util.matrix_ray_trafo import MatrixRayTrafo
from util.matrix_ray_trafo_torch import get_matrix_ray_trafo_module
from util.matrix_fbp_torch import get_matrix_fbp_module
from util.fbp import FBP
from util.torch_linked_ray_trafo import TorchLinkedRayTrafoModule
from util.matrix_ray_trafo_torch import MatrixModule

def subsample_angles_ray_trafo_matrix(matrix, cfg, proj_shape, order='C'):
    prod_im_shape = matrix.shape[1]

    matrix = matrix.reshape(
            (cfg.num_angles_orig, proj_shape[1] * prod_im_shape),
            order=order).tocsc()

    matrix = matrix[cfg.start:cfg.stop:cfg.step, :]

    matrix = matrix.reshape((np.prod(proj_shape), prod_im_shape),
                            order=order).tocsc()
    return matrix


def load_ray_trafo_matrix(name, cfg):

    if name in ['ellipses_lotus', 'ellipses_lotus_20',
                'ellipses_lotus_limited_45',
                'rectangles_lotus_20',
                'pascal_voc_lotus_20']:
        matrix = lotus.get_ray_trafo_matrix(cfg.ray_trafo_filename)
    # elif name == 'brain_walnut_120':  # currently useless as we can't use the
                                        # matrix impl for the walnut ray trafo,
                                        # because the filtering for FDK is not
                                        # implemented
    #     matrix = walnuts.get_masked_ray_trafo_matrix(cfg.ray_trafo_filename)
    else:
        raise NotImplementedError

    return matrix

def assemble_gaussian_blur_matrix(cfg):
    x = np.zeros((cfg.im_shape, cfg.im_shape))
    matrix = np.zeros(((cfg.im_shape * cfg.im_shape), (cfg.im_shape * cfg.im_shape)), dtype=np.float32)
    for i in range(cfg.im_shape):
        for j in range(cfg.im_shape):
            x[i, j] = 1.
            matrix[:, np.ravel_multi_index((i, j), (cfg.im_shape, cfg.im_shape))] = scipy.ndimage.gaussian_filter(
                    x, 
                    cfg.geometry_specs.kernel_size
                    ).flatten()
            x[i, j] = 0.
    return matrix

def get_ray_trafos(name, cfg, return_torch_module=True, return_torch_module_adjoint=False, return_torch_module_pinv=False):
    """
    Return callables evaluating the ray transform and the smooth filtered
    back-projection for a standard dataset.

    The ray trafo can be implemented either by a matrix, which is loaded by
    calling :func:`load_ray_trafo_matrix`, or an odl `RayTransform` is used, in
    which case a standard cone-beam geometry is created.

    Subsampling of angles is supported for the matrix implementation only.

    Optionally, a ray transform torch module can be returned, too.

    Returns
    -------
    ray_trafos : dict
        Dictionary with the entries `'ray_trafo'`, `'smooth_pinv_ray_trafo'`,
        and optionally `'ray_trafo_module'`.
    """

    ray_trafos = {}

    if cfg.geometry_specs.impl == 'matrix':

        # handle special cases first, will use load_ray_trafo_matrix() for other cases
        if name == 'ellipses_lotus_gaussian_blurring':
            
            proj_shape = (cfg.im_shape, cfg.im_shape)
            if cfg.geometry_specs.load_matrix_from_path is None: 
                matrix = assemble_gaussian_blur_matrix(cfg)
            else:
                matrix = np.load(os.path.join(cfg.geometry_specs.load_matrix_from_path, 
                    f'matrix_lotus_blur_{cfg.geometry_specs.kernel_size}.npy'))

            matrix[np.diag_indices(matrix.shape[0])] += np.diag(np.abs(matrix)).mean() * cfg.geometry_specs.diag_reg
            ray_trafo = lambda x: (matrix @ x.reshape(-1)).reshape((cfg.im_shape, cfg.im_shape)).astype(np.float32)
            ray_trafos['ray_trafo'] = ray_trafo
            U, S, Vh = scipy.linalg.svd(matrix)
            pinv_matrix_np = Vh.T @ np.diag(S**-1) @ U.T
            print(f'\n Gaussian Blur Matrix Condition Number: {np.max(S) / np.min(S)}\n')
            exact_pinv_ray_trafo = lambda x: (pinv_matrix_np @ x.reshape(-1)).reshape((cfg.im_shape, cfg.im_shape)).astype(np.float32)
            ray_trafos['smooth_pinv_ray_trafo'] = exact_pinv_ray_trafo

            if return_torch_module:
                ray_trafos['ray_trafo_module'] = get_matrix_ray_trafo_module(
                        matrix, (cfg.im_shape, cfg.im_shape), proj_shape,
                        sparse=False)
            if return_torch_module_adjoint:
                ray_trafos['ray_trafo_module_adjoint'] = get_matrix_ray_trafo_module(
                        matrix, (cfg.im_shape, cfg.im_shape), proj_shape,
                        adjoint=True, sparse=False)
            if return_torch_module_pinv:
                pinv_matrix = torch.from_numpy(pinv_matrix_np)
                ray_trafos['exact_pinv_ray_trafo_module'] = MatrixModule(pinv_matrix, out_shape=(cfg.im_shape, cfg.im_shape), sparse=False)
                ray_trafos['smooth_pinv_ray_trafo_module'] = ray_trafos['exact_pinv_ray_trafo_module']
            
        elif name == 'ellipses_lotus_gaussian_denoising':

            class IdModule(torch.nn.Module):
                def forward(self, x):
                    return x

            ray_trafos['ray_trafo'] = lambda x: x.astype(np.float32)
            ray_trafos['smooth_pinv_ray_trafo'] = lambda x: x.astype(np.float32)
            
            ray_trafos['ray_trafo_module'] = torch.nn.Identity()
            ray_trafos['ray_trafo_module_adjoint'] = torch.nn.Identity()
            ray_trafos['exact_pinv_ray_trafo_module'] = torch.nn.Identity()
            ray_trafos['smooth_pinv_ray_trafo_module'] = torch.nn.Identity()

        else:  # default cases using load_ray_trafo_matrix()

            matrix = load_ray_trafo_matrix(name, cfg.geometry_specs)

            proj_shape = (cfg.geometry_specs.num_angles,
                          cfg.geometry_specs.num_det_pixels)
            if 'angles_subsampling' in cfg.geometry_specs:
                matrix = subsample_angles_ray_trafo_matrix(
                        matrix, cfg.geometry_specs.angles_subsampling, proj_shape)

            matrix_ray_trafo = MatrixRayTrafo(matrix,
                    im_shape=(cfg.im_shape, cfg.im_shape),
                    proj_shape=proj_shape)

            ray_trafo = matrix_ray_trafo.apply
            ray_trafos['ray_trafo'] = ray_trafo
            smooth_pinv_ray_trafo = FBP(
                    matrix_ray_trafo.apply_adjoint, proj_shape,
                    scaling_factor=cfg.fbp_scaling_factor,
                    filter_type=cfg.fbp_filter_type,
                    frequency_scaling=cfg.fbp_frequency_scaling).apply
            ray_trafos['smooth_pinv_ray_trafo'] = smooth_pinv_ray_trafo

            if return_torch_module:
                ray_trafos['ray_trafo_module'] = get_matrix_ray_trafo_module(
                        matrix, (cfg.im_shape, cfg.im_shape), proj_shape,
                        sparse=True)
            if return_torch_module_adjoint:
                ray_trafos['ray_trafo_module_adjoint'] = get_matrix_ray_trafo_module(
                        matrix, (cfg.im_shape, cfg.im_shape), proj_shape,
                        adjoint=True, sparse=False)
            if return_torch_module_pinv:
                ray_trafos['smooth_pinv_ray_trafo_module'] = get_matrix_fbp_module(
                        get_matrix_ray_trafo_module(
                        matrix, (cfg.im_shape, cfg.im_shape), proj_shape,
                        sparse=True, adjoint=True), proj_shape,
                        scaling_factor=cfg.fbp_scaling_factor,
                        filter_type=cfg.fbp_filter_type,
                        frequency_scaling=cfg.fbp_frequency_scaling)

    elif cfg.geometry_specs.impl == 'custom':
        custom_cfg = cfg.geometry_specs.ray_trafo_custom
        if custom_cfg.name in ['walnut_single_slice',
                               'walnut_single_slice_matrix']:
            angles_subsampling = cfg.geometry_specs.angles_subsampling
            angular_sub_sampling = angles_subsampling.get('step', 1)
            # the walnuts module only supports choosing the step
            assert range(walnuts.MAX_NUM_ANGLES)[
                    angles_subsampling.get('start'):
                    angles_subsampling.get('stop'):
                    angles_subsampling.get('step')] == range(
                            0, walnuts.MAX_NUM_ANGLES, angular_sub_sampling)
            walnut_ray_trafo = walnuts.get_single_slice_ray_trafo(
                    data_path=custom_cfg.data_path,
                    walnut_id=custom_cfg.walnut_id,
                    orbit_id=custom_cfg.orbit_id,
                    angular_sub_sampling=angular_sub_sampling)
            if custom_cfg.name == 'walnut_single_slice':
                ray_trafos['ray_trafo'] = walnut_ray_trafo.apply
            elif custom_cfg.name == 'walnut_single_slice_matrix':
                matrix = walnuts.get_single_slice_ray_trafo_matrix(
                        path=custom_cfg.matrix_path,
                        walnut_id=custom_cfg.walnut_id,
                        orbit_id=custom_cfg.orbit_id,
                        angular_sub_sampling=angular_sub_sampling)
                matrix_ray_trafo = MatrixRayTrafo(matrix,
                        im_shape=(cfg.im_shape, cfg.im_shape),
                        proj_shape=(matrix.shape[0],))
                ray_trafos['ray_trafo'] = matrix_ray_trafo.apply

            # FIXME FDK is not smooth
            ray_trafos['smooth_pinv_ray_trafo'] = partial(
                    walnut_ray_trafo.apply_fdk, squeeze=True)

            if return_torch_module:
                if custom_cfg.name == 'walnut_single_slice':
                    ray_trafos['ray_trafo_module'] = (
                            walnuts.WalnutRayTrafoModule(walnut_ray_trafo))
                elif custom_cfg.name == 'walnut_single_slice_matrix':
                    ray_trafos['ray_trafo_module'] = (
                            get_matrix_ray_trafo_module(
                                    matrix, (cfg.im_shape, cfg.im_shape),
                                    (matrix.shape[0],), sparse=True))
            if return_torch_module_adjoint:
                raise NotImplementedError
            if return_torch_module_pinv:
                raise NotImplementedError
        elif custom_cfg.name == 'walnut_3d':
            vol_down_sampling = cfg.vol_down_sampling
            angles_subsampling = cfg.geometry_specs.angles_subsampling
            angular_sub_sampling = angles_subsampling.get('step', 1)
            # the walnuts module only supports choosing the step
            assert range(walnuts.MAX_NUM_ANGLES)[
                    angles_subsampling.get('start'):
                    angles_subsampling.get('stop'):
                    angles_subsampling.get('step')] == range(
                            0, walnuts.MAX_NUM_ANGLES, angular_sub_sampling)
            proj_row_sub_sampling = cfg.geometry_specs.det_row_sub_sampling
            proj_col_sub_sampling = cfg.geometry_specs.det_col_sub_sampling
            walnut_ray_trafo = walnuts.WalnutRayTrafo(
                    data_path=custom_cfg.data_path,
                    walnut_id=custom_cfg.walnut_id,
                    orbit_id=custom_cfg.orbit_id,
                    vol_down_sampling=vol_down_sampling,
                    angular_sub_sampling=angular_sub_sampling,
                    proj_row_sub_sampling=proj_row_sub_sampling,
                    proj_col_sub_sampling=proj_col_sub_sampling)
            ray_trafos['ray_trafo'] = walnut_ray_trafo.apply

            # FIXME FDK is not smooth
            ray_trafos['smooth_pinv_ray_trafo'] = walnut_ray_trafo.apply_fdk

            if return_torch_module:
                ray_trafos['ray_trafo_module'] = TorchLinkedRayTrafoModule(
                        walnut_ray_trafo.vol_geom, walnut_ray_trafo.proj_geom)
            if return_torch_module_adjoint:
                raise NotImplementedError
            if return_torch_module_pinv:
                raise NotImplementedError
        else:
            raise ValueError('Unknown custom ray trafo \'{}\''.format(
                    cfg.geometry_specs.ray_trafo_custom.name))

    else:
        phys_im_size = cfg.geometry_specs.get('phys_im_size', cfg.im_shape)  # default: 1px = 1 unit
        space = odl.uniform_discr([-phys_im_size / 2, -phys_im_size / 2],
                                  [phys_im_size / 2, phys_im_size / 2],
                                  [cfg.im_shape, cfg.im_shape],
                                  dtype='float32')
        if cfg.geometry_specs.type == 'cone':
            geometry = odl.tomo.cone_beam_geometry(space,
                    src_radius=cfg.geometry_specs.src_radius,
                    det_radius=cfg.geometry_specs.det_radius,
                    num_angles=cfg.geometry_specs.num_angles,
                    det_shape=cfg.geometry_specs.get('num_det_pixels', None))
            if 'angles_subsampling' in cfg.geometry_specs:
                raise NotImplementedError
        elif cfg.geometry_specs.type == 'parallel':
            if 'angles_subsampling' not in cfg.geometry_specs:
                geometry = odl.tomo.parallel_beam_geometry(space,
                        num_angles=cfg.geometry_specs.num_angles,
                        det_shape=cfg.geometry_specs.get('num_det_pixels', None))
            else:
                orig_geometry = odl.tomo.parallel_beam_geometry(space,
                        num_angles=cfg.geometry_specs.angles_subsampling.num_angles_orig,
                        det_shape=cfg.geometry_specs.get('num_det_pixels', None))
                geometry = odl.tomo.Parallel2dGeometry(
                        apart=odl.nonuniform_partition(orig_geometry.angles[
                                cfg.geometry_specs.angles_subsampling.get('start'):
                                cfg.geometry_specs.angles_subsampling.get('stop'):
                                cfg.geometry_specs.angles_subsampling.get('step')]),
                        dpart=orig_geometry.det_partition)
                assert len(geometry.angles) == cfg.geometry_specs.num_angles
        else:
            raise ValueError('Unknown geometry type \'{}\''.format(cfg.geometry_specs.type))

        ray_trafo = odl.tomo.RayTransform(space, geometry,
                impl=cfg.geometry_specs.impl)
        ray_trafos['ray_trafo'] = ray_trafo

        smooth_pinv_ray_trafo = odl.tomo.fbp_op(ray_trafo,
                filter_type=cfg.fbp_filter_type,
                frequency_scaling=cfg.fbp_frequency_scaling)
        ray_trafos['smooth_pinv_ray_trafo'] = smooth_pinv_ray_trafo

        if return_torch_module:
            ray_trafos['ray_trafo_module'] = OperatorModule(ray_trafo)
        if return_torch_module_adjoint:
            ray_trafos['ray_trafo_module_adjoint'] = OperatorModule(ray_trafo.adjoint)
        if return_torch_module_pinv:
            ray_trafos['smooth_pinv_ray_trafo_module'] = OperatorModule(smooth_pinv_ray_trafo)

    return ray_trafos


def get_standard_dataset(name, cfg, return_ray_trafo_torch_module=True, return_ray_trafo_torch_module_adjoint=False, return_ray_trafo_torch_module_pinv=False, **image_dataset_kwargs):
    """
    Return a standard dataset by name.
    """

    name = name.lower()

    ray_trafos = get_ray_trafos(name, cfg,
            return_torch_module=return_ray_trafo_torch_module,
            return_torch_module_adjoint=return_ray_trafo_torch_module_adjoint,
            return_torch_module_pinv=return_ray_trafo_torch_module_pinv)

    ray_trafo = ray_trafos['ray_trafo']
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    if cfg.noise_specs.noise_type == 'white':
        specs_kwargs = {'stddev': cfg.noise_specs.stddev}
    elif cfg.noise_specs.noise_type == 'poisson':
        specs_kwargs = {'mu_max': cfg.noise_specs.mu_max,
                        'photons_per_pixel': cfg.noise_specs.photons_per_pixel
                        }
    else:
        raise NotImplementedError

    if name == 'ellipses':
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = EllipsesDataset(**dataset_specs, **image_dataset_kwargs)
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo, noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name == 'ellipses_lotus':
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = EllipsesDataset(**dataset_specs, **image_dataset_kwargs)
        space = lotus.get_domain128()
        proj_space = lotus.get_proj_space128()
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name in ['ellipses_lotus_20', 'ellipses_lotus_limited_45', 'rectangles_lotus_20', 'pascal_voc_lotus_20']:
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        if name in ['ellipses_lotus_20', 'ellipses_lotus_limited_45', 'ellipses_lotus_gaussian_blur_sure', 'ellipses_lotus_gaussian_denoising_sure']:
            image_dataset = EllipsesDataset(**dataset_specs, **image_dataset_kwargs)
        elif name in ['rectangles_lotus_20']:
            image_dataset = RectanglesDataset(**dataset_specs, **image_dataset_kwargs)
        elif name in ['pascal_voc_lotus_20']:
            image_dataset = PascalVOCDataset(
                    data_path=cfg.data_path, **dataset_specs, **image_dataset_kwargs)
        else:
            raise NotImplementedError
        space = lotus.get_domain128()
        proj_space_orig = lotus.get_proj_space128()
        angles_coord_vector = proj_space_orig.grid.coord_vectors[0][
                cfg.geometry_specs.angles_subsampling.start:
                cfg.geometry_specs.angles_subsampling.stop:
                cfg.geometry_specs.angles_subsampling.step]
        proj_space = odl.uniform_discr_frompartition(
                odl.uniform_partition_fromgrid(
                        odl.RectGrid(angles_coord_vector,
                                     proj_space_orig.grid.coord_vectors[1])))
        dataset = image_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name in ['ellipses_lotus_gaussian_denoising', 'ellipses_lotus_gaussian_blurring']:
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        image_dataset = EllipsesDataset(**dataset_specs, **image_dataset_kwargs)
        class Space:
            def __init__(self, im_shape): 
                self.shape = (im_shape, im_shape)
        dataset = image_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=Space(cfg.im_shape), proj_space=Space(cfg.im_shape),
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name == 'brain_walnut_120':
        dataset_specs = {'data_path': cfg.data_path, 'shuffle': cfg.shuffle,
                         'zoom': cfg.zoom, 'zoom_fit': cfg.zoom_fit,
                         'random_rotation': cfg.random_rotation}
        brain_dataset = ACRINFMISOBrainDataset(**dataset_specs, **image_dataset_kwargs)
        space = brain_dataset.space
        proj_numel = cfg.geometry_specs.num_angles * cfg.geometry_specs.num_det_pixels
        proj_space = odl.rn(proj_numel, dtype=np.float32)
        dataset = brain_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name == 'ellipses_walnut_120':
        dataset_specs = {'diameter': cfg.disk_diameter,
                         'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = DiskDistributedEllipsesDataset(**dataset_specs, **image_dataset_kwargs)
        space = ellipses_dataset.space
        proj_numel = cfg.geometry_specs.num_angles * cfg.geometry_specs.num_det_pixels
        proj_space = odl.rn(proj_numel, dtype=np.float32)
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name == 'noise_masks_walnut_120':
        dataset_specs = {'in_circle_axis': cfg.in_circle_axis, 'use_mask': cfg.use_mask,
                        'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                        'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = DiskDistributedNoiseMasksDataset(**dataset_specs, **image_dataset_kwargs)
        space = ellipses_dataset.space
        proj_numel = cfg.geometry_specs.num_angles * cfg.geometry_specs.num_det_pixels
        proj_space = odl.rn(proj_numel, dtype=np.float32)
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name in ['ellipsoids_walnut_3d', 'ellipsoids_walnut_3d_60', 'ellipsoids_walnut_3d_down5']:
        dataset_specs = {'in_ball_axis': cfg.in_ball_axis,
                         'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipsoids_dataset = EllipsoidsInBallDataset(**dataset_specs, **image_dataset_kwargs)
        space = ellipsoids_dataset.space
        proj_space = odl.uniform_discr(  # use astra vau order
                min_pt=[-1., -np.pi, -1.], max_pt=[1., np.pi, 1.],  # dummy values
                shape=(cfg.geometry_specs.num_det_rows,
                       cfg.geometry_specs.num_angles,
                       cfg.geometry_specs.num_det_cols))
        dataset = ellipsoids_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name in ['ellipses_lodopab', 'ellipses_lodopab_200']:
        # see https://github.com/jleuschn/dival/blob/483915b2e64c1ad6355311da0429ef8f2c2eceb5/dival/datasets/lodopab_dataset.py#L78
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        image_dataset = EllipsesDataset(**dataset_specs, **image_dataset_kwargs)
        phys_im_size = cfg.geometry_specs.get(
                'phys_im_size', cfg.im_shape)  # lodopab cfg. specifies 0.26
        space = odl.uniform_discr([-phys_im_size/2,] * 2, [phys_im_size/2,] * 2,
                (cfg.im_shape,) * 2, dtype=np.float32)
        orig_geometry = odl.tomo.parallel_beam_geometry(
                space,
                num_angles=cfg.geometry_specs.num_angles,
                det_shape=(cfg.geometry_specs.num_det_pixels,))
        if name == 'ellipses_lodopab':
            geometry = orig_geometry
            proj_space = odl.uniform_discr(
                    geometry.partition.min_pt, geometry.partition.max_pt,
                    (cfg.geometry_specs.num_angles, cfg.geometry_specs.num_det_pixels), dtype=np.float32)
        elif name == 'ellipses_lodopab_200':
            # cf. https://github.com/oterobaguer/dip-ct-benchmark/blob/0539c284c94089ed86421ea0892cd68aa1d0575a/dliplib/utils/helper.py#L185
            geometry = odl.tomo.Parallel2dGeometry(
                    apart=odl.nonuniform_partition(orig_geometry.angles[
                            cfg.geometry_specs.angles_subsampling.start:
                            cfg.geometry_specs.angles_subsampling.stop:
                            cfg.geometry_specs.angles_subsampling.step]),  # lodopab_200 cfg. specifies range(0, 1000, 5)
                    dpart=orig_geometry.det_partition)
            proj_space = ray_trafo.range
        dataset = image_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name in ['lodopab_mayo', 'lodopab_mayo_200']:
        # dummy, use pretrained network from https://github.com/oterobaguer/dip-ct-benchmark/
        # see https://github.com/jleuschn/dival/blob/483915b2e64c1ad6355311da0429ef8f2c2eceb5/dival/datasets/lodopab_dataset.py#L78
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        phys_im_size = cfg.geometry_specs.get(
                'phys_im_size', cfg.im_shape)  # lodopab cfg. specifies 0.26
        image_dataset = LoDoPaBGroundTruthDataset(**dataset_specs, **image_dataset_kwargs, min_pt=[-phys_im_size/2,] * 2, max_pt=[phys_im_size/2,] * 2)
        orig_geometry = odl.tomo.parallel_beam_geometry(
                image_dataset.space,
                num_angles=cfg.geometry_specs.num_angles,
                det_shape=(cfg.geometry_specs.num_det_pixels,))
        if name == 'lodopab_mayo':
            geometry = orig_geometry
            proj_space = odl.uniform_discr(
                    geometry.partition.min_pt, geometry.partition.max_pt,
                    (cfg.geometry_specs.num_angles, cfg.geometry_specs.num_det_pixels), dtype=np.float32)
        elif name == 'lodopab_mayo_200':
            # cf. https://github.com/oterobaguer/dip-ct-benchmark/blob/0539c284c94089ed86421ea0892cd68aa1d0575a/dliplib/utils/helper.py#L185
            geometry = odl.tomo.Parallel2dGeometry(
                    apart=odl.nonuniform_partition(orig_geometry.angles[
                            cfg.geometry_specs.angles_subsampling.start:
                            cfg.geometry_specs.angles_subsampling.stop:
                            cfg.geometry_specs.angles_subsampling.step]),  # mayo_200 cfg. specifies range(0, 1000, 5)
                    dpart=orig_geometry.det_partition)
            proj_space = ray_trafo.range
        dataset = image_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=image_dataset.space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    else:
        raise NotImplementedError

    return dataset, ray_trafos


def get_test_data(name, cfg, return_torch_dataset=True):
    """
    Return external test data.

    E.g., for `'ellipses_lotus'` the scan of the lotus root is returned for
    evaluating a model trained on the `'ellipses_lotus'` standard dataset.

    Sinograms, FBPs and potentially ground truth images are returned, by
    default combined as a torch `TensorDataset` of two or three tensors.

    If `return_torch_dataset=False` is passed, numpy arrays
    ``sinogram_array, fbp_array, ground_truth_array`` are returned, where
    `ground_truth_array` can be `None` and all arrays have shape ``(N, W, H)``.
    """

    if cfg.test_data == 'lotus':
        sinogram, fbp, ground_truth = get_lotus_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]
        ground_truth_array = (ground_truth[None] if ground_truth is not None
                              else None)
    elif cfg.test_data == 'walnut':
        sinogram, fbp, ground_truth = get_walnut_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]  # FDK, actually
        ground_truth_array = ground_truth[None]
    elif cfg.test_data == 'walnut_3d':
        sinogram, fbp, ground_truth = get_walnut_3d_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]  # FDK, actually
        ground_truth_array = ground_truth[None]
    elif cfg.test_data == 'lodopab':
        sinogram, fbp, ground_truth = get_lodopab_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]
        ground_truth_array = ground_truth[None]
    elif cfg.test_data == 'mayo':
        sinogram, fbp, ground_truth = get_mayo_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]
        ground_truth_array = ground_truth[None]
    else:
        raise NotImplementedError

    if return_torch_dataset:
        if ground_truth_array is not None:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]),
                        torch.from_numpy(ground_truth_array[:, None]))
        else:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]))

        return dataset
    else:
        return sinogram_array, fbp_array, ground_truth_array


def get_validation_data(name, cfg, return_torch_dataset=True):
    """
    Return external validation data.

    E.g., for `'ellipses_lotus'` data of the Shepp-Logan phantom is returned
    for validating a model trained on the `'ellipses_lotus'` standard dataset.

    Sinograms, FBPs and potentially ground truth images are returned, by
    default combined as a torch `TensorDataset` of two or three tensors.

    If `return_torch_dataset=False` is passed, numpy arrays
    ``sinogram_array, fbp_array, ground_truth_array`` are returned, where
    `ground_truth_array` can be `None` and all arrays have shape ``(N, W, H)``.
    """

    if cfg.validation_data == 'shepp_logan':
        sinogram, fbp, ground_truth = get_shepp_logan_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]
        ground_truth_array = (ground_truth[None] if ground_truth is not None
                              else None)
    else:
        raise NotImplementedError

    if return_torch_dataset:
        if ground_truth_array is not None:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]),
                        torch.from_numpy(ground_truth_array[:, None]))
        else:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]))

        return dataset
    else:
        return sinogram_array, fbp_array, ground_truth_array


def get_lotus_data(name, cfg):

    ray_trafos = get_ray_trafos(name, cfg,
                                return_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']
    ground_truth = None
    if cfg.ground_truth_filename is not None:
        ground_truth = lotus.get_ground_truth(cfg.ground_truth_filename)
        if cfg.im_shape != ground_truth.shape[0]:
            if name in ['ellipses_lotus_gaussian_blurring', 'ellipses_lotus_gaussian_denoising']:
                ground_truth = resize(ground_truth, (cfg.im_shape, cfg.im_shape))
            else:
                raise ValueError('Image resizing is not supported for data "{}"; ground truth has side length {}, but data.im_shape={} is requested'.format(name, ground_truth.shape[0], cfg.im_shape))
    if name in ['ellipses_lotus_gaussian_blurring', 'ellipses_lotus_gaussian_denoising']:
        sinogram = ray_trafos['ray_trafo'](ground_truth)
        sinogram += np.random.normal(size=sinogram.shape) * np.mean(sinogram) * cfg.noise_specs.stddev
    else:
        sinogram = np.asarray(lotus.get_sinogram(
                        cfg.geometry_specs.ray_trafo_filename))
        if 'angles_subsampling' in cfg.geometry_specs:
            sinogram = sinogram[cfg.geometry_specs.angles_subsampling.start:
                                cfg.geometry_specs.angles_subsampling.stop:
                                cfg.geometry_specs.angles_subsampling.step, :]
    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))
    return sinogram, fbp, ground_truth


def get_walnut_data(name, cfg):

    ray_trafos = get_ray_trafos(name, cfg,
                                return_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    angles_subsampling = cfg.geometry_specs.angles_subsampling
    angular_sub_sampling = angles_subsampling.get('step', 1)
    # the walnuts module only supports choosing the step
    assert range(walnuts.MAX_NUM_ANGLES)[
            angles_subsampling.get('start'):
            angles_subsampling.get('stop'):
            angles_subsampling.get('step')] == range(
                    0, walnuts.MAX_NUM_ANGLES, angular_sub_sampling)

    sinogram_full = walnuts.get_projection_data(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            angular_sub_sampling=angular_sub_sampling)

    # MaskedWalnutRayTrafo instance needed for selecting and masking the projections
    walnut_ray_trafo = walnuts.get_single_slice_ray_trafo(
            cfg.geometry_specs.ray_trafo_custom.data_path,
            walnut_id=cfg.geometry_specs.ray_trafo_custom.walnut_id,
            orbit_id=cfg.geometry_specs.ray_trafo_custom.orbit_id,
            angular_sub_sampling=angular_sub_sampling)

    sinogram = walnut_ray_trafo.flat_projs_in_mask(
            walnut_ray_trafo.projs_from_full(sinogram_full))

    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    slice_ind = walnuts.get_single_slice_ind(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id)
    ground_truth = walnuts.get_ground_truth(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            slice_ind=slice_ind)

    return sinogram, fbp, ground_truth


def get_walnut_3d_data(name, cfg):

    ray_trafos = get_ray_trafos(name, cfg,
                                return_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    angles_subsampling = cfg.geometry_specs.angles_subsampling
    angular_sub_sampling = angles_subsampling.get('step', 1)
    # the walnuts module only supports choosing the step
    assert range(walnuts.MAX_NUM_ANGLES)[
            angles_subsampling.get('start'):
            angles_subsampling.get('stop'):
            angles_subsampling.get('step')] == range(
                    0, walnuts.MAX_NUM_ANGLES, angular_sub_sampling)

    sinogram = walnuts.get_projection_data(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_row_sub_sampling=cfg.geometry_specs.det_row_sub_sampling,
            proj_col_sub_sampling=cfg.geometry_specs.det_col_sub_sampling)

    # MaskedWalnutRayTrafo instance needed for selecting and masking the projections
    walnut_ray_trafo = walnuts.WalnutRayTrafo(
            cfg.geometry_specs.ray_trafo_custom.data_path,
            walnut_id=cfg.geometry_specs.ray_trafo_custom.walnut_id,
            orbit_id=cfg.geometry_specs.ray_trafo_custom.orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_row_sub_sampling=cfg.geometry_specs.det_row_sub_sampling,
            proj_col_sub_sampling=cfg.geometry_specs.det_col_sub_sampling,
            vol_down_sampling=cfg.vol_down_sampling)

    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    ground_truth_orig_res = walnuts.get_ground_truth_3d(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id)
    ground_truth = walnuts.down_sample_vol(ground_truth_orig_res,
            down_sampling=cfg.vol_down_sampling)

    return sinogram, fbp, ground_truth


def get_lodopab_data(name, cfg, base_seed=100):

    ray_trafos = get_ray_trafos(name, cfg,
                                return_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    angles_subsampling = cfg.geometry_specs.get('angles_subsampling', {})

    NUM_SAMPLES_PER_FILE = 128
    file_index = cfg.sample_index // NUM_SAMPLES_PER_FILE
    index_in_file = cfg.sample_index % NUM_SAMPLES_PER_FILE

    with h5py.File(
            os.path.join(cfg.data_path_test,
                         'observation_{}_{:03d}.hdf5'
                         .format(cfg.data_part, file_index)), 'r') as file:
        sinogram = file['data'][index_in_file][
                angles_subsampling.get('start'):
                angles_subsampling.get('stop'):
                angles_subsampling.get('step')]

    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    with h5py.File(
            os.path.join(cfg.data_path_test,
                         'ground_truth_{}_{:03d}.hdf5'
                         .format(cfg.data_part, file_index)), 'r') as file:
        ground_truth = file['data'][index_in_file]

    if cfg.resimulate_with_noise_specs:
        dataset, _ = get_standard_dataset(
                name, cfg, return_ray_trafo_torch_module=False)
        sinogram = dataset.ground_truth_to_obs(
                ground_truth, random_gen=np.random.default_rng(base_seed + cfg.sample_index))
        fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    return sinogram, fbp, ground_truth

def get_mayo_data(name, cfg, base_seed=100):

    ray_trafos = get_ray_trafos(name, cfg,
                                return_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    angles_subsampling = cfg.geometry_specs.get('angles_subsampling', {})

    dirs = os.listdir(os.path.join(cfg.data_path_test, cfg.sample_name))
    assert len(dirs) == 1
    full_dose_images_dirs = [d for d in os.listdir(os.path.join(cfg.data_path_test, cfg.sample_name, dirs[0])) if 'Full Dose Images' in d]
    assert len(full_dose_images_dirs) == 1
    path = os.path.join(cfg.data_path_test, cfg.sample_name, dirs[0], full_dose_images_dirs[0])
    dcm_files = os.listdir(path)
    dcm_files.sort(key=lambda f: float(dcmread(os.path.join(path, f), specific_tags=['SliceLocation'])['SliceLocation'].value))
    sample_slice = (len(dcm_files) - 1) // 2 if cfg.sample_slice == 'center' else cfg.sample_slice
    dcm_dataset = dcmread(os.path.join(path, dcm_files[sample_slice]))

    rng = np.random.default_rng((base_seed + hash((cfg.sample_name, sample_slice))) % (2**64))

    # reduce to (362 px)^2 and transpose like in LoDoPaB
    array = dcm_dataset.pixel_array[75:-75,75:-75].astype(np.float32).T
    # rescale by dicom meta info
    array *= dcm_dataset.RescaleSlope
    array += dcm_dataset.RescaleIntercept
    array += rng.uniform(0., 1., size=array.shape)
    # convert values
    MU_WATER = 20
    MU_AIR = 0.02
    MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    array *= (MU_WATER - MU_AIR) / 1000
    array += MU_WATER
    array /= MU_MAX
    np.clip(array, 0., 1., out=array)

    ground_truth = array

    dataset, _ = get_standard_dataset(
            name, cfg, return_ray_trafo_torch_module=False)
    sinogram = dataset.ground_truth_to_obs(
            ground_truth, random_gen=rng)
    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    return sinogram, fbp, ground_truth


def get_shepp_logan_data(name, cfg, modified=True, seed=30):

    dataset, ray_trafos = get_standard_dataset(
            name, cfg, return_ray_trafo_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    zoom = cfg.get('zoom', 1.)
    if zoom == 1.:
        ground_truth = odl.phantom.shepp_logan(dataset.space[1],
                                               modified=modified)
    else:
        full_shape = dataset.space[1].shape
        if len(full_shape) == 2:
            inner_shape = (int(zoom * dataset.space[1].shape[0]),
                           int(zoom * dataset.space[1].shape[1]))
            inner_space = odl.uniform_discr(
                    min_pt=[-inner_shape[0] / 2, -inner_shape[1] / 2],
                    max_pt=[inner_shape[0] / 2, inner_shape[1] / 2],
                    shape=inner_shape)
            inner_ground_truth = odl.phantom.shepp_logan(
                    inner_space, modified=modified)
            ground_truth = dataset.space[1].zero()
            i0_start = (full_shape[0] - inner_shape[0]) // 2
            i1_start = (full_shape[1] - inner_shape[1]) // 2
            ground_truth[i0_start:i0_start+inner_shape[0],
                        i1_start:i1_start+inner_shape[1]] = inner_ground_truth
        elif len(full_shape) == 3:
            # dataset.space[1] uses zyx order (ASTRA convention);
            # for inner_space_odl, use xyz instead (ODL convention)
            inner_shape = (int(zoom * dataset.space[1].shape[0]),
                           int(zoom * dataset.space[1].shape[1]),
                           int(zoom * dataset.space[1].shape[2]))
            inner_space_odl = odl.uniform_discr(
                    min_pt=[-inner_shape[2] / 2, -inner_shape[1] / 2, -inner_shape[0] / 2],
                    max_pt=[inner_shape[2] / 2, inner_shape[1] / 2, inner_shape[0] / 2],
                    shape=inner_shape[::-1])
            inner_ground_truth = np.transpose(
                    odl.phantom.shepp_logan(inner_space_odl, modified=modified), (2,1,0))
            ground_truth = dataset.space[1].zero()
            i0_start = (full_shape[0] - inner_shape[0]) // 2
            i1_start = (full_shape[1] - inner_shape[1]) // 2
            i2_start = (full_shape[2] - inner_shape[2]) // 2
            ground_truth[i0_start:i0_start+inner_shape[0],
                         i1_start:i1_start+inner_shape[1],
                         i2_start:i2_start+inner_shape[2]] = inner_ground_truth
        else:
            raise ValueError
    ground_truth = (
            ground_truth /
            cfg.get('implicit_scaling_except_for_test_data', 1.)).asarray()

    random_gen = np.random.default_rng(seed)
    sinogram = dataset.ground_truth_to_obs(ground_truth, random_gen=random_gen)
    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    return sinogram, fbp, ground_truth
