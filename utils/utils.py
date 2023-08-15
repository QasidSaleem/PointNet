"""
@Description: Module of utilities
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
import os
import math
import random

import numpy as np
import point_cloud_utils as pcu
import torch
from pytorch3d.structures import Pointclouds

random.seed = 42


def normalize_mesh(vertices):

    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)
    vertices[:, 0] = (vertices[:, 0] - x_mid) / scale
    vertices[:, 1] = (vertices[:, 1] - y_mid) / scale
    vertices[:, 2] = (vertices[:, 2] - z_mid) / scale

    return vertices

def sample_pcd(mesh, target_num_pts):
    """"""
    verts, faces = mesh
    verts = np.array(verts)
    verts = normalize_mesh(verts)
    faces = np.array(faces)

    normals = pcu.estimate_mesh_vertex_normals(verts, faces)
    fid, bc = pcu.sample_mesh_poisson_disk(
        verts, faces, num_samples=target_num_pts
    )
    rand_positions = pcu.interpolate_barycentric_coords(faces, fid, bc, verts)
    number_of_points = rand_positions.shape[0]
    if number_of_points == target_num_pts:
        return rand_positions
    elif number_of_points < target_num_pts:
        offset = target_num_pts - number_of_points
        idx = np.arange(number_of_points)
        idx = np.concatenate([idx, np.random.randint(number_of_points, size=offset)])
        return rand_positions[idx]
    else:
        rand_normals = pcu.interpolate_barycentric_coords(faces, fid, bc, normals)
        pcd = Pointclouds(
            points=[torch.from_numpy(rand_positions)],
            normals=[torch.from_numpy(rand_normals)],
        )
        pcd = pcd.subsample(target_num_pts)
        return np.array(pcd.points_list()[0])



def get_classes(configs: dict) -> dict:

    """ Gets the list of class from folders

        Params
        --------
            configs (dict): Configurations

        Returns
        --------
            classes (list): List of classes

    """

    folders = [
        _dir for _dir in sorted(os.listdir(configs['data']['data_path']))
        if os.path.isdir(os.path.join(configs['data']['data_path'], _dir))
    ]
    classes = {folder: i for i, folder in enumerate(folders)}

    return classes

class PointSampler(object):
    def __init__(self, output_size: int) -> None:

        """ Class initializer

            Params
            --------
                output_size (int): Output size of the network

        """

        assert isinstance(output_size, int)
        self.output_size = output_size
        return

    def triangle_area(
            self,
            pt1: np.ndarray,
            pt2: np.ndarray,
            pt3: np.ndarray
    ) -> np.float64:

        """ Calculates the area of a triangle

            Params
            --------
                pt1 (numpy.ndarray): Coordinate values for the 1st point
                pt2 (numpy.ndarray): Coordinate values for the 2nd point
                pt3 (numpy.ndarray): Coordinate values for the 3rd point

            Returns
            --------
                ret (np.float64): Area of the triangle

        """

        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        ret = max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5
        return ret

    def sample_point(
            self,
            pt1: np.ndarray,
            pt2: np.ndarray,
            pt3: np.ndarray
    ) -> tuple:

        """ Samples points with barycentric coordinates on a triangle

            Reference: https://mathworld.wolfram.com/BarycentricCoordinates.html

            Params
            --------
                pt1 (numpy.ndarray): Coordinate values for the 1st point
                pt2 (numpy.ndarray): Coordinate values for the 2nd point
                pt3 (numpy.ndarray): Coordinate values for the 3rd point

            Returns
            --------
                ret (np.float64): Area of the triangle

        """

        s, t = sorted([random.random(), random.random()])
        coords = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return coords(0), coords(1), coords(2)

    def __call__(self, mesh: tuple) -> np.ndarray:

        """ Class caller

            Params
            --------
                mesh (tuple): Mesh data

            Returns
            --------
                sampled_points (numpy.ndarray): Sample point data

        """

        # verts, faces = mesh
        # verts = np.array(verts)
        # areas = np.zeros((len(faces)))

        # for i in range(len(areas)):
        #     areas[i] = (
        #         self.triangle_area(
        #             verts[faces[i][0]],
        #             verts[faces[i][1]],
        #             verts[faces[i][2]]
        #         )
        #     )

        # sampled_faces = (
        #     random.choices(
        #         faces,
        #         weights=areas,
        #         cum_weights=None,
        #         k=self.output_size
        #     )
        # )

        # sampled_points = np.zeros((self.output_size, 3))

        # for i in range(len(sampled_faces)):
        #     sampled_points[i] = (
        #         self.sample_point(
        #             verts[sampled_faces[i][0]],
        #             verts[sampled_faces[i][1]],
        #             verts[sampled_faces[i][2]]
        #         )
        #     )

        # return sampled_points
        return sample_pcd(mesh, self.output_size)



class Normalize(object):
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:

        """ Class caller

            Params
            --------
                pointcloud (numpy.ndarray): Point cloud data

            Returns
            --------
                norm_pointcloud (numpy.ndarray): Normalized point cloud data

        """

        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:

        """ Class caller

            Params
            --------
                pointcloud (numpy.ndarray): Point cloud data

            Returns
            --------
                rot_pointcloud (numpy.ndarray): Rotated point cloud data

        """

        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1]
            ]
        )

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:

        """ Class caller

            Params
            --------
                pointcloud (numpy.ndarray): Point cloud data

            Returns
            --------
                noisy_pointcloud (numpy.ndarray): Point cloud data with random noise

        """

        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
