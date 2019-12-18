import open3d as o3d
import numpy as np
import copy
import time

from os.path import join

DATA_DIR = '/home/kristijan/phd/datasets/Stanford3DDataset/'
OBJECTS = ['bunny', 'blade', 'dragon', 'hand', 'happy', 'horse']

VOXEL_SIZE = 0.05   # means 5cm for the dataset


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, data_root, object_name):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(join(data_root, object_name + '.ply'))
    target = source.translate(np.array([0.1, 0.1, 0.1])).rotate(np.array([0.5, 0.5, 0.5]))
    o3d.io.write_point_cloud(join(data_root, object_name + '_copy.ply'), target)
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


def coarse_fine_matching(name):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, DATA_DIR, name)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                VOXEL_SIZE)
    print('Result RANSAC: {}'.format(result_ransac))
    print(result_ransac.transformation)
    draw_registration_result(source_down, target_down,
                            result_ransac.transformation)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                    VOXEL_SIZE, result_ransac)
    print('Result ICP: {}'.format(result_icp))
    print(result_icp.transformation)
    draw_registration_result(source, target, result_icp.transformation)


def fast_global_matching(name):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, DATA_DIR, name)

    start = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                VOXEL_SIZE)
    print(result_ransac)
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    draw_registration_result(source_down, target_down,
                             result_ransac.transformation)

    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   VOXEL_SIZE)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    draw_registration_result(source, target,
                             result_fast.transformation)


if __name__ == "__main__":
    for name in OBJECTS[:1]:
        #coarse_fine_matching(name)
        fast_global_matching(name)
