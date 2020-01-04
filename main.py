import open3d as o3d
import numpy as np
import copy
import time
import json

from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
from os.path import join

DATA_DIR = '/home/kristijan/phd/datasets/Stanford3DDataset/'
ITEMS = ['bunny', 'horse', 'hand', 'dragon', 'happy']#, 'blade']

VOXEL_SIZE = 0.05   # means 5cm for the dataset

result_dict = {}


def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def save_results(source, target, result, data_root, object_name, algo):
    metrics_path = join(data_root, '{}_{}.json'.format(object_name, algo))
    metrics = { 
                'rmse': result.inlier_rmse, 
                'fitness': result.fitness
              }
    with open(metrics_path, 'w') as mp:
        json.dump(metrics, mp)
    result_dict['{}_{}'.format(object_name, algo)] = metrics


def preprocess_point_cloud(pcd, voxel_size, object_name):
    start_time = time.time()
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    start_time = time.time()
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    start_time = time.time()
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print('[{}] Compute FPFH feature with search radius {:.3f} ({:.2f}s).'.format(
        object_name, radius_feature, time.time() - start_time))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, data_root, object_name):
    source_path = join(data_root, '{}.{}'.format(object_name, 'ply'))
    target_path = join(data_root, '{}_trans.{}'.format(object_name, 'ply'))
    source = o3d.io.read_point_cloud(source_path)
    #target = source.translate(np.array([0.1, 0.1, 0.1])).rotate(np.array([0.5, 0.5, 0.5]))
    #o3d.io.write_point_cloud(target_path, target, write_ascii=True)
    target = o3d.io.read_point_cloud(target_path)
    #trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, object_name)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, object_name)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
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
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac, icp_type):
    distance_threshold = voxel_size * 0.4
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPoint() if icp_type == 'point' \
            else o3d.registration.TransformationEstimationPointToPlane())
    return result


def coarse_fine_matching(name, icp_type):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, DATA_DIR, name)

    start_time = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                VOXEL_SIZE)
    print('[{}] Result RANSAC: {} ({:.2f}s)'.format(name, result_ransac, time.time() - start_time))
    draw_registration_result(source, target,
                            result_ransac.transformation)
    save_results(source, target, result_ransac, DATA_DIR, name, 'fpfh-ransac')

    start_time = time.time()
    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                    VOXEL_SIZE, result_ransac, icp_type)
    print('[{}] Result ICP: {} ({:.2f}s)'.format(name, result_icp, time.time() - start_time))
    draw_registration_result(source, target, result_icp.transformation)
    
    save_results(source, target, result_icp, DATA_DIR, name, 'fpfh-ransac-{}-icp'.format(icp_type))


def fast_global_matching(name, icp_type):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, DATA_DIR, name)

    start_time = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   VOXEL_SIZE)
    result_fast = o3d.registration.evaluate_registration(
        source, target, 0.01, transformation=result_fast.transformation)                                     
    print('[{}] Result FGR: {} ({:.2f}s)'.format(name, result_fast, time.time() - start_time))
    draw_registration_result(source, target,
                             result_fast.transformation)
    save_results(source, target, result_fast, DATA_DIR, name, 'fgr')

    start_time = time.time()
    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                    VOXEL_SIZE, result_fast, icp_type)
    print('[{}] Result ICP: {} ({:.2f}s)'.format(name, result_icp, time.time() - start_time))
    draw_registration_result(source, target, result_icp.transformation)
    save_results(source, target, result_icp, DATA_DIR, name, 'fgr-{}-icp'.format(icp_type))


def loadPointCloud(filename):
    pcloud = np.loadtxt(filename, skiprows=1)
    plist = pcloud.tolist()
    p3dlist = []
    for x, y, z in plist:
        pt = POINT3D(x, y, z)
        p3dlist.append(pt)
    return pcloud.shape[0], p3dlist


def goicp_to_o3d(points):
    pc_list = []
    for point in points:
        pc_list.append([point.x, point.y, point.z])
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(np.array(pc_list))
    return o3d_pc


def get_transformation(rotation, translation):
    return np.array([[rotation[0][0], rotation[0][1], rotation[0][2],
        translation[0]], [rotation[1][0], rotation[1][1], rotation[1][2],
        translation[1]], [rotation[2][0], rotation[2][1], rotation[2][2],
        translation[2]], [0., 0., 0., 1.]])


def go_icp(name):
    goicp = GoICP()
    src_path = join(DATA_DIR, '{}.{}'.format(name, 'txt'))
    tgt_path = join(DATA_DIR, '{}_trans.{}'.format(name, 'txt'))
    N_src, src_points = loadPointCloud(src_path)
    N_tgt, tgt_points = loadPointCloud(tgt_path)
    goicp.loadModelAndData(N_src, src_points, N_tgt, tgt_points)
    goicp.setDTSizeAndFactor(300, 2.0)

    start_time = time.time()
    goicp.BuildDT()
    goicp.Register()
    goicp_time = time.time() - start_time

    source = goicp_to_o3d(src_points)
    target = goicp_to_o3d(tgt_points)
    draw_registration_result(source, target)

    goicp_transformation = get_transformation(
        goicp.optimalRotation(), 
        goicp.optimalTranslation())
    draw_registration_result(target, source, goicp_transformation)

    result_goicp = o3d.registration.evaluate_registration(
        target, source, 0.001, transformation=goicp_transformation)
    print(goicp_transformation) 
    print('[{}] Result Go-ICP: {} ({:.2f})'.format(name, result_goicp, goicp_time))

    save_results(source, target, result_goicp, DATA_DIR, name, 'goicp')


if __name__ == "__main__":
    for name in ITEMS:
        coarse_fine_matching(name, 'point')
        coarse_fine_matching(name, 'plane')
        fast_global_matching(name, 'point')
        fast_global_matching(name, 'plane')
        go_icp(name)

    with open(join(DATA_DIR, 'results.json'), 'w') as fr:
        json.dump(result_dict, fr)
