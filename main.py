import open3d as o3d
import numpy as np
import copy
import time
import json

from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
from os.path import join

DATA_DIR = '/home/kristijan/phd/datasets/Stanford3DDataset/'
ITEMS = ['bunny', 'blade', 'dragon', 'hand', 'happy', 'horse']

VOXEL_SIZE = 0.05   # means 5cm for the dataset


def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def save_results(source, target, result, data_root, object_name, algo):
    pc_path = join(data_root, '{}_{}.ply'.format(object_name, algo))
    print('Saving resulting point cloud...')
    o3d.io.write_point_cloud(pc_path, source, write_ascii=True)

    metrics_path = join(data_root, '{}.json'.format(algo))
    metrics = { 
                'rmse': result.inlier_rmse, 
                'fitness': result.fitness
              }
    print('Saving metrics...')
    with open(metrics_path, 'w') as mp:
        json.dump(metrics, mp)


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
    source_path = join(data_root, '{}.{}'.format(object_name, 'ply'))
    target_path = join(data_root, object_name + '{}_copy.{}'.format(object_name, 'ply'))
    source = o3d.io.read_point_cloud(source_path)
    #target = source.translate(np.array([0.1, 0.1, 0.1])).rotate(np.array([0.5, 0.5, 0.5]))
    #o3d.io.write_point_cloud(join(data_root, object_name + '_copy.ply'), target, write_ascii=True)
    target = o3d.io.read_point_cloud(target_path)
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
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
    draw_registration_result(source, target,
                            result_ransac.transformation)
    save_results(source, target, result_ransac, DATA_DIR, name, 'fpfh-ransac')

    
    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                    VOXEL_SIZE, result_ransac)
    print('Result ICP: {}'.format(result_icp))
    print(result_icp.transformation)
    draw_registration_result(source, target, result_icp.transformation)
    
    save_results(source, target, result_icp, DATA_DIR, name, 'fpfh-ransac-icp')


def fast_global_matching(name):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, DATA_DIR, name)

    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   VOXEL_SIZE)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print('Result FGR: {}'.format(result_fast))
    print(result_fast.transformation)
    draw_registration_result(source, target,
                             result_fast.transformation)
    save_results(source, target, result_fast, DATA_DIR, name, 'fgr')

    # NOTE: The correspondence distance tolerance is very high!
    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                    VOXEL_SIZE * 10, result_fast)
    print('Result ICP: {}'.format(result_icp))
    print(result_icp.transformation)
    draw_registration_result(source, target, result_icp.transformation)
    save_results(source, target, result_icp, DATA_DIR, name, 'fgr-icp')


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
    tgt_path = join(DATA_DIR, '{}_copy.{}'.format(name, 'txt'))
    N_src, src_points = loadPointCloud(src_path)
    N_tgt, tgt_points = loadPointCloud(tgt_path)
    goicp.loadModelAndData(N_src, src_points, N_tgt, tgt_points)
    goicp.setDTSizeAndFactor(300, 2.0)
    goicp.BuildDT()
    goicp.Register()

    source = goicp_to_o3d(src_points)
    target = goicp_to_o3d(tgt_points)
    draw_registration_result(source, target)

    goicp_transformation = get_transformation(
        goicp.optimalRotation(), 
        goicp.optimalTranslation())
    draw_registration_result(target, source, goicp_transformation)

    print(goicp.optimalRotation())
    print(goicp.optimalTranslation())


if __name__ == "__main__":
    for name in ITEMS[:1]:
        #coarse_fine_matching(name)
        #fast_global_matching(name)
        go_icp(name)
