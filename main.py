import open3d as o3d

from os.path import join

DATA_DIR = '/home/kristijan/phd/datasets/Stanford3DDataset/'
OBJECTS = ['bunny', 'blade', 'dragon', 'hand', 'happy', 'horse']


if __name__ == '__main__':
    for name in OBJECTS:
        print('Reading {} point cloud...'.format(name))
        path = join(DATA_DIR, name + '.ply')
        pcd = o3d.io.read_point_cloud(path)
        print(pcd)
        new_path = join(DATA_DIR, name + '_copy.ply')
        o3d.io.write_point_cloud(new_path, pcd)
