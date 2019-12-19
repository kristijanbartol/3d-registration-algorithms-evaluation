from os.path import join

DATA_DIR = '/home/kristijan/phd/datasets/Stanford3DDataset/'
ITEMS = ['bunny', 'blade', 'dragon', 'hand', 'happy', 'horse']


if __name__ == '__main__':
    for item in ITEMS[:2]:
        ply_base = join(DATA_DIR, item)
        txt_base = join(DATA_DIR, item)
        for suffix in ['.', '_copy.']:
            flag = False
            with open(ply_base + suffix + 'ply') as rf:
                with open(txt_base + suffix + 'txt', 'w') as wf:
                    lines = rf.readlines()
                    for line in lines:
                        if 'end_header' in line:
                            flag = True
                            continue
                        if 'element vertex' in line:
                            npoints = line.rstrip().split(' ')[2]
                            wf.write('{}\n'.format(npoints))
                        if not flag:
                            continue
                        if line[0] == '3':
                            continue
                        coords = line.rstrip().split(' ')[:3]
                        wf.write(' '.join(coords) + '\n')
