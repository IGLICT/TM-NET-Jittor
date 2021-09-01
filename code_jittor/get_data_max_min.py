import os
import glob
import scipy.io as sio
import numpy as np

def test():
    f = open('./log.txt', 'r')
    lines = f.readlines()
    name_val_dict = {}
    for line in lines:
        splits = line.split(' ')
        cat = splits[0]
        part_name = splits[1]
        if cat not in name_val_dict:
            name_val_dict[cat] = {}
        name_val_dict[cat][part_name] = [float(num) for num in splits[2:]]
    # print(name_val_dict)

def get_part_names(category):
    if category == 'chair':
        part_names = ['back', 'seat', 'leg_ver_1', 'leg_ver_2', 'leg_ver_3', 'leg_ver_4', 'hand_1', 'hand_2']
    elif category == 'knife':
        part_names = ['part1', 'part2']
    elif category == 'guitar':
        part_names = ['part1', 'part2', 'part3']
    elif category == 'cup':
        part_names = ['part1', 'part2']
    elif category == 'car':
        part_names = ['body', 'left_front_wheel', 'right_front_wheel', 'left_back_wheel', 'right_back_wheel','left_mirror','right_mirror']
    elif category == 'table':
        # part_names = ['surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2']
        part_names = ['surface', 'left_leg1', 'left_leg2', 'left_leg3', 'left_leg4', 'right_leg1', 'right_leg2', 'right_leg3', 'right_leg4']
    elif category == 'plane':
        part_names = ['body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'up_tail', 'down_tail', 'front_gear', 'left_gear', 'right_gear', 'left_engine1', 'right_engine1', 'left_engine2', 'right_engine2']
    else:
        raise Exception("Error")
    return part_names

if __name__ == '__main__':
    sub_dirs = ['car', 'car_new_reg', 'chair', 'plane', 'table']
    cats = ['car', 'car', 'chair', 'plane', 'table']

    path = '/mnt/f/wutong/data/'
    for sub_dir, cat in zip(sub_dirs, cats):
        sub_path = os.path.join(path, sub_dir)
        part_names = get_part_names(cat)
        for part_name in part_names:
            mat_files = glob.glob(os.path.join(sub_path, '*', '*{}.mat'.format(part_name)))
        
            LOGR_max = -np.inf
            LOGR_min = np.inf
            S_max = -np.inf
            S_min = np.inf

            for mat_file in mat_files:
                # print(mat_file)
                geo_data = sio.loadmat(mat_file, verify_compressed_data_integrity=False)
                try:
                    LOGR = geo_data['fmlogdr']
                    S = geo_data['fms']
                    temp_LOGR_min = LOGR.min()
                    temp_LOGR_max = LOGR.max()
                    if temp_LOGR_max > LOGR_max:
                        LOGR_max = temp_LOGR_max
                    if temp_LOGR_min < LOGR_min:
                        LOGR_min = temp_LOGR_min

                    temp_S_min = S.min()
                    temp_S_max = S.max()
                    if temp_S_max > S_max:
                        S_max = temp_S_max
                    if temp_S_min < S_min:
                        S_min = temp_S_min
                except:
                    print(mat_file)
            print('{} {} {} {} {} {}'.format(cat, part_name, LOGR_max, LOGR_min, S_max, S_min))
    category_val_dict = {}
    for k1, v1 in cate_part_val_dict.items():
        category_val_dict[k1] = [-np.inf, np.inf, -np.inf, np.inf]
        for k2, v2 in cate_part_val_dict[k1].items():
            if v2[0] > category_val_dict[k1][0]:
                category_val_dict[k1][0] = v2[0]
            if v2[1] < category_val_dict[k1][1]:
                category_val_dict[k1][1] = v2[1]
            if v2[2] > category_val_dict[k1][2]:
                category_val_dict[k1][2] = v2[2]
            if v2[3] < category_val_dict[k1][3]:
                category_val_dict[k1][3] = v2[3]
    print(category_val_dict)
