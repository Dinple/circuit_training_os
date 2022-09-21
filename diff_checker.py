import numpy as np
import os, sys
np.set_printoptions(threshold=sys.maxsize)
HASH = 19350501701115144
PREFIX = ''.join(['*'] * 10)

np.set_printoptions(threshold=np.inf)


# LOCATIOIN X/Y under STATE OBSERVATION
if False:
    print(PREFIX + "LOCATION X" + PREFIX)
    with open('./failed_obs/reset_{}_feature_locations_x@locations_x.npy'.format(str(HASH)), 'rb') as f:
        gl_feature = np.load(f)
        os_feature = np.load(f)

    diff_indices = np.where(gl_feature != os_feature)[0]

    for ind in range(diff_indices.shape[0]):
        # print("At {}, GL:{}, OS:{}".format(str(ind), str(gl_feature[ind]), str(os_feature[ind])))
        print("At {}, GL:{}".format(str(ind), str(gl_feature[ind])))

    print(PREFIX + "LOCATION Y" + PREFIX)
    with open('./failed_obs/reset_{}_feature_locations_y@locations_y.npy'.format(str(HASH)), 'rb') as f:
        gl_feature = np.load(f)
        os_feature = np.load(f)

    diff_indices = np.where(gl_feature != os_feature)[0]

    for ind in range(diff_indices.shape[0]):
        # print("At {}, GL:{}, OS:{}".format(str(ind), str(gl_feature[ind]), str(os_feature[ind])))
        print("At {}, GL:{}".format(str(ind), str(gl_feature[ind])))

if False:
    with open('./init_mask/run-1_node_0.npy', 'rb') as f:
        gl_init_mask = np.load(f)
        os_init_mask = np.load(f)

    diff_indices = np.where(gl_init_mask != os_init_mask)[0]
    np.set_printoptions(linewidth=260)
    print(gl_init_mask.reshape((128, 128)))
    print(os_init_mask.reshape((128, 128)))

    # for ind in range(diff_indices.shape[0]):
    #     print("At {}, GL:{}, OS:{}".format(str(ind), str(gl_init_mask[ind]), str(os_init_mask[ind])))

if True:
    name = "1.3213340368032_vs_1.1045866148091457.npy"
    with open('./failed_proxy_coord/' + name, 'rb') as f:
        gl_proxy_coord = np.load(f, allow_pickle=True)
        os_proxy_coord = np.load(f, allow_pickle=True)
    
    for gl, os in zip(gl_proxy_coord, os_proxy_coord):
        if abs(gl[1][0] - os[1][0]) > 1e-3 or (gl[1][1] - os[1][1]) > 1e-3:
            print(gl, os)

    # print((gl_proxy_coord==os_proxy_coord).all())
