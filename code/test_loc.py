import numpy as np
from os import path as osp
import os
import tqdm
import matplotlib.pyplot as plt
from utils.self_calibration import rotate_xy

def loc_nod_init():
    loc_nod = {}
    nodes = ['1', '2', '3', '4', '5', '7', '8']
    x_delta = 1.38  # 1.418
    y_delta = 1.6
    loc_nod['1'] = [0,0,0]
    loc_nod['2'] = [-1*x_delta,1*y_delta,0]
    loc_nod['3'] = [-1*x_delta,2*y_delta,0]
    loc_nod['4'] = [-1*x_delta,3*y_delta,0]
    loc_nod['5'] = [0,4*y_delta,0]
    loc_nod['6'] = [1*x_delta,3*y_delta,0]
    loc_nod['7'] = [1*x_delta,2*y_delta,0]
    loc_nod['8'] = [1*x_delta,1*y_delta,0]

    return loc_nod


# rotate loc_ske_tgt by theta
def rotate(list_loc, theta):
    x_prim = list_loc[:, 0] * np.cos(theta) - list_loc[:, 1] * np.sin(theta)
    y_prim = list_loc[:, 0] * np.sin(theta) + list_loc[:, 1] * np.cos(theta)

    return x_prim, y_prim


def do_opt_loc(dists, initi_guess):
    from utils.self_calibration import constraint1, constraint2, constraint3, constraint4, opt_loc_ranges
    con1 = {'type':'ineq', 'fun':constraint1, 'args':(dists,)}
    con2 = {'type':'ineq', 'fun':constraint2, 'args':(dists,)}
    con3 = {'type':'ineq', 'fun':constraint3, 'args':(dists,)}
    con4 = {'type':'ineq', 'fun':constraint4, 'args':(dists,)}
    cons = [con1, con2, con3, con4]
    method = 'SLSQP'

    # get optimization
    res_xy = opt_loc_ranges(dists, initi_guess, cons)
    return res_xy

def do_opt_rot(anchor_pairs):
    from utils.self_calibration import opt_rot_loc

    res_rot = opt_rot_loc(anchor_pairs=anchor_pairs)
    return res_rot


def vis_self_loc():
    import localization as lx

    pth_res = osp.join(os.getcwd(), 'data')
    nodes = ['1', '2', '3', '4', '5', '7', '8']
    # nodes_anc = ['1', '2', '4', '5', '7', '8']
    nodes_anc = ['5', '7', '8']

    lab_har = [1, 2, 3, 4, 5]
    names_har = ["sitting", "standing", "walking", "waving", "drinking"]

    pth_tuple_nod = os.path.join(pth_res, 'tuple_nod.npy')
    pth_tuple_lab = os.path.join(pth_res, 'tuple_lab.npy')
    pth_tuple_dist_diff = os.path.join(pth_res, 'dist_diff.npy')
    pth_tuple_dist_rdm = os.path.join(pth_res, 'dist_rdm.npy')
    pth_tuple_dist_ske = os.path.join(pth_res, 'dist_ske.npy')
    pth_tuple_loc_ske = os.path.join(pth_res, 'loc_ske.npy')

    tuple_nod = np.load(pth_tuple_nod)
    tuple_lab = np.load(pth_tuple_lab)
    list_dist_ske = np.load(pth_tuple_dist_ske)
    list_dist_rdm = np.load(pth_tuple_dist_rdm)
    list_dist_diff = np.load(pth_tuple_dist_diff)
    list_loc_ske = np.load(pth_tuple_loc_ske)

    loc_nod = loc_nod_init()

    loc_xy = []
    for nod in nodes:
        loc_xy.append(loc_nod[nod])
    loc_xy = np.asarray(loc_xy)[:, :2]
    loc_xy = loc_xy.flatten()

    # P = lx.Project(mode='2D', solver='LSE')

    tuple_lab_ = dict()
    list_dist_ske_ = dict()
    list_loc_ske_ = dict()
    list_dist_rdm_ = dict()

    # dist_ske_har = dict()
    # dist_rdm_har = dict()
    # dist_loc_har = dict()

    for nod in nodes:
        # P.add_anchor(nod, loc_nod[nod])

        ix_nod_sel = np.where(tuple_nod == nod)[0]
        tuple_lab_[nod] = tuple_lab[ix_nod_sel]
        list_dist_ske_[nod] = list_dist_ske[ix_nod_sel]
        list_loc_ske_[nod] = list_loc_ske[ix_nod_sel]
        list_dist_rdm_[nod] = list_dist_rdm[ix_nod_sel]
        N = len(ix_nod_sel)

    list_loc_ske_tgt_ = np.asarray(list_loc_ske_['1'])
    list_loc_ske_tgt = np.array(list_loc_ske_tgt_)
    list_loc_ske_tgt[:,1], list_loc_ske_tgt[:,2] = list_loc_ske_tgt_[:, 2], list_loc_ske_tgt_[:, 1]

    tuple_lab_tgt = tuple_lab_['1']
    ix_lab_hars = [np.where(tuple_lab_tgt == lab_)[0] for lab_ in lab_har]

    list_loc_ske_tgt = list_loc_ske_tgt[:, :2]
    list_loc_ske_tgt[:, 0] = -list_loc_ske_tgt[:, 0]

    list_loc_ske_tgt[:, 0], list_loc_ske_tgt[:, 1] = rotate(list_loc_ske_tgt, theta=-2.5 / 180 * np.pi)
    loc_tgt_har = [list_loc_ske_tgt[ix_har] for ix_har in ix_lab_hars]

    ix_sel_walking = ix_lab_hars[2]
    loc_tgt_har_wlk = loc_tgt_har[2]
    N_wlk = len(ix_sel_walking)

    for nod in nodes:
        list_dist_rdm_[nod] = list_dist_rdm_[nod][ix_sel_walking]

    def get_ranges(ix, nodes, list_dist_rdm_):
        ranges_meas = []
        for nod in nodes:
            ranges_meas.append(list_dist_rdm_[nod][ix])
        return np.asarray(ranges_meas) + 0.35

    print(N_wlk)

    N_ap = len(nodes)
    M_sp = 3
    init_guess = np.asarray([0]*((N_ap+M_sp-1)*2))
    # create constraints

    dists = []
    step_len = 3
    for ix in range(0, N_wlk-step_len, step_len):
        ranges_cur = get_ranges(ix, nodes, list_dist_rdm_)
        dists.append(ranges_cur)
        if len(dists) == M_sp:
            dists = np.asarray(dists)
            res_xy = do_opt_loc(dists, init_guess)

            loc_ap_xy = res_xy[:N_ap*2-2]
            loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])
            # VISUALIZE THE 1ST ESTIMATE OF LOCATIONS
            plt.figure()
            x = loc_ap_xy[np.arange(N_ap) * 2]
            y = loc_ap_xy[np.arange(N_ap) * 2 + 1]
            plt.scatter(x, y, marker='*')
            # x = loc_sp_xy[np.arange(5) * 2]
            # y = loc_sp_xy[np.arange(5) * 2 + 1]
            # plt.scatter(x, y, marker='*')

            # ALIGN WITH ROTATION after 1st round
            anchor_pairs = [loc_ap_xy[2:N_ap*2], loc_xy[2:N_ap*2]]

            rot = do_opt_rot(anchor_pairs)
            print('===============')
            print(rot)
            print('===============')

            loc_x_rot, loc_y_rot = rotate_xy(loc_ap_xy[2:N_ap*2], theta=rot)
            loc_x_rot, loc_y_rot = np.concatenate([[0], loc_x_rot]), np.concatenate([[0], loc_y_rot])

            plt.figure()

            plt.scatter(loc_x_rot, loc_y_rot, marker='*')


            plt.show()

            init_guess[:N_ap*2-2] = np.asarray(res_xy[:N_ap*2-2])
            dists = []


    # # ============localization with anchor nodes=================
    # loc_ske_pred = []
    # loc_rdm_pred = []
    # for ix in tqdm(range(N)):
    #     P = None
    #     P = lx.Project(mode='2D', solver='LSE')
    #     for nod in nodes_anc:
    #         P.add_anchor(nod, loc_nod[nod])
    #
    #     t_ske = None
    #     t_rdm = None
    #
    #     t_ske, _ = P.add_target()
    #     t_rdm, _ = P.add_target()
    #
    #     for nod in nodes_anc:
    #         dist_ske = list_dist_ske_[nod][ix]
    #         dist_rdm = list_dist_rdm_[nod][ix]+0.35  # 35 cm offset
    #         t_ske.add_measure(nod, dist_ske)
    #         t_rdm.add_measure(nod, dist_rdm)
    #
    #     P.solve()
    #
    #     loc_ske_ = np.asarray([t_ske.loc.x, t_ske.loc.y, t_ske.loc.z])
    #     loc_rdm_ = np.asarray([t_rdm.loc.x, t_rdm.loc.y, t_rdm.loc.z])
    #     # loc_tgt = list_loc_ske_tgt[ix]
    #
    #     loc_ske_pred.append(loc_ske_)
    #     loc_rdm_pred.append(loc_rdm_)
    #
    # loc_ske_pred = np.asarray(loc_ske_pred)
    # loc_rdm_pred = np.asarray(loc_rdm_pred)
    #
    # # ix_lab_hars = [np.where(tuple_lab_tgt == lab_)[0] for lab_ in lab_har]
    #
    # loc_ske_pred = loc_ske_pred[:, :2]
    # loc_rdm_pred = loc_rdm_pred[:, :2]
    #
    # # list_loc_ske_tgt[:, 0] = list_loc_ske_tgt[:, 0] - 0.1
    # # list_loc_ske_tgt[:, 1] = list_loc_ske_tgt[:, 1] - 0.1
    #
    # #
    # loc_ske_har = [loc_ske_pred[ix_har] for ix_har in ix_lab_hars]
    # loc_rdm_har = [loc_rdm_pred[ix_har] for ix_har in ix_lab_hars]
        # ============localization with anchor nodes=================

    # x_cal_nod, y_cal_nod = calibrate_trace(list_dist_rdm_, ix_lab_hars, loc_ske_har=loc_ske_har, loc_rdm_har=loc_rdm_har, loc_tgt_har=loc_tgt_har, pth_fig=pth_res, subdir=''.join(nodes_anc))


if __name__ == '__main__':
    vis_self_loc()