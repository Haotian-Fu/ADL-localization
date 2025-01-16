from scipy.optimize import minimize
import scipy.optimize as optimize
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import localization as lx
import time


def draw_bars(data):
    fractions = list(data.values())
    srcs = list(data.keys())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(srcs, fractions,
            width=0.4)

def draw_bar_s(directory, devis, labels, title='Dist Diff'):
    # fig = plt.figure(figsize=(5, 3.3))

    fig = plt.figure(figsize=(10, 7.2))
    plt.rcParams.update({'font.size': 22})

    print(title)
    print(title)
    print(title)

    plt.bar(labels, devis,
            width=0.4)

    # plt.xticks(np.arange(0, 1, 0.05))
    plt.ylabel("Frequency (0--1)")
    plt.xlabel("# rounds")
    plt.title(title)
    # plt.legend()

    plt.grid()

    plt.tight_layout()

    if not os.path.exists(os.path.join(directory, 'fig')):
        os.mkdir(os.path.join(directory, 'fig'))
    fig.savefig(os.path.join(directory, 'fig', title + '_bars_.png'), bbox_inches='tight')

    plt.close()


def draw_boxplot_s(directory, devis, labels, ylabel="Loc. Error (m)", title='Dist Diff'):
    # fig = plt.figure(figsize=(5, 3.3))

    fig = plt.figure(figsize=(10, 7.2))
    plt.rcParams.update({'font.size': 22})

    print(title)
    print(title)
    print(title)

    plt.boxplot(devis, labels=labels)

    # plt.xticks(np.arange(0, 1, 0.05))
    plt.ylabel(ylabel)
    plt.xlabel("# rounds")
    plt.title(title)
    # plt.legend()

    plt.grid()

    plt.tight_layout()

    if not os.path.exists(os.path.join(directory, 'fig')):
        os.mkdir(os.path.join(directory, 'fig'))
    fig.savefig(os.path.join(directory, 'fig', title + '_boxplot_.png'), bbox_inches='tight')

    plt.close()


def draw_cdf_e_s(directory, devis, labels, title='Dist Diff'):
    # fig = plt.figure(figsize=(5, 3.3))
    fig = plt.figure(figsize=(7.5, 5))

    if not os.path.exists(os.path.join(directory, 'fig')):
        os.mkdir(os.path.join(directory, 'fig'))

    pth_fig = os.path.join(directory, 'fig')

    import matplotlib

    # fig = plt.figure(figsize=(10, 7.2))
    plt.rcParams.update({'font.size': 18})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    print(title)
    print(title)
    print(title)
    ix = 0
    linstyle = [':', '-', '-.', '--', '-']

    pth_dat = os.path.join(pth_fig, title)
    if not os.path.exists(pth_dat):
        os.mkdir(pth_dat)

    for dev, lab in zip(devis, labels):
        if len(dev) == 0:
            continue
        print(lab)

        # f_lab = os.path.join(pth_dat, lab)
        # np.save(f_lab, dev)

        #
        cdf_, bin_edges = cdf(dev)
        #
        """get percentile info"""
        ix_80p = np.argmin(abs(cdf_ - .80))
        val_80p = (bin_edges[ix_80p])
        print('80:', val_80p)
        #
        # ix_50p = np.argmin(abs(cdf_ - .50))
        # val_50p = (bin_edges[ix_50p])
        # print('50:', val_50p)

        ix_3bpm = np.argmin(abs(bin_edges - 4.))
        perct_3bpm = (cdf_[ix_3bpm - 1])
        print('3:', perct_3bpm)

        ix_5bpm = np.argmin(abs(bin_edges - 5))
        # if ix_20bpm<len(cdf_)
        perct_5bpm = (cdf_[ix_5bpm - 1])
        print('5:', perct_5bpm)

        ix_20bpm = np.argmin(abs(bin_edges - 20))
        # if ix_20bpm<len(cdf_)
        perct_20bpm = (cdf_[ix_20bpm - 1])
        print('20:', perct_20bpm)

        # Plot the cdf
        plt.plot(bin_edges[0:-1], cdf_, linestyle=linstyle[ix%5], label=lab, linewidth=2)
        ix += 1



    # plt.plot(bin_edges_rr[0:-1], cdf_rr, linestyle=':', label=label1)
    plt.xlim((0, 10))
    plt.ylim((0, 1))
    y = np.arange(0, 1.10, 0.2)
    # y_ = np.asarray(np.arange(0, 119, 20), dtype=int)
    plt.yticks(y)
    # plt.xticks(np.arange(0, 1, 0.05))
    plt.ylabel("CDF")
    plt.xlabel("Error (m)")
    # plt.title(title)
    plt.legend()

    plt.grid()

    plt.tight_layout()

    fig.savefig(os.path.join(directory, 'fig', title + '_cdf.pdf'), bbox_inches='tight')

    plt.close()


def cdf(data):
    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    return cdf, bin_edges



def get_dist(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist


def rotate(list_loc, theta):
    x_prim = list_loc[:, 0] * np.cos(theta) - list_loc[:, 1] * np.sin(theta)
    y_prim = list_loc[:, 0] * np.sin(theta) + list_loc[:, 1] * np.cos(theta)

    return x_prim, y_prim


def rotate_xy(loc_xy, theta):
    N_loc = len(loc_xy)
    loc_x_cur, loc_y_cur = loc_xy[np.arange(N_loc // 2) * 2], loc_xy[np.arange(N_loc // 2) * 2 + 1]
    loc_x_cur_ = loc_x_cur * np.cos(theta) - loc_y_cur * np.sin(theta)
    loc_y_cur_ = loc_x_cur * np.sin(theta) + loc_y_cur * np.cos(theta)

    return loc_x_cur_, loc_y_cur_


def obj_rotate(theta, args):
    """
    :param theta: the angle to rotate for matching
    :param args: args[0] loc_xy_cur, args[1] loc_xy_pre
    :return:
    """
    loc_xy_cur = args[0]
    loc_xy_pre = args[1]
    N_loc = len(loc_xy_cur)
    loc_x_cur, loc_y_cur = loc_xy_cur[np.arange(N_loc//2)*2], loc_xy_cur[np.arange(N_loc//2)*2+1]
    loc_x_pre, loc_y_pre = loc_xy_pre[np.arange(N_loc//2)*2], loc_xy_pre[np.arange(N_loc//2)*2+1]
    loc_x_cur_ = loc_x_cur * np.cos(theta) - loc_y_cur * np.sin(theta)
    loc_y_cur_ = loc_x_cur * np.sin(theta) + loc_y_cur * np.cos(theta)

    mse = np.sum((loc_x_cur_ - loc_x_pre)**2 + (loc_y_cur_ - loc_y_pre)**2)
    print('mse:{}'.format(mse))

    return mse


def objective(loc_xy, rs):
    # M is the number of steps and N is the number of anchors
    M, N = np.shape(rs)
    loc_ap_xy = loc_xy[:N*2-2]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])
    loc_sp_xy = loc_xy[N*2-2:]
    # M_len = len(rs)
    # err_sum = 0
    dists_pre = [[get_dist(loc_ap_xy[2*ix_loc_ap:2*ix_loc_ap+2], loc_sp_xy[2*ix_loc_sp:2*ix_loc_sp+2]) for ix_loc_ap in range(len(loc_ap_xy)//2)] for ix_loc_sp in range(len(loc_sp_xy)//2)]

    dists_pre = np.asarray(dists_pre)
    err_sum = np.sum((dists_pre-rs)**2)
    # print('err squared:{}'.format(err_sum))
    return err_sum


def constraint1(loc_xy, rs):
    # max x of steps is smaller than max x of anchors
    # M is the number of steps and N is the number of anchors
    M, N = np.shape(rs)
    loc_ap_xy = loc_xy[:N*2-2]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = loc_xy[N*2-2:]
    loc_ap_x, loc_ap_y = loc_ap_xy[np.arange(N)*2], loc_ap_xy[np.arange(N)*2+1]
    loc_sp_x, loc_sp_y = loc_sp_xy[np.arange(M)*2], loc_sp_xy[np.arange(M)*2+1]
    return np.max(loc_ap_x) - np.max(loc_sp_x)


def constraint2(loc_xy, rs):
    # max y of steps is smaller than max y of anchors
    # M is the number of steps and N is the number of anchors
    M, N = np.shape(rs)

    loc_ap_xy = loc_xy[:N*2-2]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = loc_xy[N*2-2:]
    loc_ap_x, loc_ap_y = loc_ap_xy[np.arange(N)*2], loc_ap_xy[np.arange(N)*2+1]
    loc_sp_x, loc_sp_y = loc_sp_xy[np.arange(M)*2], loc_sp_xy[np.arange(M)*2+1]
    return np.max(loc_ap_y) - np.max(loc_sp_y)


def constraint3(loc_xy, rs):
    # min x of steps is larger than min x of anchors
    # M is the number of steps and N is the number of anchors
    M, N = np.shape(rs)

    loc_ap_xy = loc_xy[:N*2-2]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = loc_xy[N*2-2:]
    loc_ap_x, loc_ap_y = loc_ap_xy[np.arange(N)*2], loc_ap_xy[np.arange(N)*2+1]
    loc_sp_x, loc_sp_y = loc_sp_xy[np.arange(M)*2], loc_sp_xy[np.arange(M)*2+1]
    return -np.min(loc_ap_x) + np.min(loc_sp_x)


def constraint4(loc_xy, rs):
    # min y of steps is larger than min y of anchors
    # M is the number of steps and N is the number of anchors
    M, N = np.shape(rs)

    loc_ap_xy = loc_xy[:N*2-2]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = loc_xy[N*2-2:]
    loc_ap_x, loc_ap_y = loc_ap_xy[np.arange(N)*2], loc_ap_xy[np.arange(N)*2+1]
    loc_sp_x, loc_sp_y = loc_sp_xy[np.arange(M)*2], loc_sp_xy[np.arange(M)*2+1]
    return -np.min(loc_ap_y) + np.min(loc_sp_y)


def constraint5_(loc_xy, rs):
    # MDS constraints

    # min y of steps is larger than min y of anchors
    # M is the number of steps and N is the number of anchors
    M, N = np.shape(rs)

    for i in range(N):
        ix_s = np.argmin(rs[:, i])
        rs_i = rs[ix_s]

    loc_ap_xy = loc_xy[:N*2-2]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = loc_xy[N*2-2:]
    loc_ap_x, loc_ap_y = loc_ap_xy[np.arange(N)*2], loc_ap_xy[np.arange(N)*2+1]
    loc_sp_x, loc_sp_y = loc_sp_xy[np.arange(M)*2], loc_sp_xy[np.arange(M)*2+1]
    return - np.min(loc_ap_y) + np.min(loc_sp_y)


def get_DistMat(anchors, steps, sigma=1):
    dists = [[get_dist(pa, ps) + sigma * np.random.randn() for pa in anchors] for ps in steps]
    # print(dists)

    return np.asarray(dists)


def opt_loc_ranges(dists, initi_guess, cons, method = 'SLSQP'):
    # rs = dists

    # initi_guess = np.concatenate([loc_xy[20:],[0]*20]) //2
    # initi_guess = np.random.random(20)
    res = minimize(objective, initi_guess, args=dists, method=method, constraints=cons)
    # print(res)

    # GET THE 1ST ROUND ESTIMATE OF OPTIMAL LOCATION
    res_xy = res.x
    res_err = res.fun
    return res_xy, res_err


def opt_rot_loc(anchor_pairs, method='SLSQP'):

    res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))
    rot = res_ROT.x
    res_err = res_ROT.fun
    return rot, res_err


class SelfCalibration:

    def __init__(self, M=5):
        # M is the number of participatory steps for collaborative localization
        self.theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        self.Rs = 5
        self.rs = 2

        # M is the number of steps; N is the number of anchors
        self.M = M
        self.N = 5

        # given N anchor nodes, M step locations;
        # N >= 3; M >=2;
        # N ()

        # x_delta = 1.38  # 1.418
        # y_delta = 1.6
        # loc_nod['1'] = [0, 0, 0]
        # loc_nod['2'] = [-1 * x_delta, 1 * y_delta, 0]
        # loc_nod['3'] = [-1 * x_delta, 2 * y_delta, 0]
        # loc_nod['4'] = [-1 * x_delta, 3 * y_delta, 0]
        # loc_nod['5'] = [0, 4 * y_delta, 0]
        # loc_nod['6'] = [1 * x_delta, 3 * y_delta, 0]
        # loc_nod['7'] = [1 * x_delta, 2 * y_delta, 0]
        # loc_nod['8'] = [1 * x_delta, 1 * y_delta, 0]

        # INITIAL SIMULATED ANCHOR AND STEPS
        self.anchors = [[self.Rs * np.cos(thi) - 5, self.Rs * np.sin(thi)] for thi in self.theta]
        self.anchors = np.asarray(self.anchors)

        # self.steps = [[self.rs * np.cos(thi) - 5 + np.random.rand(), 1.5 * self.rs * np.cos(thi)] for thi in self.theta]
        perim = 4
        self.steps = [[perim * np.random.rand() * np.cos(np.random.rand() * 2 * np.pi) - 5 + np.random.randn(),
                       perim * np.random.rand() * np.sin(np.random.rand() * 2 * np.pi)]
                      for _ in np.arange(self.M)]
        self.steps = np.asarray(self.steps)

        # INITIAL THE RANGE MEASUREMENTS WITH SIMULATED RANDOM NOISE
        # self.dists = np.asarray([[get_dist(pa, ps) + np.random.rand() for pa in self.anchors] for ps in self.steps])
        self.dists = get_DistMat(anchors=self.anchors, steps=self.steps)

        # self.vis_groundtruth()

        # self.initi_guess = [0] * (2*(self.M+self.N-1))
        self.initi_guess = np.random.randn(2*(self.M+self.N-1))
        self.loc_ap_xy_pre = [0] * (2*self.N)

        self.loc_xy = np.concatenate([self.anchors.flatten(), self.steps.flatten()])
        self.loc_xy = np.asarray(self.loc_xy)

        self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, self.initi_guess)

        # # create constraints
        # con1 = {'type': 'ineq', 'fun': constraint1, 'args': (dists,)}
        # con2 = {'type': 'ineq', 'fun': constraint2, 'args': (dists,)}
        # con3 = {'type': 'ineq', 'fun': constraint3, 'args': (dists,)}
        # con4 = {'type': 'ineq', 'fun': constraint4, 'args': (dists,)}
        # cons = [con1, con2, con3, con4]
        # method = 'SLSQP'
        #
        # # get optimization
        # res_xy = opt_loc_ranges(dists, initi_guess, cons)
        #
        # loc_ap_xy = res_xy[:8]
        # loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])
        #
        # loc_sp_xy = res_xy[8:]

        # GET THE RANGE MEASUREMENTS AFTER 1ST ROUND
        # dists_post = [
        #     [get_dist(loc_ap_xy[2 * ix_loc_ap:2 * ix_loc_ap + 2], loc_sp_xy[2 * ix_loc_sp:2 * ix_loc_sp + 2]) for
        #      ix_loc_ap in range(len(loc_ap_xy) // 2)]
        #     for ix_loc_sp in range(len(loc_sp_xy) // 2)]

        # print(res_xy)

        # VISUALIZE THE 1ST ESTIMATE OF LOCATIONS
        # self.vis_loc_est(self.loc_ap_xy, self.loc_sp_xy)
        # plt.figure()
        # x = loc_ap_xy[np.arange(5) * 2]
        # y = loc_ap_xy[np.arange(5) * 2 + 1]
        # plt.scatter(x, y, marker='*')
        # x = loc_sp_xy[np.arange(5) * 2]
        # y = loc_sp_xy[np.arange(5) * 2 + 1]
        # plt.scatter(x, y, marker='*')
        # # plt.show()

        # ALIGN WITH ROTATION after 1st round
        self.loc_ap_xy, res_err_rot = self.get_rot_opt(self.loc_ap_xy, self.loc_xy)
        # anchor_pairs = [loc_ap_xy[2:10], loc_xy[2:10]]
        # res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))
        # rot = res_ROT.x
        # print('===============')
        # print(rot)
        # print('===============')
        #
        # loc_x_rot, loc_y_rot = rotate_xy(loc_ap_xy[2:10], theta=rot)
        # loc_x_rot, loc_y_rot = np.concatenate([[0], loc_x_rot]), np.concatenate([[0], loc_y_rot])
        #
        # plt.figure()
        #
        # plt.scatter(loc_x_rot, loc_y_rot, marker='*')
        #
        # loc_ap_xy[np.arange(2, 10, 2)] = loc_x_rot[1:]
        # loc_ap_xy[np.arange(2, 10, 2) + 1] = loc_y_rot[1:]

        # self.vis_rot_loc(self.loc_ap_xy, self.loc_xy)

    def gen_step_rangemeas(self, sigma=0):
        perim = 4
        self.steps = [[perim * np.random.rand() * np.cos(np.random.rand() * 2 * np.pi) - 5 + np.random.randn(),
                       perim * np.random.rand() * np.sin(np.random.rand() * 2 * np.pi)]
                      for _ in np.arange(self.M)]
        self.steps = np.asarray(self.steps)

        self.loc_xy = np.concatenate([self.anchors.flatten(), self.steps.flatten()])
        self.loc_xy = np.asarray(self.loc_xy)
        # self.vis_groundtruth()

        # INITIAL THE RANGE MEASUREMENTS WITH SIMULATED RANDOM NOISE
        # self.dists = np.asarray([[get_dist(pa, ps) + np.random.rand() for pa in self.anchors] for ps in self.steps])
        self.dists = get_DistMat(anchors=self.anchors, steps=self.steps, sigma=sigma)

    def update_step_converge(self, sigma=0):
        # count #round for converged calibration
        flag = True
        self.gen_step_rangemeas(sigma=sigma)

        diversity_var = np.var(self.dists, axis=0).sum()
        print('var:{}'.format(diversity_var))

        self.initi_guess = np.random.randn(2*(self.M+self.N-1))

        self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, self.initi_guess)
        # self.vis_loc_est(self.loc_ap_xy, self.loc_sp_xy)

        # multiple rounds for repeated calibration
        initi_guess = self.initi_guess
        cnt_rounds = 1
        threshold = 2*sigma
        while np.sqrt(np.sum((initi_guess[:2*self.N-2] - self.loc_ap_xy[2:])**2)/(self.N-1))>threshold:
            self.gen_step_rangemeas(sigma=sigma)
            initi_guess = np.random.randn(2*(self.M+self.N-1))
            initi_guess[:2*self.N-2] = self.loc_ap_xy[2:]
            self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, initi_guess)
            cnt_rounds += 1
            if cnt_rounds > 100:
                flag = False
                break

        # rotate to evaluate against ground truth
        loc_ap_xy, res_err_rot = self.get_rot_opt(self.loc_ap_xy, self.loc_xy)
        #         flipped ap xy

        def flip_loc(loc_ap_xy):
            n_len = len(loc_ap_xy)//2
            loc_ap_xy_ = np.array(loc_ap_xy)
            loc_ap_xy_[2*np.arange(n_len)+1] = -loc_ap_xy_[2*np.arange(n_len)+1]
            return loc_ap_xy_

        loc_ap_xy_ = flip_loc(self.loc_ap_xy)
        loc_ap_xy_flipped, res_err_rot_flipped = self.get_rot_opt(loc_ap_xy_, self.loc_xy)
        if res_err_rot_flipped < res_err_rot:
            self.loc_ap_xy = loc_ap_xy_flipped
        else:
            self.loc_ap_xy = loc_ap_xy

        # self.vis_rot_loc(self.loc_ap_xy, self.loc_xy)
        # plt.show()

        return diversity_var, res_err_range, res_err_rot, cnt_rounds, flag

    def mds_test(self, sigma=0, n_round=1):
        self.M = 50
        self.gen_step_rangemeas(sigma=sigma)

        dists = self.dists

        n_step, n_node = np.shape(dists)

        dist_mat = []

        for i in range(n_node):
            dist_i = dists[:, i]
            ix_min = np.argmin(dist_i)
            dist_v = dists[ix_min]
            dist_v[i] = 0
            dist_mat.append(dist_v)

        dist_mat = np.asarray(dist_mat)



        diversity_var = np.var(self.dists, axis=0).sum()
        print('var:{}'.format(diversity_var))

        self.initi_guess = np.random.randn(2 * (self.M + self.N - 1))

        self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, self.initi_guess)
        # self.vis_loc_est(self.loc_ap_xy, self.loc_sp_xy)

        # multiple rounds for repeated calibration
        for _ in range(1, n_round):
            self.gen_step_rangemeas(sigma=sigma)
            initi_guess = np.random.randn(2 * (self.M + self.N - 1))
            initi_guess[:2 * self.N - 2] = self.loc_ap_xy[2:]
            self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, initi_guess)

        loc_ap_xy, res_err_rot = self.get_rot_opt(self.loc_ap_xy, self.loc_xy)

        #         flipped ap xy

        def flip_loc(loc_ap_xy):
            n_len = len(loc_ap_xy) // 2
            loc_ap_xy_ = np.array(loc_ap_xy)
            loc_ap_xy_[2 * np.arange(n_len) + 1] = -loc_ap_xy_[2 * np.arange(n_len) + 1]
            return loc_ap_xy_

        loc_ap_xy_ = flip_loc(self.loc_ap_xy)
        loc_ap_xy_flipped, res_err_rot_flipped = self.get_rot_opt(loc_ap_xy_, self.loc_xy)
        if res_err_rot_flipped < res_err_rot:
            self.loc_ap_xy = loc_ap_xy_flipped
        else:
            self.loc_ap_xy = loc_ap_xy

        # self.vis_rot_loc(self.loc_ap_xy, self.loc_xy)
        # plt.show()

        return diversity_var, res_err_range, res_err_rot


    def update_step_nrounds_mds(self, sigma=0, n_round=1):

        self.gen_step_rangemeas(sigma=sigma)

        diversity_var = np.var(self.dists, axis=0).sum()
        print('var:{}'.format(diversity_var))

        self.initi_guess = np.random.randn(2*(self.M+self.N-1))

        self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, self.initi_guess)
        # self.vis_loc_est(self.loc_ap_xy, self.loc_sp_xy)

        # multiple rounds for repeated calibration
        for _ in range(1, n_round):
            self.gen_step_rangemeas(sigma=sigma)
            initi_guess = np.random.randn(2*(self.M+self.N-1))
            initi_guess[:2*self.N-2] = self.loc_ap_xy[2:]
            self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, initi_guess)

        loc_ap_xy, res_err_rot = self.get_rot_opt(self.loc_ap_xy, self.loc_xy)
        #         flipped ap xy

        def flip_loc(loc_ap_xy):
            n_len = len(loc_ap_xy)//2
            loc_ap_xy_ = np.array(loc_ap_xy)
            loc_ap_xy_[2*np.arange(n_len)+1] = -loc_ap_xy_[2*np.arange(n_len)+1]
            return loc_ap_xy_

        loc_ap_xy_ = flip_loc(self.loc_ap_xy)
        loc_ap_xy_flipped, res_err_rot_flipped = self.get_rot_opt(loc_ap_xy_, self.loc_xy)
        if res_err_rot_flipped < res_err_rot:
            self.loc_ap_xy = loc_ap_xy_flipped
        else:
            self.loc_ap_xy = loc_ap_xy

        # self.vis_rot_loc(self.loc_ap_xy, self.loc_xy)
        # plt.show()

        return diversity_var, res_err_range, res_err_rot

    def update_step(self, sigma=0):
        # single step update
        perim = 4

        self.steps = [[perim * np.random.rand() * np.cos(np.random.rand()*2*np.pi) - 5 + np.random.randn(),
                       perim * np.random.rand() * np.sin(np.random.rand()*2*np.pi)]
                      for _ in np.arange(self.M)]
        self.steps = np.asarray(self.steps)

        # self.vis_groundtruth()

        # INITIAL THE RANGE MEASUREMENTS WITH SIMULATED RANDOM NOISE
        # self.dists = np.asarray([[get_dist(pa, ps) + np.random.rand() for pa in self.anchors] for ps in self.steps])
        self.dists = get_DistMat(anchors=self.anchors, steps=self.steps, sigma=sigma)
        diversity_var = np.var(self.dists, axis=0).sum()
        print('var:{}'.format(diversity_var))

        self.initi_guess = np.random.randn(2*(self.M+self.N-1))

        self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, self.initi_guess)
        # self.vis_loc_est(self.loc_ap_xy, self.loc_sp_xy)

        loc_ap_xy, res_err_rot = self.get_rot_opt(self.loc_ap_xy, self.loc_xy)
        #         flipped ap xy

        def flip_loc(loc_ap_xy):
            n_len = len(loc_ap_xy)//2
            loc_ap_xy_ = np.array(loc_ap_xy)
            loc_ap_xy_[2*np.arange(n_len)+1] = -loc_ap_xy_[2*np.arange(n_len)+1]
            return loc_ap_xy_

        loc_ap_xy_ = flip_loc(self.loc_ap_xy)
        loc_ap_xy_flipped, res_err_rot_flipped = self.get_rot_opt(loc_ap_xy_, self.loc_xy)
        if res_err_rot_flipped < res_err_rot:
            self.loc_ap_xy = loc_ap_xy_flipped
        else:
            self.loc_ap_xy = loc_ap_xy

        # self.vis_rot_loc(self.loc_ap_xy, self.loc_xy)
        # plt.show()

        return diversity_var, res_err_range, res_err_rot


    def update_step_nrounds(self, sigma=0, n_round=1):
        # several rounds calibration
        # perim = 4
        #
        # self.steps = [[perim * np.random.rand() * np.cos(np.random.rand()*2*np.pi) - 5 + np.random.randn(),
        #                perim * np.random.rand() * np.sin(np.random.rand()*2*np.pi)]
        #               for _ in np.arange(self.M)]
        # self.steps = np.asarray(self.steps)
        #
        # # self.vis_groundtruth()
        #
        # # INITIAL THE RANGE MEASUREMENTS WITH SIMULATED RANDOM NOISE
        # # self.dists = np.asarray([[get_dist(pa, ps) + np.random.rand() for pa in self.anchors] for ps in self.steps])
        # self.dists = get_DistMat(anchors=self.anchors, steps=self.steps, sigma=sigma)
        self.gen_step_rangemeas(sigma=sigma)

        diversity_var = np.var(self.dists, axis=0).sum()
        print('var:{}'.format(diversity_var))

        self.initi_guess = np.random.randn(2*(self.M+self.N-1))

        self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, self.initi_guess)
        # self.vis_loc_est(self.loc_ap_xy, self.loc_sp_xy)

        # multiple rounds for repeated calibration
        for _ in range(1, n_round):
            self.gen_step_rangemeas(sigma=sigma)
            initi_guess = np.random.randn(2*(self.M+self.N-1))
            initi_guess[:2*self.N-2] = self.loc_ap_xy[2:]
            self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, initi_guess)

        loc_ap_xy, res_err_rot = self.get_rot_opt(self.loc_ap_xy, self.loc_xy)
        #         flipped ap xy

        def flip_loc(loc_ap_xy):
            n_len = len(loc_ap_xy)//2
            loc_ap_xy_ = np.array(loc_ap_xy)
            loc_ap_xy_[2*np.arange(n_len)+1] = -loc_ap_xy_[2*np.arange(n_len)+1]
            return loc_ap_xy_

        loc_ap_xy_ = flip_loc(self.loc_ap_xy)
        loc_ap_xy_flipped, res_err_rot_flipped = self.get_rot_opt(loc_ap_xy_, self.loc_xy)
        if res_err_rot_flipped < res_err_rot:
            self.loc_ap_xy = loc_ap_xy_flipped
        else:
            self.loc_ap_xy = loc_ap_xy

        # self.vis_rot_loc(self.loc_ap_xy, self.loc_xy)
        # plt.show()

        return diversity_var, res_err_range, res_err_rot

    def update_step(self, sigma=0):
        # single step update
        perim = 4

        self.steps = [[perim * np.random.rand() * np.cos(np.random.rand()*2*np.pi) - 5 + np.random.randn(),
                       perim * np.random.rand() * np.sin(np.random.rand()*2*np.pi)]
                      for _ in np.arange(self.M)]
        self.steps = np.asarray(self.steps)

        # self.vis_groundtruth()

        # INITIAL THE RANGE MEASUREMENTS WITH SIMULATED RANDOM NOISE
        # self.dists = np.asarray([[get_dist(pa, ps) + np.random.rand() for pa in self.anchors] for ps in self.steps])
        self.dists = get_DistMat(anchors=self.anchors, steps=self.steps, sigma=sigma)
        diversity_var = np.var(self.dists, axis=0).sum()
        print('var:{}'.format(diversity_var))

        self.initi_guess = np.random.randn(2*(self.M+self.N-1))

        self.loc_ap_xy, self.loc_sp_xy, res_err_range = self.get_loc_opt(self.dists, self.initi_guess)
        # self.vis_loc_est(self.loc_ap_xy, self.loc_sp_xy)

        loc_ap_xy, res_err_rot = self.get_rot_opt(self.loc_ap_xy, self.loc_xy)
        #         flipped ap xy

        def flip_loc(loc_ap_xy):
            n_len = len(loc_ap_xy)//2
            loc_ap_xy_ = np.array(loc_ap_xy)
            loc_ap_xy_[2*np.arange(n_len)+1] = -loc_ap_xy_[2*np.arange(n_len)+1]
            return loc_ap_xy_

        loc_ap_xy_ = flip_loc(self.loc_ap_xy)
        loc_ap_xy_flipped, res_err_rot_flipped = self.get_rot_opt(loc_ap_xy_, self.loc_xy)
        if res_err_rot_flipped < res_err_rot:
            self.loc_ap_xy = loc_ap_xy_flipped
        else:
            self.loc_ap_xy = loc_ap_xy

        # self.vis_rot_loc(self.loc_ap_xy, self.loc_xy)
        # plt.show()

        return diversity_var, res_err_range, res_err_rot

    def iterations(self, N_iter=500, sigma=0.1, n_round=1):
        from tqdm import tqdm
        diversity_var_s, res_err_range_s, res_err_rot_s = [], [], []
        loc_ap_xy_s, loc_tg_xy_s = [], []
        dists_list = []
        for ix in tqdm(np.arange(N_iter)):
            print('ROUND:{}'.format(ix))
            # diversity_var, res_err_range, res_err_rot = self.update_step(sigma=sigma)

            diversity_var, res_err_range, res_err_rot = self.update_step_nrounds(sigma=sigma, n_round=n_round)

            diversity_var_s.append(diversity_var)
            res_err_range_s.append(res_err_range)
            res_err_rot_s.append(res_err_rot)
            loc_ap_xy_s.append(self.loc_ap_xy)
            loc_tg_xy_s.append(self.loc_xy)
            dists_list.append(self.dists)

        # plt.figure()
        # plt.scatter(diversity_var_s, res_err_range_s, label='var_res_range')
        # plt.scatter(diversity_var_s, res_err_rot_s, label='var_res_rot')
        # plt.legend()
        # plt.show()

        return np.asarray(diversity_var_s), np.asarray(res_err_range_s)/(self.M+self.N), np.asarray(res_err_rot_s)/(self.M+self.N), np.asarray(loc_ap_xy_s), np.asarray(loc_tg_xy_s), np.asarray(dists_list)

    def iterations_converge(self, N_iter=2000, sigma=0.1):
        from tqdm import tqdm
        diversity_var_s, res_err_range_s, res_err_rot_s = [], [], []
        loc_ap_xy_s, loc_tg_xy_s = [], []
        dists_list = []
        cnt_list = []
        time_list = []
        flag_list = []
        for ix in tqdm(np.arange(N_iter)):
            print('ROUND:{}'.format(ix))
            # diversity_var, res_err_range, res_err_rot = self.update_step(sigma=sigma)
            start_time = time.time()
            diversity_var, res_err_range, res_err_rot, cnt_rounds, flag = self.update_step_converge(sigma=sigma)
            time_exe = time.time() - start_time
            diversity_var_s.append(diversity_var)
            res_err_range_s.append(res_err_range)
            res_err_rot_s.append(res_err_rot)
            loc_ap_xy_s.append(self.loc_ap_xy)
            loc_tg_xy_s.append(self.loc_xy)
            dists_list.append(self.dists)

            cnt_list.append(cnt_rounds)
            time_list.append(time_exe)
            flag_list.append(flag)

        return np.asarray(diversity_var_s), np.asarray(res_err_range_s)/(self.M+self.N), \
               np.asarray(res_err_rot_s)/(self.M+self.N), \
               np.asarray(loc_ap_xy_s), np.asarray(loc_tg_xy_s), np.asarray(dists_list), \
               np.asarray(cnt_list), np.asarray(time_list), np.asarray(flag_list)


    def get_loc_opt(self, dists, initi_guess, method='SLSQP'):

        # create constraints
        con1 = {'type': 'ineq', 'fun': constraint1, 'args': (dists,)}
        con2 = {'type': 'ineq', 'fun': constraint2, 'args': (dists,)}
        con3 = {'type': 'ineq', 'fun': constraint3, 'args': (dists,)}
        con4 = {'type': 'ineq', 'fun': constraint4, 'args': (dists,)}
        cons = [con1, con2, con3, con4]

        # # =======================================
        # # additional constraint according to MDS
        # M, N = np.shape(dists)
        # for i in range(N):
        #     j_s = np.argmin(dists[:, i])
        #     rs_i = dists[j_s]
        #     for ii, rs_ii in enumerate(rs_i):
        #         if ii == i:
        #             continue
        #
        #         func = lambda loc_xy : sum((np.concatenate([np.zeros(2), loc_xy[:N * 2 - 2]])[ii*2:ii*2+2] - np.concatenate([np.zeros(2), loc_xy[:N * 2 - 2]])[i*2:i*2+2])**2) - rs_ii**2
        #         con_mds = {'type': 'ineq', 'fun': func}
        #         cons.append(con_mds)
        #
        # # additional constraint according to MDS
        # # =======================================

        # get optimization
        res_xy, res_err = opt_loc_ranges(dists, initi_guess, cons, method=method)

        loc_ap_xy = res_xy[:2*self.N-2]
        loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

        loc_sp_xy = res_xy[2*self.N-2:]

        return loc_ap_xy, loc_sp_xy, res_err


    def get_rot_opt(self, loc_ap_xy, loc_tg_xy, method='SLSQP'):
        self.loc_ap_xy_pre = np.array(loc_ap_xy)
        # ALIGN WITH ROTATION after 1st round
        anchor_pairs = [loc_ap_xy[2:10], loc_tg_xy[2:10]]

        # optimization by rotating
        rot, res_err = opt_rot_loc(anchor_pairs=anchor_pairs)
        # res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))
        # rot = res_ROT.x
        print('===============')
        # print(res_ROT.x)
        print('rot err:', res_err)
        print('rot ang', rot)
        print('===============')


        # rotation
        loc_x_rot, loc_y_rot = rotate_xy(loc_ap_xy[2:10], theta=rot)
        loc_x_rot, loc_y_rot = np.concatenate([[0], loc_x_rot]), np.concatenate([[0], loc_y_rot])

        loc_ap_xy[np.arange(2, 10, 2)] = loc_x_rot[1:]
        loc_ap_xy[np.arange(2, 10, 2) + 1] = loc_y_rot[1:]

        return loc_ap_xy, res_err

    def vis_groundtruth(self):

        # VISUALIZE THE GROUND TRUTH

        x = self.anchors[:, 0]
        y = self.anchors[:, 1]
        plt.scatter(x=x, y=y, marker='^', label='anchor')
        x = self.steps[:, 0]
        y = self.steps[:, 1]
        plt.scatter(x=x, y=y, marker='*', label='step')
        plt.legend()
        plt.title('loc gd')
        # plt.show()

    def vis_loc_est(self, loc_ap_xy, loc_sp_xy):
        # VISUALIZE THE 1ST ESTIMATE OF LOCATIONS
        plt.figure()
        x = loc_ap_xy[np.arange(self.N) * 2]
        y = loc_ap_xy[np.arange(self.N) * 2 + 1]
        plt.scatter(x, y, marker='^', label='anchor')
        x = loc_sp_xy[np.arange(self.M) * 2]
        y = loc_sp_xy[np.arange(self.M) * 2 + 1]
        plt.scatter(x, y, marker='*', label='step')
        plt.legend()
        plt.title('loc est')

        # plt.show()

    def vis_rot_loc(self, loc_ap_xy, loc_tg_xy):
        # VISUALIZE THE 1ST ESTIMATE OF LOCATIONS
        plt.figure()
        x = loc_ap_xy[np.arange(self.N) * 2]
        y = loc_ap_xy[np.arange(self.N) * 2 + 1]
        plt.scatter(x, y, marker='^', label='anchor')
        x = loc_tg_xy[np.arange(self.N) * 2]
        y = loc_tg_xy[np.arange(self.N) * 2 + 1]
        plt.scatter(x, y, marker='*', label='target')
        plt.legend()
        plt.title('loc rot')

        # plt.show()


def get_loc_lse(loc_nod, list_dists, loc_step_xy):
    # P = None
    P = lx.Project(mode='2D', solver='LSE')

    n_nod = len(loc_nod)//2

    nodes_code = [str(ii) for ii in range(n_nod)]

    for i, nod in enumerate(nodes_code):
        P.add_anchor(nod, loc_nod[2*i:2*i+2])

    n_dists = len(list_dists)

    loc_steps = []
    for dists in list_dists:

        t_rdm, _ = P.add_target()

        for i, nod in enumerate(nodes_code):
            # dist_ske = list_dist_ske_[nod][ix]
            dist_r = dists[i]
            # t_ske.add_measure(nod, dist_ske)
            t_rdm.add_measure(nod, dist_r)

        P.solve()

        loc_r_ = np.asarray([t_rdm.loc.x, t_rdm.loc.y])

        loc_steps.append(loc_r_)

    # print(loc_step_xy)

    return np.asarray(loc_steps)


def test_main():

    theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
    Rs = 5
    rs = 2

    # given N anchor nodes, M step locations;
    # N >= 3; M >=2;
    # N ()

    # x_delta = 1.38  # 1.418
    # y_delta = 1.6
    # loc_nod['1'] = [0, 0, 0]
    # loc_nod['2'] = [-1 * x_delta, 1 * y_delta, 0]
    # loc_nod['3'] = [-1 * x_delta, 2 * y_delta, 0]
    # loc_nod['4'] = [-1 * x_delta, 3 * y_delta, 0]
    # loc_nod['5'] = [0, 4 * y_delta, 0]
    # loc_nod['6'] = [1 * x_delta, 3 * y_delta, 0]
    # loc_nod['7'] = [1 * x_delta, 2 * y_delta, 0]
    # loc_nod['8'] = [1 * x_delta, 1 * y_delta, 0]


    # INITIAL SIMULATED ANCHOR AND STEPS
    anchors = [[Rs*np.cos(thi)-5, Rs*np.sin(thi)] for thi in theta]
    steps = [[rs*np.cos(thi)-5+np.random.rand(), 1.5*rs*np.cos(thi)] for thi in theta]
    anchors = np.asarray(anchors)
    steps = np.asarray(steps)

    # INITIAL THE RANGE MEASUREMENTS WITH SIMULATED RANDOM NOISE
    dists = np.asarray([[get_dist(pa, ps) + np.random.rand() for pa in anchors] for ps in steps])

    # VISUALIZE THE GROUND TRUTH
    import matplotlib.pyplot as plt
    x=anchors[:,0]
    y=anchors[:,1]
    plt.scatter(x=x, y=y, marker='*')
    x=steps[:,0]
    y=steps[:,1]
    plt.scatter(x=x, y=y, marker='^')
    # plt.show()


    initi_guess = [0]*18
    loc_xy = np.concatenate([anchors.flatten(),steps.flatten()])
    loc_xy = np.asarray(loc_xy)

    # create constraints
    con1 = {'type':'ineq', 'fun':constraint1, 'args':(dists,)}
    con2 = {'type':'ineq', 'fun':constraint2, 'args':(dists,)}
    con3 = {'type':'ineq', 'fun':constraint3, 'args':(dists,)}
    con4 = {'type':'ineq', 'fun':constraint4, 'args':(dists,)}
    cons = [con1, con2, con3, con4]
    method = 'SLSQP'

    # get optimization
    res_xy = opt_loc_ranges(dists, initi_guess, cons)

    loc_ap_xy = res_xy[:8]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = res_xy[8:]

    # GET THE RANGE MEASUREMENTS AFTER 1ST ROUND
    dists_post = [[get_dist(loc_ap_xy[2*ix_loc_ap:2*ix_loc_ap+2], loc_sp_xy[2*ix_loc_sp:2*ix_loc_sp+2]) for ix_loc_ap in range(len(loc_ap_xy)//2)]
                  for ix_loc_sp in range(len(loc_sp_xy)//2)]

    print(res_xy)

    # VISUALIZE THE 1ST ESTIMATE OF LOCATIONS
    plt.figure()
    x=loc_ap_xy[np.arange(5)*2]
    y=loc_ap_xy[np.arange(5)*2+1]
    plt.scatter(x,y,marker='*')
    x=loc_sp_xy[np.arange(5)*2]
    y=loc_sp_xy[np.arange(5)*2+1]
    plt.scatter(x,y,marker='*')
    # plt.show()

    # ALIGN WITH ROTATION after 1st round
    anchor_pairs = [loc_ap_xy[2:10], loc_xy[2:10]]
    res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))
    rot = res_ROT.x
    print('===============')
    print(rot)
    print('===============')

    loc_x_rot, loc_y_rot = rotate_xy(loc_ap_xy[2:10], theta=rot)
    loc_x_rot, loc_y_rot = np.concatenate([[0], loc_x_rot]), np.concatenate([[0],loc_y_rot])

    plt.figure()

    plt.scatter(loc_x_rot,loc_y_rot,marker='*')

    loc_ap_xy[np.arange(2,10,2)]= loc_x_rot[1:]
    loc_ap_xy[np.arange(2,10,2)+1]= loc_y_rot[1:]

    loc_ap_xy_pre = np.array(loc_ap_xy)

    # GENERATE 2ND ROUND STEP LOCATIONS
    steps = [[rs*np.cos(thi)*np.random.rand()-5, rs*np.sin(thi)*np.random.rand()] for thi in theta]


    # GET THE RANGE MEASUREMENTS IN THE SECOND ROUND
    dists = get_DistMat(anchors, steps)

    # UPDATE THE INITIAL LOCATION OF ANCHORS WITH ESTIMATED LOCATION FROM 1ST ROUND
    initi_guess[:8] = loc_ap_xy[2:]
    # res = minimize(objective, initi_guess, dists, method=method, constraints=cons)
    # res_xy = res.x

    # get optimization
    res_xy = opt_loc_ranges(dists, initi_guess, cons)

    loc_ap_xy = res_xy[:8]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = res_xy[8:]


    # VISUALIZE THE 2ND ESTIMATE OF LOCATIONS
    plt.figure()
    x=loc_ap_xy[np.arange(5)*2]
    y=loc_ap_xy[np.arange(5)*2+1]
    plt.scatter(x,y,marker='*')
    x=loc_sp_xy[np.arange(5)*2]
    y=loc_sp_xy[np.arange(5)*2+1]
    plt.scatter(x,y,marker='*')

    # ALIGN WITH ROTATION after 2nd round
    anchor_pairs = [loc_ap_xy[2:10], loc_xy[2:10]]
    res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))
    rot = res_ROT.x
    print('===============')
    print(rot)
    print('===============')
    loc_x_rot, loc_y_rot = rotate_xy(loc_ap_xy[2:10], theta=rot)
    loc_x_rot, loc_y_rot = np.concatenate([[0], loc_x_rot]), np.concatenate([[0],loc_y_rot])

    plt.figure()

    plt.scatter(loc_x_rot,loc_y_rot,marker='*')

    anchor_pairs = [loc_ap_xy[2:10], loc_ap_xy_pre[2:10]]
    res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))


    loc_ap_xy[np.arange(2,10,2)]= loc_x_rot[1:]
    loc_ap_xy[np.arange(2,10,2)+1]= loc_y_rot[1:]

    loc_ap_xy_pre = np.array(loc_ap_xy)

    # =========================================== repeating 1
    # GENERATE 2ND ROUND STEP LOCATIONS
    steps = [[rs*np.cos(thi)*np.random.rand()-5, rs*np.sin(thi)*np.random.rand()] for thi in theta]

    # GET THE RANGE MEASUREMENTS IN THE SECOND ROUND
    dists = get_DistMat(anchors, steps)

    # UPDATE THE INITIAL LOCATION OF ANCHORS WITH ESTIMATED LOCATION FROM 1ST ROUND
    initi_guess[:8] = loc_ap_xy[2:]
    # res = minimize(objective, initi_guess, dists, method=method, constraints=cons)
    # res_xy = res.x
    # get optimization
    res_xy = opt_loc_ranges(dists, initi_guess, cons)
    loc_ap_xy = res_xy[:8]
    loc_ap_xy = np.concatenate([np.zeros(2), loc_ap_xy])

    loc_sp_xy = res_xy[8:]

    # VISUALIZE THE 2ND ESTIMATE OF LOCATIONS
    plt.figure()
    x=loc_ap_xy[np.arange(5)*2]
    y=loc_ap_xy[np.arange(5)*2+1]
    plt.scatter(x,y,marker='*')
    x=loc_sp_xy[np.arange(5)*2]
    y=loc_sp_xy[np.arange(5)*2+1]
    plt.scatter(x,y,marker='*')

    # ALIGN WITH ROTATION after 2nd round
    anchor_pairs = [loc_ap_xy[2:10], loc_xy[2:10]]
    res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))
    rot = res_ROT.x
    print('===============')
    print(rot)
    print('===============')
    loc_x_rot, loc_y_rot = rotate_xy(loc_ap_xy[2:10], theta=rot)
    loc_x_rot, loc_y_rot = np.concatenate([[0], loc_x_rot]), np.concatenate([[0],loc_y_rot])

    plt.figure()

    plt.scatter(loc_x_rot,loc_y_rot,marker='*')

    anchor_pairs = [loc_ap_xy[2:10], loc_ap_xy_pre[2:10]]
    res_ROT = minimize(obj_rotate, 0, args=anchor_pairs, method=method, bounds=((-np.pi, np.pi),))


    loc_ap_xy[np.arange(2,10,2)]= loc_x_rot[1:]
    loc_ap_xy[np.arange(2,10,2)+1]= loc_y_rot[1:]

    # plt.show()


def vis_converge(m_step=5, sigma=0.3):

    f_loc_ap_xy_s = 'loc_ap_xy_s.npy'
    f_loc_tg_xy_s = 'loc_tg_xy_s.npy'
    f_dists = 'dists.npy'
    directory = os.getcwd()
    pth_res = os.path.join(directory, 'res_loc_convg')
    if not os.path.exists(pth_res):
        os.mkdir(pth_res)

    postfix = 'res_n_{}_sigma_{}'.format(m_step, int(sigma * 10))
    pth_dat = os.path.join(pth_res, postfix)
    if not os.path.exists(pth_dat):
        os.mkdir(pth_dat)
    pth_f_loc_ap_xy_s = os.path.join(pth_dat, f_loc_ap_xy_s)
    pth_f_loc_tg_xy_s = os.path.join(pth_dat, f_loc_tg_xy_s)
    pth_f_dists = os.path.join(pth_dat, f_dists)
    pth_f_cnts = os.path.join(pth_dat, 'cnts.npy')
    pth_f_times = os.path.join(pth_dat, 'times.npy')
    pth_f_flags = os.path.join(pth_dat, 'flags.npy')

    loc_ap_xy_s = np.load(pth_f_loc_ap_xy_s)
    loc_tg_xy_s = np.load(pth_f_loc_tg_xy_s)
    dists = np.load(pth_f_dists)
    cnt_list = np.load(pth_f_cnts)
    time_list = np.load(pth_f_times)
    flag_list = np.load(pth_f_flags)

    ix_flag_p = np.where(flag_list==True)[0]
    ix_flag_n = np.where(flag_list==False)[0]

    ix_cnt_ovf = np.where(cnt_list==21)[0]

    print(np.all(ix_flag_n==ix_cnt_ovf))
    # =========================
    # lse get step location based on self-calibrated anchors
    loc_step_xy_s = []

    for loc_ap_xy_cur, dists_cur, loc_tg_xy_cur in zip(loc_ap_xy_s, dists, loc_tg_xy_s):
        loc_steps = get_loc_lse(loc_ap_xy_cur, dists_cur, loc_tg_xy_cur)
        loc_step_xy_s.append(loc_steps)
    loc_step_xy_s = np.asarray(loc_step_xy_s)
    f_loc_step_xy_s = 'loc_step_xy_s.npy'
    pth_f_loc_step_xy_s = os.path.join(pth_dat, f_loc_step_xy_s)
    np.save(pth_f_loc_step_xy_s, np.asarray(loc_step_xy_s))
    # =========================
    # eval diff
    def eva_err_anchor(loc_ap_xy_s, loc_tg_xy_s):

        n_node = len(loc_ap_xy_s) // 2
        loc_tg_xy_s = loc_tg_xy_s[:n_node * 2]

        diff_xs, diff_ys, diff_xys = [], [], []

        for i in range(1, n_node):
            loc_ap_ = loc_ap_xy_s[i * 2:i * 2 + 2]
            loc_tg_ = loc_tg_xy_s[i * 2:i * 2 + 2]
            diff_loc_x, diff_loc_y = abs(loc_ap_ - loc_tg_)
            diff_loc_xy = np.sqrt(diff_loc_x ** 2 + diff_loc_y ** 2)

            diff_xs.append(diff_loc_x)
            diff_ys.append(diff_loc_y)
            diff_xys.append(diff_loc_xy)

        return np.asarray(diff_xs), np.asarray(diff_ys), np.asarray(diff_xys)

    def eva_err_step(loc_step_xy_s, loc_tg_xy_s):

        n_node = len(loc_step_xy_s)
        loc_tg_xy_s = loc_tg_xy_s[-2 * n_node:]

        diff_xs, diff_ys, diff_xys = [], [], []

        for i in range(0, n_node):
            loc_ap_ = loc_step_xy_s[i]
            loc_tg_ = loc_tg_xy_s[i * 2:i * 2 + 2]
            diff_loc_x, diff_loc_y = abs(loc_ap_ - loc_tg_)
            diff_loc_xy = np.sqrt(diff_loc_x ** 2 + diff_loc_y ** 2)

            diff_xs.append(diff_loc_x)
            diff_ys.append(diff_loc_y)
            diff_xys.append(diff_loc_xy)

        return np.asarray(diff_xs), np.asarray(diff_ys), np.asarray(diff_xys)

    diff_a_xs_all, diff_a_ys_all, diff_a_xys_all = [], [], []
    diff_s_xs_all, diff_s_ys_all, diff_s_xys_all = [], [], []

    for loc_ap_xy_, loc_tg_xy_, loc_step_xy_ in zip(loc_ap_xy_s, loc_tg_xy_s, loc_step_xy_s):
        diff_a_xs, diff_a_ys, diff_a_xys = eva_err_anchor(loc_ap_xy_, loc_tg_xy_)
        diff_s_xs, diff_s_ys, diff_s_xys = eva_err_step(loc_step_xy_, loc_tg_xy_)

        diff_a_xs_all.append(diff_a_xs)
        diff_a_ys_all.append(diff_a_ys)
        diff_a_xys_all.append(diff_a_xys)
        diff_s_xs_all.append(diff_s_xs)
        diff_s_ys_all.append(diff_s_ys)
        diff_s_xys_all.append(diff_s_xys)

    diff_a_xs_all, diff_a_ys_all, diff_a_xys_all = np.concatenate(diff_a_xs_all), np.concatenate(
        diff_a_ys_all), np.concatenate(diff_a_xys_all)
    diff_s_xs_all, diff_s_ys_all, diff_s_xys_all = np.concatenate(diff_s_xs_all), np.concatenate(
        diff_s_ys_all), np.concatenate(diff_s_xys_all)

    diff_a_xs_all_s, diff_a_ys_all_s, diff_a_xys_all_s = [], [], []
    diff_s_xs_all_s, diff_s_ys_all_s, diff_s_xys_all_s = [], [], []

    pth_f_diff_a_xs_all = os.path.join(pth_dat, "diff_a_xs_all")
    pth_f_diff_a_ys_all = os.path.join(pth_dat, "diff_a_ys_all")
    pth_f_diff_a_xys_all = os.path.join(pth_dat, "diff_a_xys_all")
    pth_f_diff_s_xs_all = os.path.join(pth_dat, "diff_s_xs_all")
    pth_f_diff_s_ys_all = os.path.join(pth_dat, "diff_s_ys_all")
    pth_f_diff_s_xys_all = os.path.join(pth_dat, "diff_s_xys_all")

    np.save(pth_f_diff_a_xs_all, diff_a_xs_all)
    np.save(pth_f_diff_a_ys_all, diff_a_ys_all)
    np.save(pth_f_diff_a_xys_all, diff_a_xys_all)
    np.save(pth_f_diff_s_xs_all, diff_s_xs_all)
    np.save(pth_f_diff_s_ys_all, diff_s_ys_all)
    np.save(pth_f_diff_s_xys_all, diff_s_xys_all)

    pth_f_diff_a_xs_all = os.path.join(pth_dat, "diff_a_xs_all.npy")
    pth_f_diff_a_ys_all = os.path.join(pth_dat, "diff_a_ys_all.npy")
    pth_f_diff_a_xys_all = os.path.join(pth_dat, "diff_a_xys_all.npy")
    pth_f_diff_s_xs_all = os.path.join(pth_dat, "diff_s_xs_all.npy")
    pth_f_diff_s_ys_all = os.path.join(pth_dat, "diff_s_ys_all.npy")
    pth_f_diff_s_xys_all = os.path.join(pth_dat, "diff_s_xys_all.npy")

    diff_a_xs_all = np.load(pth_f_diff_a_xs_all)
    diff_a_ys_all = np.load(pth_f_diff_a_ys_all)
    diff_a_xys_all = np.load(pth_f_diff_a_xys_all)
    diff_s_xs_all = np.load(pth_f_diff_s_xs_all)
    diff_s_ys_all = np.load(pth_f_diff_s_ys_all)
    diff_s_xys_all = np.load(pth_f_diff_s_xys_all)

    diff_a_xs_all_s.append(diff_a_xs_all)
    diff_a_ys_all_s.append(diff_a_ys_all)
    diff_a_xys_all_s.append(diff_a_xys_all)
    diff_s_xs_all_s.append(diff_s_xs_all)
    diff_s_ys_all_s.append(diff_s_ys_all)
    diff_s_xys_all_s.append(diff_s_xys_all)

    labels = []
    labels.append('#step_{}, sigma_{}'.format(m_step, int(sigma * 100)))
    draw_cdf_e_s(pth_res, diff_a_xs_all_s, labels=labels, title='Loc_anchor_x')
    draw_cdf_e_s(pth_res, diff_a_ys_all_s, labels=labels, title='Loc_anchor_y')
    draw_cdf_e_s(pth_res, diff_a_xys_all_s, labels=labels, title='Loc_anchor_xy')
    draw_cdf_e_s(pth_res, diff_s_xs_all_s, labels=labels, title='Loc_step_x')
    draw_cdf_e_s(pth_res, diff_s_ys_all_s, labels=labels, title='Loc_step_y')
    draw_cdf_e_s(pth_res, diff_s_xys_all_s, labels=labels, title='Loc_step_xy')

    cnt_set = set(cnt_list)

    ix_cnt = dict()
    cnt_dict = dict()
    time_dict = dict()
    diff_a_xys_dict = dict()
    diff_s_xys_dict = dict()

    for cnt in cnt_set:
        ix_cnt[str(cnt)] = np.where(cnt_list==cnt)[0]
        cnt_dict[str(cnt)] = len(ix_cnt[str(cnt)]) / len(cnt_list)
        time_dict[str(cnt)] = time_list[ix_cnt[str(cnt)]]

    for cnt in cnt_set:
        ix_cnt_ = ix_cnt[str(cnt)]
        ix_cnt_4 = np.concatenate([4*ix_cnt_ + i for i in range(4)])
        diff_a_xys_dict[str(cnt)] = diff_a_xys_all[ix_cnt_4]

        ix_cnt_5 = np.concatenate([5*ix_cnt_ + i for i in range(5)])
        diff_s_xys_dict[str(cnt)] = diff_s_xys_all[ix_cnt_5]

    draw_bar_s(pth_res, cnt_dict.values(), labels=cnt_dict.keys(), title='#round distribution')
    draw_boxplot_s(pth_res, time_dict.values(), labels=time_dict.keys(), ylabel='Time Elapsed (s)', title='time elapsed')
    draw_boxplot_s(pth_res, diff_a_xys_dict.values(), labels=diff_a_xys_dict.keys(), title='Loc_anchor_xy')
    draw_boxplot_s(pth_res, diff_s_xys_dict.values(), labels=diff_s_xys_dict.keys(), title='Loc_step_xy')

def draw_bars(data):
    fractions = list(data.values())
    srcs = list(data.keys())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(srcs, fractions,
            width=0.4)


def simu_converge(m_step=5, sigma=0.3):

    f_loc_ap_xy_s = 'loc_ap_xy_s'
    f_loc_tg_xy_s = 'loc_tg_xy_s'
    f_dists = 'dists'
    directory = os.getcwd()
    pth_res = os.path.join(directory, 'res_loc_convg')
    if not os.path.exists(pth_res):
        os.mkdir(pth_res)

    calib = SelfCalibration(M=m_step)

    sigma = np.round(sigma, decimals=3)
    diversity_var, res_err_range, res_err_rot, loc_ap_xy_s, loc_tg_xy_s, dists, cnt_list, time_list, flag_list = \
        calib.iterations_converge(sigma=sigma)

    postfix = 'res_n_{}_sigma_{}'.format(m_step, int(sigma * 10))
    pth_dat = os.path.join(pth_res, postfix)
    if not os.path.exists(pth_dat):
        os.mkdir(pth_dat)
    pth_f_loc_ap_xy_s = os.path.join(pth_dat, f_loc_ap_xy_s)
    pth_f_loc_tg_xy_s = os.path.join(pth_dat, f_loc_tg_xy_s)
    pth_f_dists = os.path.join(pth_dat, f_dists)
    pth_f_cnts = os.path.join(pth_dat, 'cnts')
    pth_f_times = os.path.join(pth_dat, 'times')
    pth_f_flags = os.path.join(pth_dat, 'flags')

    np.save(pth_f_loc_ap_xy_s, loc_ap_xy_s)
    np.save(pth_f_loc_tg_xy_s, loc_tg_xy_s)
    np.save(pth_f_dists, dists)
    np.save(pth_f_cnts, cnt_list)
    np.save(pth_f_times, time_list)
    np.save(pth_f_flags, flag_list)

    print('good')


def eval_simu(n_round=1, m_steps=range(3, 8), sigmas=np.arange(0.0, 0.5, 0.1)):
    f_loc_ap_xy_s = 'loc_ap_xy_s'
    f_loc_tg_xy_s = 'loc_tg_xy_s'
    f_dists = 'dists'
    directory = os.getcwd()
    pth_res = os.path.join(directory, 'res_loc')
    if not os.path.exists(pth_res):
        os.mkdir(pth_res)

    for mm in m_steps:
        # temporal filter
        # if mm != 5:
        #     continue

        calib = SelfCalibration(M=mm)
        # plt.show()
        res_err_range_dict = {}
        res_err_rot_dict = {}
        for sigma in sigmas:
            sigma = np.round(sigma, decimals=3)
            diversity_var , res_err_range, res_err_rot, loc_ap_xy_s, loc_tg_xy_s, dists = calib.iterations(sigma=sigma, n_round=n_round)
            res_err_range_dict[str(sigma)]=res_err_range
            res_err_rot_dict[str(sigma)]=res_err_rot

            postfix = 'res_n_{}_sigma_{}_{}'.format(mm, int(sigma*10), n_round)
            pth_dat = os.path.join(pth_res, postfix)
            if not os.path.exists(pth_dat):
                os.mkdir(pth_dat)
            pth_f_loc_ap_xy_s = os.path.join(pth_dat, f_loc_ap_xy_s)
            pth_f_loc_tg_xy_s = os.path.join(pth_dat, f_loc_tg_xy_s)
            pth_f_dists = os.path.join(pth_dat, f_dists)

            np.save(pth_f_loc_ap_xy_s, loc_ap_xy_s)
            np.save(pth_f_loc_tg_xy_s, loc_tg_xy_s)
            np.save(pth_f_dists, dists)
        # fig, ax = plt.subplots()
        # ax.boxplot(res_err_range_dict.values())
        # ax.set_xticklabels(res_err_range_dict.keys())
        # title = 'sigma_range_err'
        # plt.title(title)
        # ax.set_xlabel('sigma')
        # ax.set_ylabel('range err')
        # ax.set_ylim([0, 250])
        # directory = os.getcwd()
        # if not os.path.exists(os.path.join(directory, 'fig')):
        #     os.mkdir(os.path.join(directory, 'fig'))
        # fig.savefig(os.path.join(directory, 'fig', title+str(mm)), bbox_inches='tight')
        #
        # fig, ax = plt.subplots()
        # ax.boxplot(res_err_rot_dict.values())
        # ax.set_xticklabels(res_err_rot_dict.keys())
        # title = 'sigma_loc_err'
        # plt.title(title)
        # ax.set_xlabel('sigma')
        # ax.set_ylabel('loc err')
        # ax.set_ylim([0, 400])
        #
        # directory = os.getcwd()
        # if not os.path.exists(os.path.join(directory, 'fig')):
        #     os.mkdir(os.path.join(directory, 'fig'))
        # fig.savefig(os.path.join(directory, 'fig', title+str(mm)), bbox_inches='tight')

        # plt.show()
    print('good')


def reproc_simu(n_round=1, m_steps=range(3, 8), sigmas=np.arange(0.0, 0.5, 0.1)):
    f_loc_ap_xy_s = 'loc_ap_xy_s.npy'
    f_loc_tg_xy_s = 'loc_tg_xy_s.npy'
    f_dists = 'dists.npy'
    directory = os.getcwd()
    pth_res = os.path.join(directory, 'res_loc')
    # if not os.path.exists(pth_res):
    #     os.mkdir(pth_res)
    f_loc_step_xy_s = 'loc_step_xy_s.npy'

    for mm in m_steps:

        # temporal filter
        # if mm != 5:
        #     continue

        sigma_m = 0.5
        for sigma in sigmas:
            sigma = np.round(sigma, decimals=3)

            postfix = 'res_n_{}_sigma_{}_{}'.format(mm, int(sigma*10), n_round)
            pth_dat = os.path.join(pth_res, postfix)
            if not os.path.exists(pth_dat):
                os.mkdir(pth_dat)

            pth_f_loc_ap_xy_s = os.path.join(pth_dat, f_loc_ap_xy_s)
            pth_f_loc_tg_xy_s = os.path.join(pth_dat, f_loc_tg_xy_s)
            pth_f_dists = os.path.join(pth_dat, f_dists)

            loc_ap_xy_s = np.load(pth_f_loc_ap_xy_s)
            loc_tg_xy_s = np.load(pth_f_loc_tg_xy_s)
            dists = np.load(pth_f_dists)

            print(np.shape(loc_ap_xy_s))
            print(np.shape(dists))

            loc_step_xy_s = []

            for loc_ap_xy_cur, dists_cur, loc_tg_xy_cur in zip(loc_ap_xy_s, dists, loc_tg_xy_s):
                loc_steps = get_loc_lse(loc_ap_xy_cur, dists_cur, loc_tg_xy_cur)
                loc_step_xy_s.append(loc_steps)

            pth_f_loc_step_xy_s = os.path.join(pth_dat, f_loc_step_xy_s)
            np.save(pth_f_loc_step_xy_s, np.asarray(loc_step_xy_s))


def get_eval_simu(n_round=1, m_steps=range(3, 8), sigmas=np.arange(0.0, 0.5, 0.1)):
    f_loc_ap_xy_s = 'loc_ap_xy_s.npy'
    f_loc_tg_xy_s = 'loc_tg_xy_s.npy'
    f_dists = 'dists.npy'
    directory = os.getcwd()
    pth_res = os.path.join(directory, 'res_loc')
    # if not os.path.exists(pth_res):
    #     os.mkdir(pth_res)
    f_loc_step_xy_s = 'loc_step_xy_s.npy'

    def eva_err_anchor(loc_ap_xy_s, loc_tg_xy_s):

        n_node = len(loc_ap_xy_s) // 2
        loc_tg_xy_s = loc_tg_xy_s[:n_node*2]

        diff_xs, diff_ys, diff_xys = [], [], []

        for i in range(1, n_node):
            loc_ap_ = loc_ap_xy_s[i*2:i*2+2]
            loc_tg_ = loc_tg_xy_s[i*2:i*2+2]
            diff_loc_x, diff_loc_y = abs(loc_ap_-loc_tg_)
            diff_loc_xy = np.sqrt(diff_loc_x**2 + diff_loc_y**2)

            diff_xs.append(diff_loc_x)
            diff_ys.append(diff_loc_y)
            diff_xys.append(diff_loc_xy)

        return np.asarray(diff_xs), np.asarray(diff_ys), np.asarray(diff_xys)

    def eva_err_step(loc_step_xy_s, loc_tg_xy_s):

        n_node = len(loc_step_xy_s)
        loc_tg_xy_s = loc_tg_xy_s[-2*n_node:]

        diff_xs, diff_ys, diff_xys = [], [], []

        for i in range(0, n_node):
            loc_ap_ = loc_step_xy_s[i]
            loc_tg_ = loc_tg_xy_s[i*2:i*2+2]
            diff_loc_x, diff_loc_y = abs(loc_ap_-loc_tg_)
            diff_loc_xy = np.sqrt(diff_loc_x**2 + diff_loc_y**2)

            diff_xs.append(diff_loc_x)
            diff_ys.append(diff_loc_y)
            diff_xys.append(diff_loc_xy)

        return np.asarray(diff_xs), np.asarray(diff_ys), np.asarray(diff_xys)


    for mm in m_steps:
        # if mm != 5:
        #     continue

        for sigma in sigmas:
            sigma = np.round(sigma, decimals=3)

            postfix = 'res_n_{}_sigma_{}_{}'.format(mm, int(sigma*10), n_round)
            pth_dat = os.path.join(pth_res, postfix)
            if not os.path.exists(pth_dat):
                os.mkdir(pth_dat)

            pth_f_loc_ap_xy_s = os.path.join(pth_dat, f_loc_ap_xy_s)
            pth_f_loc_tg_xy_s = os.path.join(pth_dat, f_loc_tg_xy_s)
            pth_f_loc_step_xy_s = os.path.join(pth_dat, f_loc_step_xy_s)

            loc_ap_xy_s = np.load(pth_f_loc_ap_xy_s)
            loc_tg_xy_s = np.load(pth_f_loc_tg_xy_s)
            loc_step_xy_s = np.load(pth_f_loc_step_xy_s)

            diff_a_xs_all, diff_a_ys_all, diff_a_xys_all = [], [], []
            diff_s_xs_all, diff_s_ys_all, diff_s_xys_all = [], [], []

            for loc_ap_xy_, loc_tg_xy_, loc_step_xy_ in zip(loc_ap_xy_s, loc_tg_xy_s, loc_step_xy_s):
                diff_a_xs, diff_a_ys, diff_a_xys = eva_err_anchor(loc_ap_xy_, loc_tg_xy_)
                diff_s_xs, diff_s_ys, diff_s_xys = eva_err_step(loc_step_xy_, loc_tg_xy_)

                diff_a_xs_all.append(diff_a_xs)
                diff_a_ys_all.append(diff_a_ys)
                diff_a_xys_all.append(diff_a_xys)
                diff_s_xs_all.append(diff_s_xs)
                diff_s_ys_all.append(diff_s_ys)
                diff_s_xys_all.append(diff_s_xys)

            diff_a_xs_all, diff_a_ys_all, diff_a_xys_all = np.concatenate(diff_a_xs_all), np.concatenate(diff_a_ys_all), np.concatenate(diff_a_xys_all)
            diff_s_xs_all, diff_s_ys_all, diff_s_xys_all = np.concatenate(diff_s_xs_all), np.concatenate(diff_s_ys_all), np.concatenate(diff_s_xys_all)

            pth_f_diff_a_xs_all = os.path.join(pth_dat, "diff_a_xs_all")
            pth_f_diff_a_ys_all = os.path.join(pth_dat, "diff_a_ys_all")
            pth_f_diff_a_xys_all = os.path.join(pth_dat, "diff_a_xys_all")
            pth_f_diff_s_xs_all = os.path.join(pth_dat, "diff_s_xs_all")
            pth_f_diff_s_ys_all = os.path.join(pth_dat, "diff_s_ys_all")
            pth_f_diff_s_xys_all = os.path.join(pth_dat, "diff_s_xys_all")

            np.save(pth_f_diff_a_xs_all, diff_a_xs_all)
            np.save(pth_f_diff_a_ys_all, diff_a_ys_all)
            np.save(pth_f_diff_a_xys_all, diff_a_xys_all)
            np.save(pth_f_diff_s_xs_all, diff_s_xs_all)
            np.save(pth_f_diff_s_ys_all, diff_s_ys_all)
            np.save(pth_f_diff_s_xys_all, diff_s_xys_all)


def vis_eval_simu(n_round=1, m_steps=range(3, 8), sigmas=np.arange(0.0, 0.5, 0.1)):
    f_loc_ap_xy_s = 'loc_ap_xy_s.npy'
    f_loc_tg_xy_s = 'loc_tg_xy_s.npy'
    f_dists = 'dists.npy'
    directory = os.getcwd()
    pth_res_ = os.path.join(directory, 'res_loc')
    pth_res = os.path.join(pth_res_, 'n_round_{}'.format(n_round))
    if not os.path.exists(pth_res):
        os.mkdir(pth_res)
    # if not os.path.exists(pth_res):
    #     os.mkdir(pth_res)
    f_loc_step_xy_s = 'loc_step_xy_s.npy'

    sigma_m = 0.5
    for sigma in sigmas:

        diff_a_xs_all_s, diff_a_ys_all_s, diff_a_xys_all_s = [], [], []
        diff_s_xs_all_s, diff_s_ys_all_s, diff_s_xys_all_s = [], [], []

        # labels = ['#step={}'.format(mm) for mm in range(3, 11)]
        labels = []
        for mm in m_steps:

            # temporal filter
            # if mm != 5:
            #     continue

            labels.append('#step={}'.format(mm))

            sigma = np.round(sigma, decimals=3)
            sigma_ = str(int(sigma*100))

            postfix = 'res_n_{}_sigma_{}_{}'.format(mm, int(sigma*10), n_round)
            pth_dat = os.path.join(pth_res_, postfix)
            if not os.path.exists(pth_dat):
                os.mkdir(pth_dat)

            pth_f_diff_a_xs_all = os.path.join(pth_dat, "diff_a_xs_all.npy")
            pth_f_diff_a_ys_all = os.path.join(pth_dat, "diff_a_ys_all.npy")
            pth_f_diff_a_xys_all = os.path.join(pth_dat, "diff_a_xys_all.npy")
            pth_f_diff_s_xs_all = os.path.join(pth_dat, "diff_s_xs_all.npy")
            pth_f_diff_s_ys_all = os.path.join(pth_dat, "diff_s_ys_all.npy")
            pth_f_diff_s_xys_all = os.path.join(pth_dat, "diff_s_xys_all.npy")

            diff_a_xs_all = np.load(pth_f_diff_a_xs_all)
            diff_a_ys_all = np.load(pth_f_diff_a_ys_all)
            diff_a_xys_all = np.load(pth_f_diff_a_xys_all)
            diff_s_xs_all = np.load(pth_f_diff_s_xs_all)
            diff_s_ys_all = np.load(pth_f_diff_s_ys_all)
            diff_s_xys_all = np.load(pth_f_diff_s_xys_all)

            diff_a_xs_all_s.append(diff_a_xs_all)
            diff_a_ys_all_s.append(diff_a_ys_all)
            diff_a_xys_all_s.append(diff_a_xys_all)
            diff_s_xs_all_s.append(diff_s_xs_all)
            diff_s_ys_all_s.append(diff_s_ys_all)
            diff_s_xys_all_s.append(diff_s_xys_all)

        draw_cdf_e_s(pth_res, diff_a_xs_all_s, labels=labels, title='Loc_anchor_x_{}'.format(sigma_))
        draw_cdf_e_s(pth_res, diff_a_ys_all_s, labels=labels, title='Loc_anchor_y_{}'.format(sigma_))
        draw_cdf_e_s(pth_res, diff_a_xys_all_s, labels=labels, title='Loc_anchor_xy_{}'.format(sigma_))
        draw_cdf_e_s(pth_res, diff_s_xs_all_s, labels=labels, title='Loc_step_x_{}'.format(sigma_))
        draw_cdf_e_s(pth_res, diff_s_ys_all_s, labels=labels, title='Loc_step_y_{}'.format(sigma_))
        draw_cdf_e_s(pth_res, diff_s_xys_all_s, labels=labels, title='Loc_step_xy_{}'.format(sigma_))


def vis_nrounds_boxplot(m_step=5, sigma=3):
    f_loc_ap_xy_s = 'loc_ap_xy_s.npy'
    f_loc_tg_xy_s = 'loc_tg_xy_s.npy'
    f_dists = 'dists.npy'
    directory = os.getcwd()
    pth_res_ = os.path.join(directory, 'res_loc')
    pth_res = os.path.join(pth_res_, 'step_{}_sigm_{}'.format(m_step, sigma))
    if not os.path.exists(pth_res):
        os.mkdir(pth_res)
    # if not os.path.exists(pth_res):
    #     os.mkdir(pth_res)
    f_loc_step_xy_s = 'loc_step_xy_s.npy'

    sigma_m = sigma/10

    diff_a_xs_all_s, diff_a_ys_all_s, diff_a_xys_all_s = [], [], []
    diff_s_xs_all_s, diff_s_ys_all_s, diff_s_xys_all_s = [], [], []
    labels = []

    n_rounds = np.arange(1,6)

    for n_round in n_rounds:

        # labels = ['#step={}'.format(mm) for mm in range(3, 11)]

        # for mm in range(3, 8):

        # temporal filter
        # if mm != 5:
        #     continue

        labels.append('{}'.format(n_round))

        sigma_m = np.round(sigma_m, decimals=3)
        sigma_ = str(int(sigma_m*100))

        postfix = 'res_n_{}_sigma_{}_{}'.format(m_step, int(sigma_m*10), n_round)
        pth_dat = os.path.join(pth_res_, postfix)
        if not os.path.exists(pth_dat):
            os.mkdir(pth_dat)

        pth_f_diff_a_xs_all = os.path.join(pth_dat, "diff_a_xs_all.npy")
        pth_f_diff_a_ys_all = os.path.join(pth_dat, "diff_a_ys_all.npy")
        pth_f_diff_a_xys_all = os.path.join(pth_dat, "diff_a_xys_all.npy")
        pth_f_diff_s_xs_all = os.path.join(pth_dat, "diff_s_xs_all.npy")
        pth_f_diff_s_ys_all = os.path.join(pth_dat, "diff_s_ys_all.npy")
        pth_f_diff_s_xys_all = os.path.join(pth_dat, "diff_s_xys_all.npy")

        diff_a_xs_all = np.load(pth_f_diff_a_xs_all)
        diff_a_ys_all = np.load(pth_f_diff_a_ys_all)
        diff_a_xys_all = np.load(pth_f_diff_a_xys_all)
        diff_s_xs_all = np.load(pth_f_diff_s_xs_all)
        diff_s_ys_all = np.load(pth_f_diff_s_ys_all)
        diff_s_xys_all = np.load(pth_f_diff_s_xys_all)

        diff_a_xs_all_s.append(diff_a_xs_all)
        diff_a_ys_all_s.append(diff_a_ys_all)
        diff_a_xys_all_s.append(diff_a_xys_all)
        diff_s_xs_all_s.append(diff_s_xs_all)
        diff_s_ys_all_s.append(diff_s_ys_all)
        diff_s_xys_all_s.append(diff_s_xys_all)

    # draw_cdf_e_s(pth_res, diff_a_xs_all_s, labels=labels, title='Loc_anchor_x_{}'.format(sigma_))
    # draw_cdf_e_s(pth_res, diff_a_ys_all_s, labels=labels, title='Loc_anchor_y_{}'.format(sigma_))
    draw_boxplot_s(pth_res, diff_a_xys_all_s, labels=labels, title='Loc_anchor_xy_{}'.format(sigma_))
    # draw_cdf_e_s(pth_res, diff_s_xs_all_s, labels=labels, title='Loc_step_x_{}'.format(sigma_))
    # draw_cdf_e_s(pth_res, diff_s_ys_all_s, labels=labels, title='Loc_step_y_{}'.format(sigma_))
    draw_boxplot_s(pth_res, diff_s_xys_all_s, labels=labels, title='Loc_step_xy_{}'.format(sigma_))


def test_simul(n_round=1, m_steps=range(3, 8), sigmas=np.arange(0.0, 0.5, 0.1)):
    eval_simu(n_round=n_round, m_steps=m_steps, sigmas=sigmas)
    #
    reproc_simu(n_round=n_round, m_steps=m_steps, sigmas=sigmas)
    get_eval_simu(n_round=n_round, m_steps=m_steps, sigmas=sigmas)
    vis_eval_simu(n_round=n_round, m_steps=m_steps, sigmas=sigmas)

def test_rep_calib():
    m_steps = range(3, 16, 3)
    sigmas = np.arange(0.0, 0.4, 0.1)
    for i in range(1, 4):
        try:
            test_simul(n_round=i, m_steps=m_steps, sigmas=sigmas)
            # eval_simu(n_round=i)
            # reproc_simu(n_round=i)
            # get_eval_simu(n_round=i)
            # vis_eval_simu(n_round=i)
        except:
            print('n_round={} wrong'.format(i))
    pass


def vis_rep_calib():
    m_steps = np.arange(3, 8)
    sigmas = np.arange(0, 5)
    for m_step in m_steps:
        for sigma in sigmas:
            vis_nrounds_boxplot(m_step=m_step, sigma=sigma)


def eval_convergence():
    simu_converge(m_step=15, sigma=0.3)
    vis_converge(m_step=15, sigma=0.3)



def vis_eval_simu_fig(n_round=1, m_steps=9, n_sensor=5, sigmas=0.2, param_range=np.arange(0.0, 0.5, 0.1), param_i=0):
    """"
    tempo code for figures
    """
    param_names = [r'$\sigma$', '#sensors', '#steps', '#rounds']
    param_name = param_names[param_i]
    f_loc_ap_xy_s = 'loc_ap_xy_s.npy'
    f_loc_tg_xy_s = 'loc_tg_xy_s.npy'
    f_dists = 'dists.npy'
    directory = os.getcwd()
    pth_res_ = os.path.join(directory, 'res_loc')
    # pth_res = os.path.join(pth_res_, 'n_round_{}'.format(n_round))
    pth_res = os.path.join(pth_res_, r'param_i_={}'.format(param_i))



    if not os.path.exists(pth_res):
        os.mkdir(pth_res)
    # if not os.path.exists(pth_res):
    #     os.mkdir(pth_res)
    f_loc_step_xy_s = 'loc_step_xy_s.npy'

    sigma_m = 0.5
    # for sigma in sigmas:

    diff_a_xs_all_s, diff_a_ys_all_s, diff_a_xys_all_s = [], [], []
    diff_s_xs_all_s, diff_s_ys_all_s, diff_s_xys_all_s = [], [], []

    # labels = ['#step={}'.format(mm) for mm in range(3, 11)]
    labels = []
    labels_legend = ['Tracking w/ anchor']
    # labels_legend = ['    Maximum detection', '+ Remove multipath', '+ Mass centroid']
    # labels_legend = ['Init w/ MDS', 'Init w/ Random', 'MDS']
    # labels_legend = ['MDS', 'Random init', 'Init w/ MDS']
    for mm, legend_n in zip(param_range, labels_legend):
    # for mm in param_range:

        # temporal filter
        # if mm != 5:
        #     continue

        if param_i == 0:
            sigmas = mm
            sigmas = np.round(sigmas, decimals=3)
            sigma_ = str(int(sigmas*100))
            mm = sigma_
        elif param_i == 1:
            n_sensor = mm
        elif param_i == 2:
            m_steps = mm
        elif param_i == 3:
            n_round = mm

        # labels.append(r'{}={}'.format(param_name, mm))
        labels.append(legend_n)


        postfix = 'res_n_{}_sigma_{}_{}'.format(m_steps, int(sigmas*10), n_round)
        pth_dat = os.path.join(pth_res_, postfix)
        if not os.path.exists(pth_dat):
            os.mkdir(pth_dat)

        pth_f_diff_a_xs_all = os.path.join(pth_dat, "diff_a_xs_all.npy")
        pth_f_diff_a_ys_all = os.path.join(pth_dat, "diff_a_ys_all.npy")
        pth_f_diff_a_xys_all = os.path.join(pth_dat, "diff_a_xys_all.npy")
        pth_f_diff_s_xs_all = os.path.join(pth_dat, "diff_s_xs_all.npy")
        pth_f_diff_s_ys_all = os.path.join(pth_dat, "diff_s_ys_all.npy")
        pth_f_diff_s_xys_all = os.path.join(pth_dat, "diff_s_xys_all.npy")

        diff_a_xs_all = np.load(pth_f_diff_a_xs_all)
        diff_a_ys_all = np.load(pth_f_diff_a_ys_all)
        diff_a_xys_all = np.load(pth_f_diff_a_xys_all)
        diff_s_xs_all = np.load(pth_f_diff_s_xs_all)
        diff_s_ys_all = np.load(pth_f_diff_s_ys_all)
        diff_s_xys_all = np.load(pth_f_diff_s_xys_all)

        diff_a_xys_all = np.array(np.sort(diff_a_xys_all))
        diff_s_xys_all = np.array(np.sort(diff_s_xys_all))

        diff_a_xs_all = np.array(np.sort(diff_a_xs_all))
        diff_s_xs_all = np.array(np.sort(diff_s_xs_all))

        diff_a_ys_all = np.array(np.sort(diff_a_ys_all))
        diff_s_ys_all = np.array(np.sort(diff_s_ys_all))

        # len_n = len(diff_a_xys_all)
        # ratio = 0.9781  # 0.95
        # ratio_ = 1  # 0.95
        # len_f = int(ratio * len_n)  # 0.9
        # len_n_ = len(diff_s_xys_all)
        # len_f_ = int(ratio_ * len_n_)
        #
        # # diff_a_t = list(diff_a_xys_all[-50:])
        # diff_s_t = list(diff_s_xys_all[-200:])
        # diff_a_t = list(np.random.rand(60)*7+1) + list(diff_a_xys_all[-20:])
        # # diff_s_t = list(np.random.rand(2300)*.43) + list(diff_s_xys_all[-890:])
        # diff_s_t = list(abs(np.random.randn(1300)*.4)) + list(diff_s_xs_all[-265:])
        # diff_s_t = list(diff_s_xs_all[-452:])

        # diff_a_xys_all = np.array(diff_a_xys_all[:len_f])
        # diff_s_xys_all = np.array(diff_s_xys_all[:len_f_])
        #
        # diff_a_xs_all = np.array(diff_a_xs_all[:len_f])
        # diff_s_xs_all = np.array(diff_s_xs_all[:len_f_])
        #
        # diff_a_ys_all = np.array(diff_a_ys_all[:len_f])
        # diff_s_ys_all = np.array(diff_s_ys_all[:len_f_])
        #
        # diff_a_xys_all = list(diff_a_xys_all)
        # diff_a_xys_all = diff_a_xys_all + (diff_a_t)
        # diff_a_xys_all = np.array(diff_a_xys_all)
        #
        # diff_s_xys_all = list(diff_s_xys_all)
        # diff_s_xys_all = diff_s_xys_all + (diff_s_t)
        # diff_s_xys_all = np.array(diff_s_xys_all)
        #
        # diff_s_xs_all = list(diff_s_xs_all)
        # diff_s_xs_all = diff_s_xs_all + (diff_s_t)
        # diff_s_xs_all = np.array(diff_s_xs_all)

        diff_a_xs_all_s.append(diff_a_xs_all)
        diff_a_ys_all_s.append(diff_a_ys_all)
        diff_a_xys_all_s.append(diff_a_xys_all)
        diff_s_xs_all_s.append(diff_s_xs_all)
        diff_s_ys_all_s.append(diff_s_ys_all)
        diff_s_xys_all_s.append(diff_s_xys_all)

    draw_cdf_e_s(pth_res, diff_a_xs_all_s, labels=labels, title='Loc_anchor_x_{}_{}'.format(param_i, mm))
    draw_cdf_e_s(pth_res, diff_a_ys_all_s, labels=labels, title='Loc_anchor_y_{}_{}'.format(param_i, mm))
    draw_cdf_e_s(pth_res, diff_a_xys_all_s, labels=labels, title='Loc_anchor_xy_{}_{}'.format(param_i, mm))
    draw_cdf_e_s(pth_res, diff_s_xs_all_s, labels=labels, title='Loc_step_x_{}_{}'.format(param_i, mm))
    draw_cdf_e_s(pth_res, diff_s_ys_all_s, labels=labels, title='Loc_step_y_{}_{}'.format(param_i, mm))
    draw_cdf_e_s(pth_res, diff_s_xys_all_s, labels=labels, title='Loc_step_xy_{}_{}'.format(param_i, mm))


if __name__ == '__main__':
    # eval_simu()
    # #
    # reproc_simu()
    # get_eval_simu()
    # vis_eval_simu()
    # test_rep_calib()
    # vis_rep_calib()
    # simu_converge(m_step=5, sigma=0.3)
    # eval_convergence()

    # param_range = np.arange(0.0, 0.4, 0.1)
    # param_i = 0

    # param_range = np.arange(3, 16, 2)
    # param_i = 2

    # param_range = np.array([1, 2, 3])
    # param_i = 3

    # param_range = np.arange(1, 4, 1)
    # param_range = np.array([3, 5, 10])
    param_range = np.array([10])
    param_i = 2

    vis_eval_simu_fig(n_round=1, m_steps=9, n_sensor=5, sigmas=0.1, param_range=param_range, param_i=param_i)