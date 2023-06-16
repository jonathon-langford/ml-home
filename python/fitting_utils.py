import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.style.use("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
from matplotlib.lines import Line2D

def sigmoid(x, scale=100):
    return 1/(1+np.exp(-scale*x))

def phi(x, mean=1, sigma=0.01):
    return norm.pdf(x, loc=mean, scale=sigma)/norm.pdf(mean, loc=mean, scale=sigma)

def calc_nll( data, proc_id_pred_var="proc_id_pred", signal_ids=[1,2], background_ids=[0], fit_var="diphotonMass", x_range=(123,127), n_bins=1, lumi_scale=137000, mu_points=100, mu_range=(-0.5,2.5)):
    
    sig_hists, bkg_hists, data_hists = {}, {}, {}

    # Loop over "signal" categories i.e. pred argmax in signal_ids
    for cat in signal_ids:
        sig_hists[cat] = {}
        # Loop over signal processes i.e. true proc id in signal_ids and add hist 
        for proc in signal_ids:
            mask = ( data[proc_id_pred_var] == cat )&( data['proc_id'] == proc )
            sig_hists[cat][proc] = np.histogram( data[mask][fit_var], bins=n_bins, range=x_range, weights=data[mask]['weight']*lumi_scale )[0]

        # Make bkg hists
        mask = data['proc_id'] == background_ids[0]
        if len(background_ids) > 1:
            for proc in background_ids[1:]: mask = mask | ( data['proc_id'] == proc )
        mask = mask & ( data[proc_id_pred_var] == cat )
        bkg_hists[cat] = np.histogram( data[mask][fit_var], bins=n_bins, range=x_range, weights=data[mask]['weight']*lumi_scale )[0]

        # Make data hist
        data_hists[cat] = bkg_hists[cat]
        for proc in signal_ids: data_hists[cat] = data_hists[cat] + sig_hists[cat][proc]


    # Make signal strength parameters for different signal processes
    mu = []
    for i in signal_ids: mu.append( np.linspace(mu_range[0],mu_range[1],mu_points) )
    mu_grid = np.meshgrid(*mu) 

    # Build array of NLL values for different signal strengths
    nll = np.zeros_like( mu_grid[0] )
    # Calculate the nll
    for cat in signal_ids:
        sig_hist_total = np.zeros_like( mu_grid[0] )
        for i, proc in enumerate(signal_ids): sig_hist_total += sig_hists[cat][proc]*mu_grid[i]
        nll = nll - data_hists[cat]*np.log( sig_hist_total + bkg_hists[cat] ) + (sig_hist_total + bkg_hists[cat])

    # Reformulate as 2*deltaNLL
    q = 2*(nll-nll.min())

    return mu_grid, q


def calc_fisher( data, proc_id_pred_var="proc_id_pred", signal_ids=[1,2], background_ids=[0], fit_var="diphotonMass", x_range=(123,127), n_bins=1, lumi_scale=137000 ):

    sig_hists, bkg_hists, data_hists = {}, {}, {}

    # Loop over "signal" categories i.e. pred argmax in signal_ids
    for cat in signal_ids:
        sig_hists[cat] = {}
        # Loop over signal processes i.e. true proc id in signal_ids and add hist 
        for proc in signal_ids:
            mask = ( data[proc_id_pred_var] == cat )&( data['proc_id'] == proc )
            sig_hists[cat][proc] = np.histogram( data[mask][fit_var], bins=n_bins, range=x_range, weights=data[mask]['weight']*lumi_scale )[0]

        # Make bkg hists
        mask = data['proc_id'] == background_ids[0]
        if len(background_ids) > 1:
            for proc in background_ids[1:]: mask = mask | ( data['proc_id'] == proc )
        mask = mask & ( data[proc_id_pred_var] == cat )
        bkg_hists[cat] = np.histogram( data[mask][fit_var], bins=n_bins, range=x_range, weights=data[mask]['weight']*lumi_scale )[0]

        # Make data hist
        data_hists[cat] = bkg_hists[cat]
        for proc in signal_ids: data_hists[cat] = data_hists[cat] + sig_hists[cat][proc]

    fisher = np.zeros( (len(signal_ids), len(signal_ids)) )
    for i, proc_i in enumerate(signal_ids):
        for k, proc_k in enumerate(signal_ids):
            f = 0
            for j, cat in enumerate(signal_ids):
                f += (sig_hists[cat][proc_i]*sig_hists[cat][proc_k])/data_hists[cat] 
            fisher[i][k] = f

    cov = linalg.inv(fisher)
    unc = []
    for i in range(len(signal_ids)):
        unc.append( np.sqrt(cov[i][i]) )
    unc = np.array(unc)

    corr = np.divide(cov, np.outer(unc,unc))

    return fisher, (cov, unc, corr)


def extract_fisher_info(F):

    detF = linalg.det(F)
    traceF = np.diag(F).sum()
    prodF = np.prod(np.diag(F))

    V = linalg.inv(F)    
    detV = linalg.det(V)
    traceV = np.diag(V).sum()

    u = np.diag(V)**0.5
    prodU = np.prod(u)
    
    # Area of ellipse
    A = np.real(np.pi*np.prod(linalg.eig(V)[0]))

    return np.array([detF,traceF,prodF,detV,traceV,prodU,A])


def prepare_fit_inputs( data, signal_ids=[1,2], background_ids=[0], fit_var="diphotonMass", x_range=(123,127), n_bins=1, lumi_scale=137000, plot_map={} ):

    # Only use in signal region
    mask_x = (data[fit_var]>x_range[0])&(data[fit_var]<x_range[1])
    data = data[mask_x]

    score_vars = []
    i_score_var = 0
    label_map = {}
    for i in background_ids:
        score_vars.append( plot_map[i][2] )
        label_map[i] = i_score_var
        i_score_var += 1
    for i in signal_ids:
        score_vars.append( plot_map[i][2] )
        label_map[i] = i_score_var
        i_score_var += 1

    score = np.array( data[score_vars] )
    x = np.array( data[fit_var] )
    truth_label = np.array([label_map[j] for j in np.array(data['proc_id'])])
    event_weights = np.array(data['weight'])*lumi_scale

    return score, truth_label, event_weights, label_map


def calc_fisher_metric_differentiable( W, score, truth_label, event_weights, metric="detF", signal_ids=[1,2], background_ids=[0] ):

    # TODO: make compatible with histogram, so far considers only one bin

    weight_array = np.ones(len(background_ids))
    weight_array = np.concatenate([weight_array,W])

    score_w = (score*weight_array).T

    n_cats = len(background_ids)+len(signal_ids)

    # Build predicted label array
    pred_label_array = []
    for k in range(n_cats):
        theta = np.ones_like(truth_label, dtype='float')
        for l in range(n_cats):
            if k == l: continue
            theta *= sigmoid(score_w[k]-score_w[l])
        pred_label_array.append(theta)
    pred_label_array = np.array( pred_label_array )

    # Array of sigmoid multiplied by event weights: only consider signal-like categories
    theta_weights = pred_label_array[len(background_ids):]*event_weights

    # Construct Fisher: using truth slicing
    #theta_weights = theta_weights.T
    #F = np.zeros((len(sig_ids),len(sig_ids)))
    #for j, proc_j in enumerate(sig_ids):
    #    for k, proc_k in enumerate(sig_ids):
    #        F[j][k] = ((theta_weights[truth_label==proc_j].T.sum(axis=1)*theta_weights[truth_label==proc_k].T.sum(axis=1))/theta_weights.T.sum(axis=1)).sum()

    # Fisher matrix
    F = np.zeros( (len(signal_ids),len(signal_ids)) )
    for i, proc_i in enumerate(signal_ids):
        for j, proc_j in enumerate(signal_ids):
            F[i][j] = ((theta_weights*phi(truth_label,proc_i)).sum(axis=1)*(theta_weights*phi(truth_label,proc_j)).sum(axis=1)/theta_weights.sum(axis=1)).sum()

    if metric == "matrix": return F
    elif metric == "ndetF": return -1*linalg.det(F)
    elif metric == "nlogdetF": return -1*np.log(linalg.det(F))
    elif metric == "ntraceF": -1*np.diag(F).sum()
    elif metric == "nprodF": -1*np.prod(np.diag(F))
    elif metric == "detV": return linalg.det(linalg.inv(F))
    elif metric == "traceV": return np.diag(linalg.inv(F)).sum()
    elif metric == "prodU": return np.prod(np.diag(linalg.inv(F))**0.5)
    elif metric == "area": return np.real(np.pi*np.prod(linalg.eig(linalg.inv(F)[0])))
    else:
        print("Metric not recognised. Returning -1")
        return -1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting scripts for fits    

def plot_2d_nll( fig, ax, q, mu_grid, signal_ids=[1,2], plot_map={}, vmax=9, plot_path=".", ext="", do_1d_levels=False, plot_surface=True, q_labels=[] ):
    extent = (mu_grid[0].min(), mu_grid[0].max(), mu_grid[1].min(), mu_grid[1].max())
    plt.set_cmap("Blues_r")

    colors = ['k','r','royalblue']

    for i, qi in enumerate(q):
        if plot_surface: mat = ax.imshow(qi, extent=extent, vmax=vmax, origin='lower')
        if do_1d_levels:
            c1 = ax.contour(mu_grid[0], mu_grid[1], qi, levels=[1], colors=colors[i])
            c2 = ax.contour(mu_grid[0], mu_grid[1], qi, levels=[4], colors=colors[i], linestyles='dashed')
        else:
            c1 = ax.contour(mu_grid[0], mu_grid[1], qi, levels=[2.3], colors=colors[i])
            c2 = ax.contour(mu_grid[0], mu_grid[1], qi, levels=[5.99], colors=colors[i], linestyles='dashed')

    sm = ax.scatter(1, 1, marker='+', c='red', label='SM', s=50)
    if plot_surface: cbar = fig.colorbar(mat)

    ax.set_xlabel("$\\mu_{%s}$"%plot_map[signal_ids[0]][0].split(" ")[0])
    ax.set_ylabel("$\\mu_{%s}$"%plot_map[signal_ids[1]][0].split(" ")[0])

    if len(q_labels) > 0:
        custom_lines = []
        for i in range(len(q_labels)): custom_lines.append( Line2D([0],[0], color=colors[i], lw=2) )
        ax.legend(custom_lines, q_labels, loc='best')

    ext_str = "_%s"%ext if ext != "" else ""
    fig.savefig("%s/nll_surface%s.png"%(plot_path,ext_str))

    ax.cla()
    if plot_surface: cbar.remove()

def plot_2d_surface( fig, ax, z, grid, z_name="Z", plot_path=".", ext="", plot_extreme="max"):
    extent = (grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max())

    if plot_extreme == "max":
        plt.set_cmap("Reds")
        mat = ax.imshow(z, extent=extent, origin='lower')
    elif plot_extreme == "min":
        plt.set_cmap("Reds_r")
        mat = ax.imshow(z, extent=extent, origin='lower', vmax=np.percentile(z,95))
    else:
        plt.set_cmap("hot")
        mat = ax.imshow(z, extent=extent, origin='lower')

    # Find extreme
    if plot_extreme == "max":
        max_index = np.unravel_index(z.argmax(), z.shape)
        x,y = grid[0][max_index], grid[1][max_index]
        ax.scatter(x, y, marker='+', c='cornflowerblue', label='Maximum: (%.3f,%.3f)'%(x,y), s=100)

    elif plot_extreme == "min":
        plt.set_cmap("Reds_r")
        min_index = np.unravel_index(z.argmin(), z.shape)
        x,y = grid[0][min_index], grid[1][min_index]
        ax.scatter(x, y, marker='+', c='cornflowerblue', label='Minimum: (%.3f,%.3f)'%(x,y), s=100)

    ax.set_xlabel("$w_{ggH}$")
    ax.set_ylabel("$w_{VBF}$")

    #ax.scatter(1, 1, marker='+', c='k', label='Default: (1,1)', s=100)

    ax.legend(loc='best')

    cbar = fig.colorbar(mat)
    cbar.set_label(z_name)

    ext_str = "_%s"%ext if ext != "" else ""

    fig.savefig("%s/surface%s.png"%(plot_path,ext_str))

    ax.cla()
    cbar.remove()
