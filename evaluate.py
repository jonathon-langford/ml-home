import pandas as pd
import numpy as np
from scipy.optimize import minimize

import pickle as pkl

import matplotlib.pyplot as plt
import mplhep

import warnings
warnings.simplefilter(action='ignore')

from python.evaluation_utils import *
from python.fitting_utils import *

from optparse import OptionParser
def get_options():
    parser = OptionParser()
    parser.add_option('-i','--input-file', dest='input_file', default=None)
    parser.add_option('--do-input-features', dest='do_input_features', default=False, action="store_true")
    parser.add_option('--do-ml-output', dest='do_ml_output', default=False, action="store_true")
    parser.add_option('--do-roc', dest='do_roc', default=False, action="store_true")
    parser.add_option('--do-confusion-matrix', dest='do_confusion_matrix', default=False, action="store_true")
    parser.add_option('--do-likelihood', dest='do_likelihood', default=False, action="store_true")
    parser.add_option('--optimise-weights', dest='optimise_weights', default=None, help="Use metric to optimise weight array")
    return parser.parse_args()
(opt,args) = get_options()

# Plotting options
bkg_ids = [0]
sig_ids = [1,2,3,4,5]

plot_map = {
    0:["Bkg","black","y_pred_bkg"],
    1:["ggH","cornflowerblue","y_pred_ggH",(0,2,40)],
    2:["VBF","darkorange","y_pred_VBF",(-1,3,40)],
    3:["VH","forestgreen","y_pred_VH",(-3,5,40)],
    4:["ttH","orchid","y_pred_ttH",(-3,5,40)],
    5:["tH","gold","y_pred_tH",(-20,20,40)]
}

fig, ax = plt.subplots()

train_vars = [
    'leadPhotonPt', 'leadPhotonEta', 'leadPhotonIDMVA',
    'subleadPhotonPt', 'subleadPhotonEta', 'subleadPhotonIDMVA',
    'leadJetPt', 'leadJetEta', 'leadJetQGL', 'leadJetBTagScore', 'leadJetDiphoDPhi', 'leadJetDiphoDEta',
    'subleadJetPt', 'subleadJetEta', 'subleadJetQGL', 'subleadJetBTagScore', 'subleadJetDiphoDPhi', 'subleadJetDiphoDEta',
    'subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetQGL', 'subsubleadJetBTagScore',
    'leadElectronPt', 'leadElectronEta',
    'subleadElectronPt', 'subleadElectronEta',
    'leadMuonPt', 'leadMuonEta',
    'subleadMuonPt', 'subleadMuonEta',
    'dijetMass', 'dijetPt', 'dijetEta', 'dijetAbsDEta', 'dijetDPhi', 'dijetMinDRJetPho', 'dijetCentrality', 'dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'nSoftJets',
    'metPt', 'metPhi', 'metSumET', 'metSignificance'
]

# Merge parquet files
df = pd.read_parquet(opt.input_file, engine="pyarrow")

# Make plots of input features
if opt.do_input_features:
    for var in train_vars:
        hist_feature(fig, ax, df, var, plot_map=plot_map, plot_path="plots/input_features", weight="weight", test_train_split=False)

# Make plots of probabilities
if opt.do_ml_output:
    for k,v in plot_map.items():
        var = v[2]
        hist_feature(fig, ax, df, var, plot_map=plot_map, plot_path="plots/ml_output", weight="weight", x_range=(0,1), test_train_split=False)
        hist_feature(fig, ax, df, var, plot_map=plot_map, plot_path="plots/ml_output", weight="weight", x_range=(0,1), test_train_split=True)

if opt.do_roc:
    ROCs = {}
    ROCs['ggH'] = roc_three_class( fig, ax, df, signal_proc_id=1, var="y_pred_ggH", plot_map=plot_map, plot_path="plots/roc_curves")
    ROCs['VBF'] = roc_three_class( fig, ax, df, signal_proc_id=2, var="y_pred_VBF", plot_map=plot_map, plot_path="plots/roc_curves")
    ROCs['VH'] = roc_three_class( fig, ax, df, signal_proc_id=3, var="y_pred_VH", plot_map=plot_map, plot_path="plots/roc_curves")
    ROCs['ttH'] = roc_three_class( fig, ax, df, signal_proc_id=4, var="y_pred_ttH", plot_map=plot_map, plot_path="plots/roc_curves")
    ROCs['tH'] = roc_three_class( fig, ax, df, signal_proc_id=5, var="y_pred_tH", plot_map=plot_map, plot_path="plots/roc_curves")
    ROCs['bkg'] = roc_three_class( fig, ax, df, signal_proc_id=0, var="y_pred_bkg", plot_map=plot_map, plot_path="plots/roc_curves")

if opt.do_confusion_matrix:
    # Only plot for test data
    mask = df['dataset_type'] == 'test'
    conf_matrix, conf_matrix_labels = confusion_matrix(fig, ax, df[mask], plot_map, plot_path="plots/confusion_matrix" )
    conf_matrix, conf_matrix_labels = confusion_matrix(fig, ax, df[mask], plot_map, plot_path="plots/confusion_matrix", norm_by="pred" )
    conf_matrix, conf_matrix_labels = confusion_matrix(fig, ax, df[mask], plot_map, plot_path="plots/confusion_matrix", norm_by="pred" , mass_cut=True)

if opt.do_likelihood:
    df = add_proc_id_pred(df, plot_map)
    S,B,D = prepare_nll_inputs(df, signal_ids=sig_ids, background_ids=bkg_ids)
    
    Q_fixed, Q_prof, X_prof = {}, {}, {}
    for sid in sig_ids:
        sig_ids_prof = [i for i in sig_ids if i!=sid]
        sig_ids_fixed = [sid]
    
        print(" --> Fitting for: %s"%plot_map[sid][0])
    
        mu_fixed = np.linspace(*plot_map[sid][3])
        mu_prof = np.ones(len(sig_ids)-1)
        mu_prof_update = np.ones(len(sig_ids)-1)
    
        nll, nll_prof, x_prof = [], [], []
        for mu in mu_fixed:
            nll.append( calc_nll_prof(mu_prof, [mu], S, B, D, signal_ids_prof=sig_ids_prof, signal_ids_fixed=sig_ids_fixed) )
            res = minimize( calc_nll_prof, mu_prof_update, args=([mu],S,B,D,sig_ids_prof,sig_ids_fixed) )
            nll_prof.append(res.fun)
            x_prof.append(res.x)
            mu_prof_update = np.array(res.x)
        nll = np.array(nll)
        nll_prof = np.array(nll_prof)
        q = 2*(nll-nll.min())
        q_prof = 2*(nll_prof-nll_prof.min())
    
        Q_fixed[plot_map[sid][0]] = q
        Q_prof[plot_map[sid][0]] = q_prof
        X_prof[plot_map[sid][0]] = x_prof
    
    # FIXME: add this to a function in evaluation_utils
    # Make plots
    fig, (ax0,ax1) = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
    
    for sid in sig_ids:
        sig_ids_prof = [i for i in sig_ids if i!=sid]
        mu_fixed = np.linspace(*plot_map[sid][3])
        ax0.plot(mu_fixed, Q_fixed[plot_map[sid][0]], c=plot_map[sid][1], ls='--', label='Fixed other to SM')
        ax0.plot(mu_fixed, Q_prof[plot_map[sid][0]], c=plot_map[sid][1], label='Profiled') 
    
        ax0.set_ylabel("2NLL")
        ax0.legend(loc='best')
    
        ax0.axhline(1, c='grey', ls='--')
        ax0.axhline(4, c='grey', ls='--')
        ax0.axvline(1, c='grey', ls='--')
    
        ax0.set_xlim(mu_fixed.min(),mu_fixed.max())
        ax0.set_ylim(0,10)
    
        for i, sid_prof in enumerate(sig_ids_prof):
            mu = np.array(X_prof[plot_map[sid][0]]).T[i]
            ax1.plot(mu_fixed, mu, c=plot_map[sid_prof][1], label=plot_map[sid_prof][0])
        
        ax1.set_xlabel("$\\mu_{%s}$"%plot_map[sid][0])
        ax1.set_xlim(mu_fixed.min(),mu_fixed.max())
        ax1.set_ylim(-1,3)
        ax1.legend(loc='best')
    
        ax1.axhline(1, c='grey', ls='--')
        ax1.axvline(1, c='grey', ls='--')
    
    
        fig.savefig("plots/likelihood_scan/nll_%s.png"%plot_map[sid][0])
    
        ax0.cla()
        ax1.cla()

# Make fisher information matrix
df = add_proc_id_pred(df, plot_map)
F, res = calc_fisher(df, signal_ids=sig_ids)

# Calculate fisher matrix in a differentiable way
score, truth_label, event_weights, label_map = prepare_fit_inputs(df, signal_ids=sig_ids, plot_map=plot_map)
W = np.ones(len(sig_ids))
F_diff = calc_fisher_differentiable(W, score, truth_label, event_weights, signal_ids=sig_ids)

# Run optimiser code to find best collection of weights
from timeit import default_timer as timer
W_bounds = []
W = np.array([0.8,0.1,0.1,0.1,0.05])
for i in sig_ids: W_bounds.append((0.025,4))
print(" --> Optimising the per-pred label weights...")
start = timer()
res = minimize( calc_fisher_differentiable, W, args=(score,truth_label,event_weights,"nlogdetF",[1,2,3,4,5]), bounds=W_bounds, method='TNC' ) 
end = timer()
print(" --> Finished optimising weights. Time taken: %.3f s"%(end-start))
print(res)

#TODO: check mass distributions after weighting, are we just in region of zero counts


exit(0)

# Build fit inputs
df = add_proc_id_pred(df, plot_map)
mu_grid, q = calc_nll(df)

weight_array = np.array([1,0.167,0.113])
df = add_proc_id_pred(df, plot_map, weight_array=weight_array)
mu_grid, qopt = calc_nll(df)

plot_2d_nll( fig, ax, [q,qopt], mu_grid, plot_map=plot_map, plot_path="plots/nll_opt", ext="opt_by_detF", plot_surface=False, q_labels=["Default argmax: (1,1,1)","det(I) optimised: (1,%.3f,%.3f)"%(weight_array[1],weight_array[2])])


score, truth_label, event_weights, label_map = prepare_fit_inputs( df, plot_map=plot_map )
W_default = np.ones(len(sig_ids))
F = calc_fisher_metric_differentiable(W_default, score, truth_label, event_weights, metric="matrix")

W_init = 0.2*np.ones(len(sig_ids))
W_bounds = []
for i in sig_ids: W_bounds.append((0.05,0.4))

print(" --> Running optimizer:")
res = minimize( calc_fisher_metric_differentiable, W_init, args=(score,truth_label,event_weights,"nlogdetF"), bounds=W_bounds, method='TNC')

# Plot likelihood surface with default and optimized weights
# First get nominal
df = add_proc_id_pred(df, plot_map)
mu_grid, q = calc_nll(df)

weight_array = np.ones(len(bkg_ids))
weight_array = np.concatenate([weight_array,res.x])
df = add_proc_id_pred(df, plot_map, weight_array=weight_array)
mu_grid, qopt = calc_nll(df)

plot_2d_nll( fig, ax, [q,qopt], mu_grid, plot_map=plot_map, plot_path="plots/nll_opt", ext="opt_by_nlogdetF", plot_surface=False, q_labels=["Default argmax: (1,1,1)","-ln(det(I)) optimised: (1,%.3f,%.3f)"%(weight_array[1],weight_array[2])])







