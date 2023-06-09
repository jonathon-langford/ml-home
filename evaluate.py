import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplhep

import warnings
warnings.simplefilter(action='ignore')

from python.evaluation_utils import *

from optparse import OptionParser
def get_options():
    parser = OptionParser()
    parser.add_option('-i','--input-file', dest='input_file', default=None)
    parser.add_option('--do-input-features', dest='do_input_features', default=False, action="store_true")
    parser.add_option('--do-ml-output', dest='do_ml_output', default=False, action="store_true")
    parser.add_option('--do-roc', dest='do_roc', default=False, action="store_true")
    parser.add_option('--do-confusion-matrix', dest='do_confusion_matrix', default=False, action="store_true")
    return parser.parse_args()
(opt,args) = get_options()

# Plotting options
plot_map = {
    0:["Bkg","black","y_pred_bkg"],
    1:["ggH (125)","cornflowerblue","y_pred_ggH"],
    2:["VBF (125)","darkorange","y_pred_VBF"]
}

fig, ax = plt.subplots()

train_vars = ['diphotonPt', 'dijetMass', 'dijetCentrality', 'leadPhotonPtOvM', 'subleadPhotonPtOvM', 'leadPhotonEta', 'leadPhotonIDMVA', 'subleadPhotonEta', 'subleadPhotonIDMVA', 'leadJetPt', 'leadJetEta', 'subleadJetPt', 'subleadJetEta', 'dijetAbsDEta']

# Merge parquet files
df = pd.read_parquet(opt.input_file, engine="pyarrow")

# Make plots of input features
if opt.do_input_features:
    for var in train_vars:
        hist_feature(fig, ax, df, var, plot_map=plot_map, plot_path="plots/input_features", weight="weight", test_train_split=False)

# Make plots of probabilities
if opt.do_ml_output:
    for var in ['y_pred_ggH', 'y_pred_VBF', 'y_pred_bkg']:
        hist_feature(fig, ax, df, var, plot_map=plot_map, plot_path="plots/ml_output", weight="weight", x_range=(0,1), test_train_split=False)
        hist_feature(fig, ax, df, var, plot_map=plot_map, plot_path="plots/ml_output", weight="weight", x_range=(0,1), test_train_split=True)

if opt.do_roc:
    ROCs = {}
    ROCs['ggH'] = roc_three_class( fig, ax, df, signal_proc_id=1, var="y_pred_ggH", plot_map=plot_map, plot_path="plots/roc_curves")
    ROCs['VBF'] = roc_three_class( fig, ax, df, signal_proc_id=2, var="y_pred_VBF", plot_map=plot_map, plot_path="plots/roc_curves")
    ROCs['bkg'] = roc_three_class( fig, ax, df, signal_proc_id=0, var="y_pred_bkg", plot_map=plot_map, plot_path="plots/roc_curves")

if opt.do_confusion_matrix:
    # Only plot for test data
    mask = df['dataset_type'] == 'test'
    conf_matrix, conf_matrix_labels = confusion_matrix(fig, ax, df[mask], plot_map, plot_path="plots/confusion_matrix" )
    conf_matrix, conf_matrix_labels = confusion_matrix(fig, ax, df[mask], plot_map, plot_path="plots/confusion_matrix", norm_by="pred" )
    conf_matrix, conf_matrix_labels = confusion_matrix(fig, ax, df[mask], plot_map, plot_path="plots/confusion_matrix", norm_by="pred" , mass_cut=True)


