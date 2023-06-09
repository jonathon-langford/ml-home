import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.style.use("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

# Function to plot numbers on matrix
def plot_numbers_on_matrix(ax,mat,fontsize=16):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            c = mat[j,i]
            ax.text(i,j,'{:.2f}'.format(c),fontdict={'size': fontsize},va='center',ha='center')

# Plot feature in dataframe
def hist_feature( fig, ax, data, var, nbins=40, x_range=None, plot_map={}, plot_path="./plots", weight=None, density=True, test_train_split=True ):

    weights = np.ones_like(data['proc_id']) if weight is None else data[weight]

    if x_range is None:
        x_range = (data[var].min(),np.percentile(data[var], 98))

    for proc_id in np.unique( data['proc_id'] ):

        mask = data['proc_id']==proc_id

        if test_train_split:

            mask_train = mask &( data['dataset_type']=='train' )
            mask_test = mask &( data['dataset_type']=='test' )

            if density:
                w = weights[mask_train]*(1./weights[mask_train].sum())
            else:
                w = weights[mask_train]

            ax.hist( data[mask_train][var], bins=nbins, range=x_range, label=plot_map[proc_id][0]+" (train)", color=plot_map[proc_id][1], histtype="step", weights=w, ls='--', linewidth=2 )

            if density:
                w = weights[mask_test]*(1./weights[mask_test].sum())
            else:
                w = weights[mask_test]

            mask_test = mask &( data['dataset_type']=='test' )
            ax.hist( data[mask_test][var], bins=nbins, range=x_range, label=plot_map[proc_id][0]+" (test)", color=plot_map[proc_id][1], histtype="step", weights=w, linewidth=2 )

        else:

            if density:
                w = weights[mask]*(1./weights[mask].sum())
            else:
                w = weights[mask]

            ax.hist( data[mask][var], bins=nbins, range=x_range, label=plot_map[proc_id][0], color=plot_map[proc_id][1], histtype="step", weights=w, linewidth=2 )

    ax.set_xlim(x_range)
    ax.set_xlabel(var)
    if density: ax.set_ylabel("Fraction of events")
    else: ax.set_ylabel("Events")
    ax.legend(loc='best')

    ext_str = "_testtrainsplit" if test_train_split else ""
    fig.savefig("%s/%s%s.png"%(plot_path, var, ext_str))

    ax.cla()





# ROC curve for three-class
def roc_three_class( fig, ax, data, signal_proc_id=None, var=None, plot_map={}, plot_path="./plots", test_train_split=True):
    
    # Extract full set of proc_ids 
    proc_ids = np.unique(data['proc_id'])
    other_ids = []
    for proc_id in proc_ids:
        if proc_id != signal_proc_id: other_ids.append( proc_id )

    # Loop over other_ids, build roc objects and plot on axes
    rocs = {}
    for other_proc_id in other_ids:

        data_sig_vs_other = data[( data['proc_id']==signal_proc_id )|( data['proc_id']==other_proc_id )]

        y_true = data_sig_vs_other['proc_id'] == signal_proc_id
        y_pred = data_sig_vs_other[var]
        w = data_sig_vs_other['weight']

        if test_train_split:
            mask_train = data_sig_vs_other['dataset_type'] == 'train'
            rocs[(other_proc_id, 'train')] = roc_curve( y_true[mask_train], y_pred[mask_train], sample_weight=w[mask_train] )
            ax.plot( rocs[(other_proc_id, 'train')][1], rocs[(other_proc_id, 'train')][0], label=plot_map[other_proc_id][0]+" (train)", color=plot_map[other_proc_id][1], ls='--' )

            mask_test = data_sig_vs_other['dataset_type'] == 'test'
            rocs[(other_proc_id, 'test')] = roc_curve( y_true[mask_test], y_pred[mask_test], sample_weight=w[mask_test] )
            ax.plot( rocs[(other_proc_id, 'test')][1], rocs[(other_proc_id, 'test')][0], label=plot_map[other_proc_id][0]+" (test)", color=plot_map[other_proc_id][1])

        else:
            rocs[(other_proc_id, 'all')] = roc_curve( y_true, y_pred, w )
            ax.plot( rocs[(other_proc_id, 'all')][1], rocs[(other_proc_id, 'all')][0], label=plot_map[other_proc_id][0], color=plot_map[other_proc_id][1] )

    # Plot ROC curve on axis
    ax.set_xlabel("Signal efficiency (%s)"%plot_map[signal_proc_id][0].split(" ")[0])
    ax.set_ylabel("Background efficiency")
    ax.legend(loc='best')

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid()

    ext_str = "" if test_train_split else "_alldata"
    fig.savefig("%s/roc_%s%s.png"%(plot_path,var,ext_str))

    ax.cla()

    return rocs


# Confusion matrix: normalised by truth or predicted label
def confusion_matrix( fig, ax, data, plot_map, plot_path="./plots", norm_by="truth", ext=None, mass_cut=False ):

    if norm_by == "truth": plt.set_cmap("Blues")
    elif norm_by == "pred": plt.set_cmap("Greens")
    else: plt.set_cmap("Purples")

    if mass_cut:
        mask = ( data['diphotonMass'] > 123. )&( data['diphotonMass'] < 127. )
        data = data[mask]

    # Get argmax of predicted probabilities to assign class
    # Require mapping to proc_id label in case of different ordering in plot_map
    pred_vars, pred_var_map = [], {}
    i_pred_var = 0
    for k, v in plot_map.items():
        pred_vars.append( data[v[2]] )
        pred_var_map[i_pred_var] = k
        i_pred_var += 1
    data['proc_id_pred'] = np.array([pred_var_map[j] for j in np.argmax( np.array(pred_vars).T, axis=1 )])

    n_classes = len( np.unique(data['proc_id']) )
    conf_matrix = np.zeros((n_classes,n_classes))
    conf_matrix_labels = []
    axis_labels = []

    for i, proc_id_true in enumerate( np.unique(data['proc_id']) ):
        conf_matrix_labels.append([])
        axis_labels.append( plot_map[proc_id_true][0].split(" ")[0] )

        for j, proc_id_pred in enumerate( np.unique(data['proc_id']) ):
            mask = (data['proc_id']==proc_id_true)&(data['proc_id_pred']==proc_id_pred)
            conf_matrix[i][j] += data[mask]['weight'].sum()
            conf_matrix_labels[i].append( "true_%s__pred_%s"%(plot_map[proc_id_true][2],plot_map[proc_id_pred][2]) )

    if norm_by == "truth":
        conf_matrix = (conf_matrix.T/conf_matrix.sum(axis=1)).T
    elif norm_by == "pred":
        conf_matrix = conf_matrix/conf_matrix.T.sum(axis=1) 

    if norm_by in ['truth','pred']:
        mat = ax.matshow( conf_matrix, vmin=0, vmax=1 )
    else:
        mat = ax.matshow( conf_matrix )

    plot_numbers_on_matrix(ax, conf_matrix)
    ax.set_xticks( np.arange( len( axis_labels) ) )
    ax.set_yticks( np.arange( len( axis_labels) ) )
    ax.set_xticklabels( axis_labels, fontsize=16 )
    ax.set_yticklabels( axis_labels, fontsize=16 )
    ax.xaxis.tick_bottom()

    ax.set_xlabel("Predicted label (argmax)")
    ax.set_ylabel("Truth label")

    cbar = fig.colorbar(mat)
    label = ""
    if norm_by in ['truth','pred']:     
        label += "Normalised by %s label"%norm_by
    if mass_cut: label += " ($m_{\\gamma\\gamma} \\in$ [123,127] GeV)"
    cbar.set_label(label)

    ext_str = "" if norm_by is None else "_normby%s"%norm_by
    if ext is not None: ext_str += "_%s"%ext
    if mass_cut: ext_str += "_masscut"
    fig.savefig("%s/confmatrix%s.png"%(plot_path,ext_str))

    ax.cla()
    cbar.remove()

    return conf_matrix, conf_matrix_labels


# Take argmax as



# Add ROC curve for three-class: compare test and train

# Plot probability distributions for test and train

# Play season out with loop for re-training when adding next round of games, exponential decay on importance from previous seasons
