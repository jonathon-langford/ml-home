import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

input_path = "/vols/cms/jl2117/ix/hgg/inference_aware/Jun23/samples"
plot_path = "./plots_perprodmode/input_features_pretrain"
do_plots = True

# Define procs to process
procs = ['ggH_M125', 'VBF_M125', 'VH_M125', 'ttH_M125', 'tHq_M125', 'tHW_M125', 'DiPhotonJetsBox_M40_80', 'DiPhotonJetsBox_MGG-80toInf', 'GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf', 'GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80', 'GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf'] #'QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf', 'QCD_Pt-30toInf_DoubleEMEnriched_MGG-40to80', 'QCD_Pt-40toInf_DoubleEMEnriched_MGG-80toInf']
#procs = ['ggH_M125', 'VBF_M125', 'DiPhotonJetsBox_M40_80', 'DiPhotonJetsBox_MGG-80toInf', 'GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf', 'GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80', 'GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf'] #'QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf', 'QCD_Pt-30toInf_DoubleEMEnriched_MGG-40to80', 'QCD_Pt-40toInf_DoubleEMEnriched_MGG-80toInf']

proc_map = {
    "ggH_M125":1,
    "VBF_M125":2,
    "VH_M125":3,
    "ttH_M125":4,
    "tHq_M125":5,
    "tHW_M125":5
}

# Open parquet files
dfs = []
#columns = ['diphotonMass', 'diphotonPt', 'dijetMass', 'dijetCentrality', 'leadPhotonPtOvM', 'subleadPhotonPtOvM', 'leadPhotonEta', 'leadPhotonIDMVA', 'subleadPhotonEta', 'subleadPhotonIDMVA', 'leadJetPt', 'leadJetEta', 'subleadJetPt', 'subleadJetEta', 'dijetAbsDEta', 'weight']
columns = [
    'leadPhotonPt', 'leadPhotonEta', 'leadPhotonIDMVA', 
    'subleadPhotonPt', 'subleadPhotonEta', 'subleadPhotonIDMVA',
    'leadJetPt', 'leadJetEta', 'leadJetPhi', 'leadJetQGL', 'leadJetBTagScore', 'leadJetDiphoDPhi', 'leadJetDiphoDEta',
    'subleadJetPt', 'subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', 'subleadJetBTagScore', 'subleadJetDiphoDPhi', 'subleadJetDiphoDEta',
    'subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', 'subsubleadJetBTagScore', 
    'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 
    'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi',
    'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 
    'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi',
    'dijetMass', 'dijetPt', 'dijetEta', 'dijetPhi', 'dijetAbsDEta', 'dijetDPhi', 'dijetMinDRJetPho', 'dijetCentrality', 'dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'nSoftJets',
    'metPt', 'metPhi', 'metSumET', 'metSignificance',
    'diphotonMass','diphotonSigmaMoM', 
    'weight',
    'HTXS_stage1_2_cat_pTjet30GeV'
]

for proc in procs:

    print(" --> Processing: %s"%proc )
    df = pd.read_parquet( "%s/%s_processed.parquet"%(input_path,proc), engine="pyarrow", columns=columns ) 

    # Apply selection: require events to have two jets and dijet mass > 350 GeV
    mask = ( df['diphotonMass'] > 100 )&( df['diphotonMass'] < 180 )
    #mask = mask & (df['dijetMass'] > 350.)
    df = df[mask]

    # Add labels
    df['proc'] = proc

    if proc in proc_map: df['proc_id'] = proc_map[proc]
    else: df['proc_id'] = 0

    # Append to list
    dfs.append(df)

df = pd.concat(dfs)

# Add normalised weight column for each class
df['sumw'] = 0
for proc_id in np.unique( df['proc_id'] ):
    df['sumw'] += (df['proc_id']==proc_id)*(df[df['proc_id']==proc_id]['weight'].sum())
df['weight_norm'] = df['weight']/df['sumw']


if do_plots:
    # Make plots for each feature
    proc_imap = {
        0:["Background","black"],
        1:["ggH","cornflowerblue"],
        2:["VBF","darkorange"],
        3:["VH","forestgreen"],
        4:["ttH","orchid"],
        5:["tH","gold"]
    }
    
    fig, ax = plt.subplots()
    for var in columns:
        if var in ["weight","HTXS_stage1_2_cat_pTjet30GeV"]: continue

        print(" -> Plotting variable: %s"%var)

        # Remove nans
        mask = (df[var]!=-999)&(df[var]==df[var])
        df_mask = df[mask]
    
        minimum = df_mask[var].min()
        maximum = np.percentile(df_mask[var], 98)
    
        for proc_id in np.unique( df_mask['proc_id'] ):

            ax.hist( df_mask[df_mask['proc_id']==proc_id][var], bins=40, range=(minimum,maximum), label=proc_imap[proc_id][0], color=proc_imap[proc_id][1], histtype='step', weights=df_mask[df_mask['proc_id']==proc_id]['weight_norm'] )
    
        ax.set_xlabel(var)
        ax.set_ylabel("Fraction of events")
        ax.legend(loc='best')
        fig.savefig("%s/%s.png"%(plot_path,var))
    
        ax.cla()

# Save dataframe as parquet file
#df.to_parquet( "%s/input_data.parquet"%input_path, engine="pyarrow" )
df.to_parquet( "%s/input_data_perprodmode.parquet"%input_path, engine="pyarrow" )


