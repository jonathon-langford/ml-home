import glob
import uproot
import awkward as ak
import pandas as pd
import vector
import re
vector.register_awkward()
import numpy as np

from timeit import default_timer as timer

from optparse import OptionParser
def get_options():
    parser = OptionParser()
    parser.add_option('-i','--input-dir', dest='input_dir', default="/vols/cms/jl2117/postdoc/teaching/mres/samples/Jan23/2017")
    return parser.parse_args()
(opt,args) = get_options()

# Common variables
tree_name = "Events"

# Mappings for correct normalisation
xs_map = {
    "ggH_M125" : 48.58*0.00270,
    "VBF_M125" : 3.782*0.00270,
    "VH_M125" : (1.373+0.7612)*0.00270,
    "ttH_M125" : 0.5071*0.00270,
    "tHq_M125" : (0.002879+0.07425)*0.00270,
    "tHW_M125" : 0.01517*0.00270,
    "DY" : 4746.0,
    "DiPhotonJetsBox_M40_80" : 303.2, 
    "DiPhotonJetsBox_MGG-80toInf": 84.4,
    "GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80" : 3255.0,
    "GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf" : 220.0,
    "GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf" : 862.4,
    "QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf" :  22110.0,
    "QCD_Pt-40toInf_DoubleEMEnriched_MGG-80toInf" : 113400.0,
    "QCD_Pt-30toInf_DoubleEMEnriched_MGG-40to80" : 260500.0 
}

ea_map = {
    "2017":{
        "ggH_M125" : 0.5254670,
        "VBF_M125" : 0.5339179,
        "VH_M125" : 0.4789266,
        "ttH_M125" : 0.5821746,
        "tHq_M125" : 0.5795102,
        "tHW_M125" : 0.5910044,
        "DY" : 0.0468661,
        "DiPhotonJetsBox_M40_80" : 0.0009297,
        "DiPhotonJetsBox_MGG-80toInf": 0.1732172,
        "GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80" : 0.0004272,
        "GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf" : 0.0188552,
        "GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf" : 0.0425448,
        "QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf" :  0.0003714,
        "QCD_Pt-40toInf_DoubleEMEnriched_MGG-80toInf" : 0.0007673,
        "QCD_Pt-30toInf_DoubleEMEnriched_MGG-40to80" : 0.0000336,
    }
}

lumi_map = {'2016A':19.52, '2016B':16.81,'2017':41.5, '2018':59.7}

# List of branches: 
branches = [
    'leadPhotonEn', 'leadPhotonMass', 'leadPhotonPt', 'leadPhotonEta', 'leadPhotonPhi', 'leadPhotonIDMVA', 'leadPhotonSigmaE', 'leadPhotonHoE', 'leadPhotonPfRelIsoAll', 'leadPhotonPfRelIsoChg', 'leadPhotonR9', 'leadPhotonSieie', 'leadPhotonElectronVeto', 'leadPhotonPixelSeed',
    'subleadPhotonEn', 'subleadPhotonMass', 'subleadPhotonPt', 'subleadPhotonEta', 'subleadPhotonPhi', 'subleadPhotonIDMVA', 'subleadPhotonSigmaE', 'subleadPhotonHoE', 'subleadPhotonPfRelIsoAll', 'subleadPhotonPfRelIsoChg', 'subleadPhotonR9', 'subleadPhotonSieie', 'subleadPhotonElectronVeto', 'subleadPhotonPixelSeed',
    'subsubleadPhotonEn', 'subsubleadPhotonMass', 'subsubleadPhotonPt', 'subsubleadPhotonEta', 'subsubleadPhotonPhi', 'subsubleadPhotonIDMVA', 'subsubleadPhotonSigmaE', 'subsubleadPhotonHoE', 'subsubleadPhotonPfRelIsoAll', 'subsubleadPhotonPfRelIsoChg', 'subsubleadPhotonR9', 'subsubleadPhotonSieie', 'subsubleadPhotonElectronVeto', 'subsubleadPhotonPixelSeed',
    'leadJetEn', 'leadJetMass', 'leadJetPt', 'leadJetEta', 'leadJetPhi', 'leadJetQGL', 'leadJetID', 'leadJetPUJID', 'leadJetBTagScore',
    'subleadJetEn', 'subleadJetMass', 'subleadJetPt', 'subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', 'subleadJetID', 'subleadJetPUJID', 'subleadJetBTagScore',
    'subsubleadJetEn', 'subsubleadJetMass', 'subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', 'subsubleadJetID', 'subsubleadJetPUJID', 'subsubleadJetBTagScore',
    'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronMvaFall17V2Iso', 'leadElectronCharge', 'leadElectronConvVeto',
    'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronMvaFall17V2Iso', 'subleadElectronCharge', 'subleadElectronConvVeto',
    'subsubleadElectronEn', 'subsubleadElectronMass', 'subsubleadElectronPt', 'subsubleadElectronEta', 'subsubleadElectronPhi', 'subsubleadElectronMvaFall17V2Iso', 'subsubleadElectronCharge', 'subsubleadElectronConvVeto',
    'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonPfRelIso04', 'leadMuonCharge',
    'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonPfRelIso04', 'subleadMuonCharge',
    'subsubleadMuonEn', 'subsubleadMuonMass', 'subsubleadMuonPt', 'subsubleadMuonEta', 'subsubleadMuonPhi', 'subsubleadMuonPfRelIso04', 'subsubleadMuonCharge',
    'diphotonMass', 'diphotonPt', 'diphotonEta', 'diphotonPhi', 'diphotonCosPhi', 'diphotonSigmaMoM', 'leadPhotonPtOvM', 'subleadPhotonPtOvM',
    'dijetMass', 'dijetPt', 'dijetEta', 'dijetPhi', 'dijetAbsDEta', 'dijetDPhi', 'dijetMinDRJetPho', 'dijetCentrality', 'dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'leadJetDiphoDPhi', 'leadJetDiphoDEta', 'subleadJetDiphoDPhi', 'subleadJetDiphoDEta', 'nSoftJets',
    'metPt', 'metPhi', 'metSumET', 'metSignificance',
     'centralObjectWeight', 'weight'
]

branches_mc = branches
branches_mc.extend( ['HTXS_Higgs_pt', 'HTXS_Higgs_y', 'HTXS_stage1_2_cat_pTjet30GeV', 'HTXS_njets30', 'genWeight'] )

# Get list of files to process
files = glob.glob("%s/*.root"%(opt.input_dir))
    
for f in files:

    process_name = f.split("/")[-1].split(".root")[0]

    print("\n\n --> Processing (%s): %s"%(process_name,f))

    f_upr = uproot.open(f)
    t = f_upr[tree_name]


    if "Data" in process_name:
        events = t.arrays( branches, library='pd')
        events['weight_lumiScaled'] = events['weight']
    else:
        events = t.arrays( branches_mc, library='pd')    
        # Apply normalisation get expected number of events after selection
        sum_gen_w = np.sum(events['genWeight'].values)
        # Scale by XS, efficiency
        events['weight'] *= xs_map['ggH_M125']
        events['weight'] *= ea_map['2017']['ggH_M125']
        events['weight'] /= sum_gen_w
        
        # Also weight scaled by luminosity
        events['weight_lumiScaled'] = events['weight'] * lumi_map['2017'] * 1000

    output_file_name = re.sub(".root", "_processed.parquet", f)

    events.to_parquet(output_file_name, engine="pyarrow")
    

