import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore')

from training.bdt_utils import *

from optparse import OptionParser
def get_options():
    parser = OptionParser()
    parser.add_option('-i','--input-file', dest='input_file', default=None)
    parser.add_option('--ext', dest='ext', default="")
    return parser.parse_args()
(opt,args) = get_options()

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

#train_vars = ['diphotonPt', 'dijetMass', 'dijetCentrality', 'leadPhotonPtOvM', 'subleadPhotonPtOvM', 'leadPhotonEta', 'leadPhotonIDMVA', 'subleadPhotonEta', 'subleadPhotonIDMVA', 'leadJetPt', 'leadJetEta', 'subleadJetPt', 'subleadJetEta', 'dijetAbsDEta']

proc_map = {
    "ggH":1,
    "VBF":2,
    "VH":3,
    "ttH":4,
    "tH":5,
    "bkg":0
}

#proc_map = {
#    "ggH":1,
#    "VBF":2,
#    "bkg":0
#}


# Merge parquet files
df = pd.read_parquet(opt.input_file, engine="pyarrow")

bdt = BDT(df, train_vars, options=opt, proc_map=proc_map)
bdt.create_X_and_y()

# Train classifier
bdt.train_classifier()
bdt.evaluate_classifier()

# Add predicted probabilities to each event for evaluation
df_out = bdt.package_output()

ext_str = "_%s"%opt.ext if opt.ext != "" else ""
output_file = "/".join(opt.input_file.split("/")[:-1]) + "/output_data%s.parquet"%ext_str
df_out.to_parquet(output_file, engine="pyarrow")
