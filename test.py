import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

ggH_file_name = "/vols/cms/jl2117/postdoc/teaching/mres/samples/Jan23/2017/processed_data/ggH_M125_processed.parquet"
VBF_file_name = "/vols/cms/jl2117/postdoc/teaching/mres/samples/Jan23/2017/processed_data/VBF_M125_processed.parquet"

ggH_df = pd.read_parquet( ggH_file_name, engine="pyarrow" )
VBF_df = pd.read_parquet( VBF_file_name, engine="pyarrow" )

# Extract lead photon pt
ggH_photon_pt = ggH_df['leadPhotonPt']
ggH_weight = ggH_df['weight']
VBF_photon_pt = VBF_df['leadPhotonPt']
VBF_weight = VBF_df['weight']

#Normalise sum of weights to 1
ggH_weight = ggH_weight/np.sum(ggH_weight)
VBF_weight = VBF_weight/np.sum(VBF_weight)

fig, ax = plt.subplots()
ax.hist( ggH_photon_pt, bins=40, range=(0,200), label="ggH ($m_{H}=125$ GeV)", color='blue', histtype='step', weights=ggH_weight)
ax.hist( VBF_photon_pt, bins=40, range=(0,200), label="VBF ($m_{H}=125$ GeV)", color='orange', histtype='step', weights=VBF_weight)
ax.set_xlabel("Lead photon $p_T$ [GeV]")
ax.set_ylabel("Fraction of events")
ax.legend(loc='best')
fig.savefig("test.pdf")


