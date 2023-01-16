CondaVer=3
unset PYTHONPATH
MINICONDA_DIR=/vols/cms/${USER}/miniconda${CondaVer}

cwd=${PWD}

wget https://repo.continuum.io/miniconda/Miniconda${CondaVer}-latest-Linux-x86_64.sh
bash Miniconda${CondaVer}-latest-Linux-x86_64.sh -b -p $MINICONDA_DIR
rm Miniconda${CondaVer}-latest-Linux-x86_64.sh

cd $MINICONDA_DIR

source etc/profile.d/conda.sh
conda update -y -n base -c defaults conda

export PYTHONNOUSERSITE=true

cd ${cwd}
