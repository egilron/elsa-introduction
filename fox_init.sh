source /etc/profile
module purge
module load git-lfs/3.0.2
module load PyTorch/1.9.0-fosscuda-2020b
# python -m venv ~/venvs/transformers --clear
source ~/venvs/transformers/bin/activate
pip install -r requirements.txt
