#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --constrain='balrog|belegost|gondor|khazadum|rivendell|shelob|shire|smaug'

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate env_cv2

python3 /Midgard/home/areichlin/ObjRepresentation/trainer.py --loss_type ${LT} --frq ${F} --tau ${T} --m ${M} --e2e ${E} --layers_f ${L} --gamma ${G} --seed ${SEED}