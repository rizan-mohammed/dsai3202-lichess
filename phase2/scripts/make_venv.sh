#!/usr/bin/env bash
# A clean Python with the SAME versions the model was trained with, for the offline side of same_input_check.py.
set -uo pipefail
if [ ! -x ~/pv/bin/python ]; then
  python3 -m venv ~/pv 2>/dev/null || { sudo apt-get install -y python3-venv >/dev/null && python3 -m venv ~/pv; } || { echo "could not create ~/pv"; exit 1; }
fi
~/pv/bin/pip install -q numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 pandas==2.2.3 mlflow-skinny==2.17.2 cloudpickle || exit 1
~/pv/bin/python -c "import sklearn, numpy, pandas, mlflow; print('ok: scikit-learn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'mlflow', mlflow.__version__)"
