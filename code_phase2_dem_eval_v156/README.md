# Phase 2-D — LightGBM + spatial GAT + DEM

This is an independent DEM-enhanced copy of the existing Phase 2 model. It
does not overwrite code_phase2 or its outputs.

The model adds USGS 3DEP/NED 1 arc-second (~30 m) county terrain statistics
from:

~~~text
data/geo/county_terrain.csv
~~~

The seven static fields are elevation mean/std/min/max, slope mean/std, and
terrain ruggedness. Original DEM tile URLs and source metadata are in:

~~~text
data/geo/dem_3dep_1arcsec/manifest.json
~~~

The complete Chinese technical explanation is in README_CN.md.

## Build terrain features

~~~bash
conda activate myenv
pip install -r code_phase2_dem/requirements.txt
python code_phase2_dem/build_terrain.py --workers 8
~~~

## Run the full DEM model

~~~bash
conda activate myenv
python code_phase2_dem/main.py \
  --base-mode direct \
  --device cuda \
  --epochs 220 \
  --patience 35 \
  --time-stride 1 \
  --k 8
~~~

The full-scale run keeps all 144 forecast snapshots, 5 county folds, k=8
spatial neighbours, and up to 220 GAT epochs. Outputs are written only to:

~~~text
code_phase2_dem/outputs/
~~~

The final competition-format file is:

~~~text
code_phase2_dem/outputs/submission_phase2_dem_gat.csv
~~~
