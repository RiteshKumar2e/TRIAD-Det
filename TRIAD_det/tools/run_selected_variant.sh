#!/bin/sh
# Variant "selected": TRIAD-Det uses the validation-selected recipe per dataset (GC10-DET: v1, others: v2).
cd "$(dirname "$0")/../TRIAD_Det_Risk_Control" || exit 1
PY=../.venv/Scripts/python.exe
$PY experiments/cross_domain.py --variant selected && $PY experiments/recalibration.py --variant selected && $PY experiments/ablation.py --variant selected
$PY experiments/in_domain.py --variant selected
$PY experiments/figures.py --variant selected
$PY experiments/validity_check.py --variant selected
echo finished
