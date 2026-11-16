#!/usr/bin/env bash
# The figure. Run after the experiment: bash make_figures.sh
set -e
R=${RESULTS:-results}; F=${FIGURES:-figures}

python -m sprlhf.plot $R/strategic_fraction.csv.gz --x actual_strategic_fraction --y norm_suboptimality \
   --col env_name --row labelers.strategy --algorithms pooled_btl maxmin psw pmomle \
   --title "Welfare lost as the share of liars rises" --out $F/norm_suboptimality.png
