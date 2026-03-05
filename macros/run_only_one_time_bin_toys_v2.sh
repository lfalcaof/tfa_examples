#!/bin/bash

python d02kspipi_toys_v2.py \
  --output Toy_WithACC_tbin9_only_1Mi_events_selTwoTracks_dalitz \
  --ntoys 1000000 \
  --seed 1 \
  --use_acceptance \
  --acc_root_dp ../../time_dependent_acceptance/file_results/acc_time_dep_selTwoTracks_dalitz.root \
  --acc_hist_pattern "accDP_tbin_{:02d}" \
  --acc_single_tbin 9
