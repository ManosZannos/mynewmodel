#!/usr/bin/env bash
# run_phase3.sh — βήματα της Φάσης 3 στον server, ένα τη φορά (σε tmux). Όλα γράφουν log στο reports/.
#
#   bash run_phase3.sh tests            tests (λίγα λεπτά) — πρώτα αυτό· αν κάτι αποτύχει, σταμάτα
#   bash run_phase3.sh field            στάδιο Δ (πολικό πεδίο) στα τρία builds, CPU
#   bash run_phase3.sh field_report     ταβάνι κέρδους κάθε καναλιού στο val, με αναφορά το CV (CPU, μετά το field)
#   bash run_phase3.sh base           LSTM αναφοράς + CV + M0, seeds 0–2 (πύλη Π0)
#   bash run_phase3.sh m0_trials        οι τρεις δοκιμές του M0: χωρίς περιστροφή, χωρίς CV, GRU
#   bash run_phase3.sh channels         ξηρά, πλοία, ξηρά+πλοία, όλα, και το σκέλος σύγκρισης shore, seeds 0–5
#   bash run_phase3.sh shuffle          έλεγχος μηχανισμού στα μοντέλα με context (θέλει τα checkpoints του channels)
#   bash run_phase3.sh score            ετυμηγορίες
#   C=land bash run_phase3.sh train_shuffle   σκέλος ελέγχου για κανάλια που ΠΕΡΑΣΑΝ: εκπαίδευση με ανακατεμένο context, seeds 0–5
#
# ΕΦΕΔΡΙΚΑ σκέλη, μόνο αν το πολικό πεδίο δεν περάσει τις πύλες (docs/phase3_handoff.md §6δ):
#   bash run_phase3.sh arms_data        στάδιο Ε: γείτονες και raster SDF, CPU, στον server προεπεξεργασίας
#   bash run_phase3.sh arms             pooling γειτόνων και patch SDF, seeds 0–5
#   bash run_phase3.sh arms_shuffle     έλεγχος μηχανισμού στα εφεδρικά
#   bash run_phase3.sh arms_score       ετυμηγορίες των εφεδρικών
#
# Μία GPU: όλα τρέχουν σε σειρά. Μόνο validation· τα test sets δεν αγγίζονται από αυτό το script.
# Ο φάκελος εξόδου έχει το όνομα του μηχανήματος (RUN), γιατί συγκρίνουμε μόνο runs του ίδιου μηχανήματος.
set -eo pipefail
export PYTHONUNBUFFERED=1
S03=${S03:-dataset/socal_2025-03_b19c45d7}
S12=${S12:-dataset/socal_2025-12_b19c45d7}
HOU=${HOU:-dataset/houston_2025-03_7bfeb419}
OSM=${OSM:-data/osm/2026-10-29}
RUN=${RUN:-rtx}
OUT=reports/phase3_$RUN
REF=reports/conv_v37_$RUN
mkdir -p "$OUT" "$REF"

case "$1" in
tests)
  # δεν σταματά στο πρώτο λάθος: δείχνει όλα τα αποτυχημένα· χωρίς shapely (DGX) αποτυγχάνουν μόνο τα *shapely* και το test_stageC
  for t in test_loader test_stageC test_stageD test_phase3 test_arms; do
    echo "== $t"; python $t.py 2>&1 | grep -E "✗|πέρασαν|Error" || true
  done ;;
field)
  { python stageD_field.py $S03 --osm_dir $OSM
    python stageD_field.py $S12 --osm_dir $OSM --prior_build $S03
    python stageD_field.py $HOU --osm_dir $OSM; } 2>&1 | tee reports/stageD_$RUN.log ;;
field_report)
  python field_report.py $S03 2>&1 | tee reports/field_report_$RUN.log ;;
base)
  { python baselines.py --train_build $S03 --model cv --eval_split val --out_dir $REF
    for s in 0 1 2; do python baselines.py --train_build $S03 --model lstm --eval_split val --seed $s --out_dir $REF; done
    for s in 0 1 2; do python phase3.py --train_build $S03 --seed $s --out_dir $OUT; done; } 2>&1 | tee reports/phase3_base_$RUN.log ;;
m0_trials)
  { for f in --no_rotate --no_cv "--encoder gru"; do for s in 0 1 2; do
      python phase3.py --train_build $S03 --seed $s $f --out_dir $OUT
    done; done; } 2>&1 | tee reports/phase3_m0_trials_$RUN.log ;;
channels)
  { for s in 3 4 5; do python phase3.py --train_build $S03 --seed $s --out_dir $OUT; done      # M0 to 6 seeds
    for c in land vessels land,vessels land,vessels,habit shore; do for s in 0 1 2 3 4 5; do
      python phase3.py --train_build $S03 --seed $s --channels $c --out_dir $OUT
    done; done; } 2>&1 | tee reports/phase3_channels_$RUN.log ;;
shuffle)
  { for m in $OUT/m_*_vessel_val_s?.pt; do
      case $m in *m_pool_*|*m_sdf_*) continue ;; esac          # τα εφεδρικά έχουν το δικό τους βήμα
      for k in all scene; do
        python phase3.py --train_build $S03 --load_model $m --shuffle_context $k --out_dir $OUT
      done
    done; } 2>&1 | tee reports/phase3_shuffle_$RUN.log ;;
train_shuffle)
  # control arm for a channel set that passed its gate: C=land bash run_phase3.sh train_shuffle
  [ -n "$C" ] || { echo "δώσε C=<κανάλια>, π.χ. C=land ή C=land,vessels"; exit 1; }
  N=$(echo $C | tr ',' '\n' | sort | paste -sd+); B=$(basename $S03); P="${B}_${B}_vessel_val_s?"
  { for s in 0 1 2 3 4 5; do python phase3.py --train_build $S03 --seed $s --channels $C --shuffle_train --out_dir $OUT; done
    echo "== $N: εκπαίδευση με ανακατεμένο context (a) έναντι πραγματικού (b)"
    python scorecard.py --a "$OUT/m_${N}_${P}_trshuf.json" --b "$OUT/m_${N}_${P}.json" || true
    echo "== M0 έναντι εκπαίδευσης με ανακατεμένο context (αναμένεται: χωρίς effect)"
    python scorecard.py --a "$OUT/m0_${P}.json" --b "$OUT/m_${N}_${P}_trshuf.json" || true
  } 2>&1 | tee reports/phase3_train_shuffle_${N}_$RUN.log ;;
score)
  B=$(basename $S03); P="${B}_${B}_vessel_val_s?"
  { echo "== Π0: M0 έναντι LSTM"; python scorecard.py --a "$REF/lstm_${P}.json" --b "$OUT/m0_${P}.json"
    for v in norot nocv gru; do echo "== M0 έναντι δοκιμής $v"; python scorecard.py --a "$OUT/m0_${P}.json" --b "$OUT/m0_${P}_$v.json" || true; done
    echo "== ξηρά έναντι M0"; python scorecard.py --a "$OUT/m0_${P}.json" --b "$OUT/m_land_${P}.json" \
        --target by_scene_group.harbor.FDE || true
    echo "== πλοία έναντι M0"; python scorecard.py --a "$OUT/m0_${P}.json" --b "$OUT/m_vessels_${P}.json" || true
    echo "== ξηρά+πλοία έναντι ξηράς"; python scorecard.py --a "$OUT/m_land_${P}.json" --b "$OUT/m_land+vessels_${P}.json" || true
    echo "== +συνήθεια έναντι ξηράς+πλοίων"; python scorecard.py --a "$OUT/m_land+vessels_${P}.json" --b "$OUT/m_habit+land+vessels_${P}.json" || true
    echo "== shore (σκέλος σύγκρισης) έναντι M0"; python scorecard.py --a "$OUT/m0_${P}.json" --b "$OUT/m_shore_${P}.json" || true
    echo "== ξηρά έναντι shore"; python scorecard.py --a "$OUT/m_shore_${P}.json" --b "$OUT/m_land_${P}.json" || true
  } 2>&1 | tee reports/phase3_score_$RUN.log ;;
arms_data)
  { for b in $S03 $S12 $HOU; do python stageE_arms.py nb $b; python stageE_arms.py sdf $b --osm_dir $OSM; done
  } 2>&1 | tee reports/stageE_$RUN.log ;;
arms)
  { for a in pool sdf; do for s in 0 1 2 3 4 5; do
      python phase3_$a.py --train_build $S03 --seed $s --out_dir $OUT
    done; done; } 2>&1 | tee reports/phase3_arms_$RUN.log ;;
arms_shuffle)
  { for a in pool sdf; do for m in $OUT/m_${a}_*_vessel_val_s?.pt; do for k in all scene; do
      python phase3_$a.py --train_build $S03 --load_model $m --shuffle_context $k --out_dir $OUT
    done; done; done; } 2>&1 | tee reports/phase3_arms_shuffle_$RUN.log ;;
arms_score)
  B=$(basename $S03); P="${B}_${B}_vessel_val_s?"
  { echo "== pooling γειτόνων έναντι M0"; python scorecard.py --a "$OUT/m0_${P}.json" --b "$OUT/m_pool_${P}.json" || true
    echo "== patch SDF έναντι M0"; python scorecard.py --a "$OUT/m0_${P}.json" --b "$OUT/m_sdf_${P}.json" \
        --target by_scene_group.harbor.FDE || true
    echo "== patch SDF έναντι shore"; python scorecard.py --a "$OUT/m_shore_${P}.json" --b "$OUT/m_sdf_${P}.json" || true
  } 2>&1 | tee reports/phase3_arms_score_$RUN.log ;;
*)
  sed -n 2,20p "$0" ;;
esac
