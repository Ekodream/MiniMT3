# MiniMT3-Piano

MiniMT3-Piano is a compact piano automatic music transcription project. The first milestone is an offline MVP:

```text
audio -> log-mel -> dense piano AMT -> post-processing -> MIDI/MusicXML -> local UI
```

The current practical line is the dense AMT family. v5.3 is the stable display baseline, v5.4 adds ScorePolish
readable-score interpretation, and v8 is the active quality push toward stronger MT3/Onsets-and-Frames style behavior.
Earlier seq2seq/MT3-style experiments are kept in the repository for ablations, but the recommended inference path is
`scripts/infer_amt.py`.

## Setup

```bash
conda env create -f environment.yml
conda activate MiniMT3
pip install -e ".[eval,dev]"
```

MuseScore and/or Verovio are optional for score PDF/PNG/SVG rendering. Without them, the system still writes MIDI and MusicXML.

## Data

Download MAESTRO v3 into `data/maestro/`, then index it:

```bash
python scripts/prepare_maestro.py --data_dir data/maestro --out data/cache
python scripts/build_eval_manifest.py --index data/cache/maestro_index.json --out_dir data/cache
```

Validation uses fixed clips from `data/cache/maestro_val_clips.json`. This is intentional: random validation crops make `val_loss` noisy and unsuitable for checkpoint selection.

Inspect target token distribution before a long run:

```bash
python scripts/inspect_tokens.py --metadata data/cache/maestro_val_clips.json --split validation --sampling fixed
```

## Train

Recommended v5.2 dense AMT training:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v5_2_recall.yaml
```

The current best checkpoint from the 8-GPU run is:

```text
outputs/ckpt_amt_v5_2_recall/best.pt
```

It uses calibrated decoding defaults:

```text
onset_threshold=0.40
frame_threshold=0.10
offset_threshold=0.15
```

For the v5.3 display-quality pass, continue from the v5.2 checkpoint and add pedal supervision:

```bash
python scripts/train_amt.py --config configs/train_amt_v5_3_pedal_score_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v5_3_pedal_score.yaml
```

v5.3 keeps onset/frame/offset/velocity training, adds an optional `pedal_head`, and selects checkpoints with debug
note F1 plus note-count balance. When other GPU jobs are already using memory, lower `batch_size` in
`configs/train_amt_v5_3_pedal_score.yaml` before starting the 8-GPU run.

v6 duration/chord training keeps the v5.3 acoustic backbone, adds a duration head, and uses a zero-initialized
onset-conditioned frame adapter so training starts from the v5.3 behavior instead of resetting the frame predictor:

```bash
python scripts/train_amt.py --config configs/train_amt_v6_duration_chord_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v6_duration_chord.yaml
```

This line targets long notes and chord recall: duration regression helps note ends survive weak frame/offset regions,
and onset-frame fusion lets frame energy recover weaker chord tones during decoding.

v7 is the current mainline for improving model quality. It fixes the biggest issues found after v6: limited data
coverage, missing encoder position information, random new-module initialization, and slow uncached audio loading. It
uses a 123k-clip uniform MAESTRO manifest, learned zero-initialized position embeddings, residual zero-initialized
head towers, a residual bidirectional GRU temporal layer, conservative Onsets&Frames-style decoding, and stricter
checkpoint scoring against over-generated notes:

```bash
python scripts/build_amt_manifest.py \
  --index data/cache/maestro_index.json \
  --split train \
  --out data/cache/amt_train_2s_uniform128_v7.json \
  --clip_seconds 2.0 \
  --stride_seconds 1.0 \
  --max_clips_per_piece 128 \
  --sampling uniform \
  --seed 127

python scripts/train_amt.py --config configs/train_amt_v7_pos_towers_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v7_pos_towers.yaml
```

The v7 data loader reads only the requested WAV segment instead of loading the full performance for every 2s clip. This
matters when the training cache is cold.

v8 is the current response to low note F1 and poor long-note/chord behavior. The key diagnosis is that v7's 2s windows
and 123k clips are still too small compared with stronger piano AMT systems, which typically rely on onset-constrained
frame modeling, explicit offset/pedal targets, much larger segment coverage, and long training schedules. v8 keeps the
stable dense AMT path, starts from the v7 best checkpoint, moves to 4s clips, adds soft onset/offset boundary targets,
uses a 184k-clip manifest, extends duration supervision to 12s, and adds zero-gated extra context layers so new capacity
can learn without destroying the initialized v7 behavior:

```bash
python scripts/build_amt_manifest.py \
  --index data/cache/maestro_index.json \
  --split train \
  --out data/cache/amt_train_4s_uniform192_v8.json \
  --clip_seconds 4.0 \
  --stride_seconds 2.0 \
  --max_clips_per_piece 192 \
  --sampling uniform \
  --seed 137

python scripts/build_amt_manifest.py \
  --index data/cache/maestro_index.json \
  --split validation \
  --out data/cache/amt_val_4s_s8_v8.json \
  --clip_seconds 4.0 \
  --stride_seconds 8.0 \
  --max_clips_per_piece 2 \
  --sampling grid \
  --seed 137

python scripts/train_amt.py --config configs/train_amt_v8_context4_soft_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v8_context4_soft.yaml
```

Use v8 for new quality experiments. If it improves note F1 but remains underfitted, the next safe capacity step is to
raise `batch_size` and training steps first, then widen `d_model/head_hidden`; widening the backbone resets more weights
and should be treated as a separate ablation.

The first v8 run reached 6000 steps. It improved recall but over-generated with the training-time debug threshold
(`onset_threshold=0.34`, pred/ref around 2.6-2.8). A fixed 48-clip threshold sweep showed a better demo/eval operating
point around:

```text
onset_threshold=0.42
frame_threshold=0.12
offset_threshold=0.15
pred_ref_ratio≈1.23
note_f1≈0.30
```

`infer_amt.py` now automatically uses this more conservative onset threshold for v8-style 4s checkpoints unless an
explicit `--onset_threshold` is passed.

v8.1 precision fine-tuning is a follow-up for the over-generation seen in v8. It starts from the v8 best checkpoint,
reduces the soft onset radius from 2 frames to 1 frame, strengthens negative onset/frame penalties, and raises the
debug/default onset threshold to 0.42:

```bash
python scripts/train_amt.py --config configs/train_amt_v8_1_precision4_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v8_1_precision4.yaml
```

Use this run when the priority is cleaner notes and less score clutter rather than maximum recall.

v8.1 should be decoded less conservatively than its training debug default. A 48-clip sweep found that
`onset_threshold=0.32` gives a better note-count balance than 0.42:

```text
v8.1 best, onset_threshold=0.32: note_f1≈0.313, pred_ref_ratio≈1.21
```

v9 capacity-plus tests whether the model is now capacity-limited without throwing away v8.1 weights. It keeps
`d_model=256` and the existing heads, then adds four zero-gated Transformer adapter layers. Because the adapter gates
start at zero, the initial model behaves like v8.1 and learns to use the extra capacity gradually:

```bash
python scripts/train_amt.py --config configs/train_amt_v9_capacity_plus_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v9_capacity_plus.yaml
```

Only consider a true widened backbone, such as `d_model=384`, after v9 has been evaluated; that route discards many
checkpoint weights and is closer to a new training run.

v10 high-resolution shift is the next mainline after v9 showed that adapter capacity alone was not enough. The key
change is ByteDance-style high-resolution timing: the dense AMT output is moved from roughly 40 ms to roughly 20 ms by
changing the encoder strides from `[2, 2]` to `[2, 1]`, and the model adds onset/offset sub-frame shift regression
heads. This keeps compatible v9 weights where shapes match while giving the decoder finer onset/offset placement:

```bash
python scripts/train_amt.py --config configs/train_amt_v10_hires20_shift_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v10_hires20_shift.yaml
```

Use v10 when the priority is breaking the note-F1 ceiling. If v10 improves onset F1, the next capacity step is a true
wide high-resolution model (`d_model/head_hidden` 320-384). If it does not, the bottleneck is likely target/data quality
or decoding calibration rather than parameter count.

The first v10 long run reached 9000 steps. It learned the shift heads and improved offset quality, but did not break the
note-F1 ceiling:

```text
v10 best step≈4500: debug note_f1≈0.291, offset_f1≈0.118, pred/ref≈1.32
v10 step 9000:      debug note_f1≈0.300, offset_f1≈0.133, pred/ref≈1.63
48-clip sweep:      best note_f1≈0.301 at onset=0.34, frame=0.16, offset=0.12
```

v11 high-resolution wide is the follow-up when v10 plateaus. It widens the model to `d_model=384/head_hidden=384`,
keeps the high-resolution `[2, 1]` strides and shift regression, and uses expanded initialization from v10: matching
sub-blocks are copied into the wider tensors while new dimensions start at zero. This avoids restarting from scratch
while giving the model more acoustic capacity:

```bash
python scripts/train_amt.py --config configs/train_amt_v11_hires20_wide384_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v11_hires20_wide384.yaml
```

v11 also enables debug-score early stopping so long runs do not keep training after the best checkpoint has clearly
stalled. This is important because v8-v10 often kept lowering loss while pred/ref drifted upward.

The completed v11 run confirms that simply widening the existing Transformer-style dense AMT model is not enough:

```text
v11 best score: step=1200, debug note_f1≈0.259, offset_f1≈0.092, pred/ref≈1.18
v11 step=2700:  debug note_f1≈0.286, offset_f1≈0.107, pred/ref≈1.51
v11 step=3600:  debug note_f1≈0.267, offset_f1≈0.104, pred/ref≈1.48, early-stopped
```

v12 is therefore a larger architectural change, not another threshold sweep. It follows the practical direction of
high-resolution piano transcription systems: 10 ms-ish features, independent CRNN towers for onset/frame/offset/
velocity/pedal, onset-conditioned frame prediction, and sub-frame onset/offset shift regression. The goal is to move
the bottleneck from "shared small encoder cannot separate boundaries" to supervised acoustic boundary detection:

```bash
python scripts/train_amt.py --config configs/train_amt_v12_crnn_bytedance_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v12_crnn_bytedance.yaml
```

If v12 still plateaus far below `note_f1=0.5`, the next correct step is not more decode tuning. Inspect target alignment,
audio/MIDI synchronization, and manifest difficulty; then increase data scale and training duration with this
high-resolution multi-head architecture.

The first v12 run showed that the model was much better than the original debug threshold suggested. Low-threshold debug
at step 6000 over-generated (`pred/ref≈2.86`), but a fixed 32-clip sweep with energy-consumption decoding gave:

```text
v12 last.pt + consume_note_energy:
  best global thresholds: onset=0.42, frame=0.20, offset=0.24
  note_f1≈0.626, offset_f1≈0.250, pred/ref≈0.94
```

This means the main failure was decode calibration and duplicate/echo notes, not pure model capacity. The current main
demo/eval path should use `consume_note_energy=true`. A too-strong precision fine-tune (`v12.1`) quickly became
under-generating, so `v12.2` uses a much lighter mass regularizer and keeps the calibrated lower onset threshold:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v12_2_crnn_calibrated.yaml
```

v13-wide is the next main quality push. It does not keep polishing v12 thresholds indefinitely: v12 is used as a
calibrated baseline and as expanded initialization, then v13 increases CRNN capacity and trains on 8s windows with the
center 6s supervised. This gives the model more left/right context for chords, long notes, pedal/legato, and window
boundary continuity while keeping 10 ms high-resolution targets.

Build the v13 manifests, smoke, and launch the 8-GPU run on the remote machine:

```bash
bash scripts/remote_start_v13_wide.sh
```

Equivalent manual commands:

```bash
python scripts/build_amt_manifest.py --index data/cache/maestro_index.json --split train \
  --out data/cache/amt_train_8s_uniform2048_v13.json --clip_seconds 8 \
  --sampling uniform --max_clips 2048 --max_clips_per_piece 8 --seed 173
python scripts/build_amt_manifest.py --index data/cache/maestro_index.json --split validation \
  --out data/cache/amt_val_8s_s8_v13.json --clip_seconds 8 --stride_seconds 8 --max_clips 128
python scripts/train_amt.py --config configs/train_amt_v13_wide_smoke.yaml
torchrun --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v13_wide.yaml
```

v13 default decoding uses Onsets-and-Frames style onset gating, Basic-Pitch-style frame-diff onset recovery, and
energy-consumption duplicate suppression. For v12 checkpoints, `infer_amt.py` now defaults to the calibrated profile
(`onset=0.42, frame=0.20, offset=0.24, consume_note_energy=true`) unless explicit CLI thresholds are passed.

v5.4 ScorePolish is an inference/post-processing layer rather than a new architecture checkpoint. It separates
performance MIDI from readable score export, prunes pedal-resonance long notes from the score path, estimates
key/tempo, quantizes to a beat grid, uses dynamic-programming hand assignment, trims overlapping score durations per
hand, and writes piano grand-staff MusicXML:

```bash
python scripts/infer_amt.py \
  --audio ../demo/test_30s.wav \
  --ckpt outputs/ckpt_amt_v5_3_pedal_score/best.pt \
  --out outputs/demo \
  --time_signature 4/4
```

For a polished demo with known score context, override the automatic estimates:

```bash
python scripts/infer_amt.py \
  --audio ../demo/test_30s.wav \
  --ckpt outputs/ckpt_amt_v5_3_pedal_score/best.pt \
  --out outputs/demo \
  --key_signature "F# major" \
  --time_signature 4/4 \
  --tempo_bpm 80
```

Useful display controls:

```bash
--window_seconds 4.0              # match v8's 4s training context
--overlap_seconds 0.75            # reduce window-boundary chord/long-note damage
--score_max_notes_per_beat 8      # lower for cleaner scores, higher for dense passages
--score_max_overlap_beats 0.0     # keep score durations from overlapping the next hand event
--disable_score_overlap_trim      # keep longer notated durations for ablation
--time_signature 12/8             # useful for compound-meter pieces such as Merry Christmas Mr. Lawrence
--score_beat_divisions 2,4        # suppress triplet clutter; add 3 only for real tuplets
--score_max_short_rest_beats 0.75 # fill tiny score gaps by extending notes to the next onset
--disable_score_start_align       # keep true opening rests/pickups instead of trimming leading silence
--score_start_offset_beats 1.0    # manually remove a known pickup/leading offset from the score view
--score_chord_snap_seconds 0.075  # align near-simultaneous chord tones without collapsing arpeggios
--disable_score_key_filter        # ablation: keep weak chromatic/non-key notes
```

The score path is intentionally stricter than the performance MIDI path. It now filters weak non-key/isolated notes,
suppresses tuplets by default, fills short notation gaps, and spells pitches according to the chosen key signature.
For ablations, compare `--disable_score_key_filter`, `--disable_score_isolation_filter`,
`--disable_score_fill_rests`, and `--score_allow_tuplets`.

### Dense-AMT quality push: v13 large data before larger models

The current dense-AMT bottleneck is no longer just parameter count. The known scale checkpoints are:

```text
v12_crnn_bytedance:      about 24.93M params, 184,704 train clips
v13_wide:               about 64.89M params,   2,048 train clips
v13_large_finetune:     about 64.89M params, large 8s manifest target
v14_mid fallback:       about 47M params, large 8s manifest target
```

v13 widened the model but trained on too few 8s clips; the best debug point was around step 1200 and later checkpoints
over-generated. The next main experiment is therefore data repair plus calibrated selection, not another blind width
increase.

Prepare the large train/calibration/score-quality manifests on the remote host:

```bash
bash scripts/remote_prepare_v13_quality.sh
```

Launch the first large-data fine-tune only after checking the remote GPUs:

```bash
LAUNCH_TRAIN=1 bash scripts/remote_prepare_v13_quality.sh
```

The first-stage config is `configs/train_amt_v13_large_finetune.yaml`. It initializes from
`outputs/ckpt_amt_v13_wide/best.pt`, uses the large 8s manifest, adds light mass regularization, and selects checkpoints
with a small decode-threshold sweep. If two v13 fine-tunes do not improve the calibrated validation set meaningfully,
switch to `configs/train_amt_v14_mid.yaml` instead of continuing to lower LR indefinitely.

For the next note-F1-first run, use v15. The default v15 path keeps the v13 backbone scale, adds duration supervision,
adds one lightweight attention-context block, enables SpecAugment-style feature masking, and selects checkpoints with
precision/recall-aware decode sweeps:

```bash
PROFILE=f1_duration LAUNCH_TRAIN=1 bash scripts/remote_start_v15_f1.sh
```

Only use the larger fallback after the v15 duration run is evaluated:

```bash
PROFILE=xlarge_duration LAUNCH_TRAIN=1 bash scripts/remote_start_v15_f1.sh
```

Report model scale and manifest size:

```bash
python scripts/amt_model_report.py \
  --config configs/train_amt_v12_crnn_bytedance.yaml \
  --config configs/train_amt_v13_wide.yaml \
  --config configs/train_amt_v13_large_finetune.yaml \
  --config configs/train_amt_v14_mid.yaml \
  --config configs/train_amt_v15_f1_duration.yaml \
  --config configs/train_amt_v15_xlarge_duration.yaml
```

Run diagnostic eval with precision/recall, duration buckets, chord metrics, velocity error, score-quality metrics, and
false-positive/false-negative examples:

```bash
python scripts/eval_amt.py \
  --ckpt outputs/ckpt_amt_v13_wide/best.pt \
  --manifest data/cache/amt_val_8s_s8_calib512_v13.json \
  --items 512 \
  --cache_dir data/cache/amt_v13_calib512_val_8s_10ms_229mel_center6 \
  --decode_preset practice_score \
  --score_quality_eval \
  --analysis_json_out outputs/eval_v13_best_practice_score.json
```

For student transcription, `scripts/infer_amt.py` now defaults to `--decode_preset practice_score`, which favors clean,
readable MusicXML. Use `--decode_preset analysis_midi` or `--decode_preset v13_recall` for higher-recall debugging.

For the current v13/v15 mainline comparison on the remote host, keep the running training session untouched and launch
the single-GPU eval in a separate tmux session:

```bash
LAUNCH_EVAL=1 bash scripts/remote_eval_v13_v15.sh
```

For the Test_new score-quality comparison, use the fixed output root below instead of ad-hoc temporary folders:

```bash
LAUNCH_DEMO=1 bash scripts/remote_run_test_new_compare.sh
```

The score renderer is measure-aware across both piano staves. It hides filler rests in active measures, keeps visible
full-measure rests only when a staff is genuinely silent, protects sustained bass/inner voices from overlap trimming,
and locks near-onset chord groups before MusicXML rendering.

Legacy seq2seq training is still available:

```bash
python scripts/train.py --config configs/train.yaml
```

For multi-GPU training:

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/train_8gpu.yaml
```

Tiny overfit sanity check:

```bash
python scripts/train.py --config configs/train_tiny.yaml
```

## Infer

Recommended dense AMT inference:

```bash
python scripts/infer_amt.py \
  --audio path/to/audio.wav \
  --ckpt outputs/ckpt_amt_v5_2_recall/best.pt \
  --out outputs/demo
```

For the bundled demo clip, run:

```bash
python scripts/infer_amt.py \
  --audio ../demo/test_30s.wav \
  --ckpt outputs/ckpt_amt_v5_2_recall/best.pt \
  --out outputs/demo
```

The script writes:

```text
outputs/demo/<audio_name>.mid
outputs/demo/<audio_name>_score.mid
outputs/demo/<audio_name>.musicxml
outputs/demo/<audio_name>_debug.json
```

It performs sliding-window inference, removes duplicated notes produced by overlap regions, and then splits the output
into two views: a performance MIDI with sustain pedal regions, and a cleaner quantized score view for MusicXML. For
checkpoints without a learned pedal head, `infer_amt.py` uses a conservative sustain heuristic; pass
`--disable_sustain_heuristic` to turn that off.

Summarize score readability metrics:

```bash
python scripts/eval_score_polish.py --out_dir outputs/demo
```

Legacy event-decoder inference:

```bash
python scripts/infer.py --audio path/to/audio.wav --ckpt outputs/ckpt/best.pt --out outputs/demo
```

## UI

```bash
python app/app.py --config configs/ui.yaml
```

The UI is local/offline by default.
