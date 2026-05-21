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
--score_max_notes_per_beat 8      # lower for cleaner scores, higher for dense passages
--score_max_overlap_beats 0.0     # keep score durations from overlapping the next hand event
--disable_score_overlap_trim      # keep longer notated durations for ablation
--time_signature 12/8             # useful for compound-meter pieces such as Merry Christmas Mr. Lawrence
--score_beat_divisions 2,4        # suppress triplet clutter; add 3 only for real tuplets
--score_max_short_rest_beats 0.75 # fill tiny score gaps by extending notes to the next onset
--disable_score_key_filter        # ablation: keep weak chromatic/non-key notes
```

The score path is intentionally stricter than the performance MIDI path. It now filters weak non-key/isolated notes,
suppresses tuplets by default, fills short notation gaps, and spells pitches according to the chosen key signature.
For ablations, compare `--disable_score_key_filter`, `--disable_score_isolation_filter`,
`--disable_score_fill_rests`, and `--score_allow_tuplets`.

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
