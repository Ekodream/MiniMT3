# MiniMT3-Piano

MiniMT3-Piano is a compact piano automatic music transcription project. The first milestone is an offline MVP:

```text
audio -> log-mel -> encoder-decoder events -> constrained decoding -> MIDI/MusicXML -> local UI
```

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
```

## Train

```bash
python scripts/train.py --config configs/train.yaml
```

For multi-GPU training:

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/train.yaml
```

## Infer

```bash
python scripts/infer.py --audio path/to/audio.wav --ckpt outputs/ckpt/best.pt --out outputs/demo
```

## UI

```bash
python app/app.py --config configs/ui.yaml
```

The UI is local/offline by default.
