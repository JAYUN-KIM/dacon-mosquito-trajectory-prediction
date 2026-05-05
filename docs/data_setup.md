# Data Setup

Place the DACON `open.zip` contents under `data/raw`.

```text
data/raw/
├── train/
│   ├── TRAIN_00001.csv
│   ├── TRAIN_00002.csv
│   └── ...
├── test/
│   ├── TEST_00001.csv
│   ├── TEST_00002.csv
│   └── ...
├── train_labels.csv
└── sample_submission.csv
```

Each trajectory CSV should contain:

```text
timestep_ms, x, y, z
```

`train_labels.csv` and `sample_submission.csv` should contain:

```text
id, x, y, z
```

After the files are in place, run:

```powershell
python scripts/run_physics_baselines.py
```

The script evaluates deterministic physics baselines on a train holdout and writes a submission CSV to `submissions/`.

For a quick smoke test without creating a full submission:

```powershell
python scripts/run_physics_baselines.py --limit-train 100 --limit-test 100 --skip-submission
```
