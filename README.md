# ETF Shortlist Rescue Benchmark

An ETF can look ordinary on its own and still fit well with the others. This pilot checks what a shortlist leaves out and which samplers can find it.

## Run

Python 3.12

```sh
python -m pip install -r requirements.txt
python -m pilot --check
```

Tables in `runs/results/`, per-run CSVs in `runs/raw/`. To try a sampler, start with [`propose()`](pilot/samplers.py).

## Results

[Screening](results/screening.csv): the shortlist missed the best mix in **4 of 6** synthetic cases. Each case has 12 funds and the shortlist keeps 4 of 6 per sleeve. All mixes use 2 funds per sleeve and the same weight optimizer.

[Sampler comparison](results/samplers.csv) · [Three-score follow-up](results/tilts.csv)

| Follow-up sampler | Mean profiles found near the best score |
| --- | ---: |
| XY circuit, one score | 1.87 |
| XY circuit, three scores | 2.13 |
| Classical ranking | 2.33 |

Three fresh cases, five repeats each, 16 shared + 16 new portfolio evaluations per method. Profiles are yield and duration bins. In this table, “near” means within 15% of the gap between the best and median feasible cost.

[Main settings](pilot/protocol.json) · [Follow-up settings](pilot/tilt_protocol.json) · [XY mixer reference](https://arxiv.org/abs/1709.03489)
