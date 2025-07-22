# PerceSeg

## Install

```
conda activate -n perceseg python=3.8
conda activate perceseg
pip install -r requirements.txt
```

## Dataset

Please download the datasets from the official website and split them yourself.

Vaihingen

Postdam

LoveDA

## Training

```
python PerceSeg/train_supervision.py -c PerceSeg/config/vaihingen/perceseg.py
```


## Testing

```
python vaihingen_test.py -c PerceSeg/config/vaihingen/perceseg.py -o fig_results/vaihingen/perceseg-test --rgb -t lr
```
