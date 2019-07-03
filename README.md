# LSTM time sequence prediction

LSTM time sequence prediction is a Python sources for dealing with n-dimension periodic signals prediction

## Installation

```bash
python install setup.py
```

## Usage

```python
from lstm_time_series_prediction.api import TimeSeriesPrediction


model = TimeSeriesPrediction()
model.train(...)
...

xi = ...
yi = model.inference(xi)

```
