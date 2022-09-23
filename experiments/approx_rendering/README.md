# Approximate rendering

To run the benchmark tests, setup a Python environment (`> 3.10`) with `poetry`.

Then, run:

```
poetry run pytest approx_rendering.py --benchmark-disable-gc --benchmark-warmup=on --benchmark-min-rounds=100
```
