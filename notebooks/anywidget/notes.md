
To install a poetry kernel for jupyter labs:
```bash 
poetry run python -m ipykernel install --user --name genjax
```

To manage installed kernels:
```
jupyter kernelspec list
jupyter kernelspec uninstall genjax
```