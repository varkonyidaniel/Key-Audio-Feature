# Key-Audio-Feature

### venv

- env export :

```python -m pip freeze > requirements.txt```

- env create:

```python -m venv ./venv/```

```.\venv\Scripts\activate.bat``` (WIN)

```python -m pip install -r requirements.txt```


# Server/Singularity:
- sing image

```sudo singularity build /singularity/21_Peter/kaf.simg sing.recipe```

- move data (test data already there):
```scp -P 10024 -r DATA/WAV  peter.kiss@157.181.176.110:/Key-Audio-Feature/DATA/WAV```

modify permissions:
- chmod 777 init.sh
- chmod 777 exp_runner.sh

- preprocess:
```./init.sh```

