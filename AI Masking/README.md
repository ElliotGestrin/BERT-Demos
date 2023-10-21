# AI Maskning
AI Masking requires python >= 3.10. While positioned in this folder the environment can then be configured with:
```
python3 -m venv env
source env/bin/activate
pip install -r requirments.txt
```

You can then deactivate this python environment with
```
deactivate
```

## Quickstart
For a quick demo, simply run `AIMasker.py` while using the environment setup above.
```
python AIMasker.py # For Swedish
python AIMasker.py eng # For English
```

You can also use an anonymising mode, which fills with automatic anonymous names.
```
python AIMasker.py anon # For Swedish
python AIMasker.py anon eng # For English
```

## GUI
To be designed