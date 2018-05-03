# Kaggle - Fashion Identification

- Different saturations
- Levels of Occulusion (objects overlapping)
- Need to download/convert from JSON to CSV
- Color Analysis
- Proper masking of target within image ?

## Getting Started

```
pip install -r requirements.txt
```

### Downloading

```
python3 downloader.py data/<train.json|validation.json|test.json> data/<output_dir/>
```


### Folder Structure

```
data/
    train.json
    test.json
    validation.json
    train/
    test/
    validation/
```