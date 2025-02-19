# Murtel Image Processing Pipeline

A suite of scripts to process and analyze **Murtel Rock Glacier** images, focusing on:
- **Weather filtering** for TIR (thermal infrared) and RGB images  
- **RGB spatial alignment** (SuperGlue-based)  
- **TIR-RGB overlay** or matching  

---

## Installation

1. **Clone** this repository

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Add SuperGlue Model**:
   Place the `models` directory from [magicleap](https://github.com/magicleap/SuperGluePretrainedNetwork) into `aligning` directory in order for the rgb aligning to work. Make sure to adhere to their License.

---

## Example Usage

Below are sample commands for each key functionality:

### 1. Filtering

#### a) RGB Filter
Trains an svm using labeled data:
```bash
python weather_filter/rgb_filter_model_training.py \
        data/RGB_images \
        data/filtering/rgb_labels.csv \
        --extracted_features data/filtering/all_extracted_features.csv \
        --train_features data/filtering/train_extracted_features.csv \
        --test_features data/filtering/test_extracted_features.csv \
        --train_csv data/filtering/train.csv \
        --test_csv data/filtering/test.csv \
        --model svm \
        --model_dir data/filtering \
        --re_extract
```
(This orchestrates feature extraction, splitting, model training and testing.)

To predict unlabeled data:
```bash
python weather_filter/rgb_filtering.py \
        data/RGB_DL_images \
        --scaler_path scaler.joblib \
        --model_path model.joblib \
        --output_features data/filtering/predictions.csv
```

#### b) TIR Filter
Trains a threshold entropy-based classifier on TIR images:
```bash
python weather_filter/tir_filter_train.py \
        data/TIR_images \
        data/filtering/tir_labels.csv \
        --outputs_csv data/filtering/tir_filtering.csv \
        --threshold_file data/filtering/tir_entropy_threshold.joblib
```
(This calculates entropies, selects and saves decision boundary.)


For classification on unlabeled TIR data:
```bash
    python weather_filter/tir_filtering.py \
        data/TIR_images \
        data/filtering/tir_entropy_threshold.joblib
```

---

## 2. Preprocessing

**Full Preprocess** example:
```bash
python preprocessing/full_tir_and_rgb_proproccess.py \
        fullmix \
        --data_dir ../data/align_test_dirs \
        --do_rgb \
        --do_tir
```
(Both rgb and tir data of a input dir with the directories `rgb` and `tir` is compressed, filtered and masked.
Several different strategies are available)

---

## 3. Alignment

**SuperGlue Workflow** (align a directory of images to the first as reference):
```bash
python aligning/superglue_workflow.py \
        data/align_test_dirs/fullmix \
        data/fullmix_aligned
```
(This script estimates an affine warp for each image to match the reference image and saves aligned images.)

---

## 4. TIR Matching

Using **cross-correlation** with template matching:

```bash
python tir_matching/xcorr_rgbtir.py \
        data/align_test_dirs/fullmix \
        --output_dir data/tir_overlays_fullmix

```
(This matches and overlays TIR images with RGB spatially)


## Other Functionalities

In the `utils` dir, bash helper scripts for image pairing and gif creation is found.

Labeling Helpers are in `labelers`. Some legacy code in `trial_code`.

The directory `aligning` also includes visualization code for SuperGlue operations as well as surface movement estimation.

---

## Acknowledgements

- **SuperPoint** and **SuperGlue** from [magicleap](https://github.com/magicleap/SuperGluePretrainedNetwork)
- The images were acquired in the framework of the PERMA-XT project run by University of Fribourg/GEOTEST/PERMOS.
- Special thanks goes out to the Project Partners Dr. Dominik Amschwand and Prof. Jan Beutel.

---

## Contact