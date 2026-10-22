# Pakistan tomato-leaf dataset and external evaluation

Source: [Tomato Leaf Disease Classification Dataset in Pakistan, release 2](https://data.mendeley.com/datasets/3mbnb82mxd/2), DOI **10.17632/3mbnb82mxd.2**. Authors: Mohammad Naeem Malik, Syed Hashir Ali and Hamid Ali. Licence: **CC BY 4.0**; see the [publisher's licence text](LICENSE.txt).

The source describes field photographs from Pakistan, labelled by visible symptoms. Acquisition location and diagnoses were not independently verified in this project. The downloaded archive contains **830 raw images plus 7,200 augmented copies**; its advertised 7,200-image count should not be treated as 7,200 independent original photographs.

| Raw source category | Raw images | Binary reference |
|---|---:|---|
| Healthy | 137 | Healthy |
| Early blight | 119 | Unhealthy |
| Late blight | 113 | Unhealthy |
| Leaf mold | 186 | Unhealthy |
| Septoria leaf spot | 195 | Unhealthy |
| Yellow leaf curl virus | 80 | Unhealthy |

All six category names are present in the PlantVillage portion of the CNN's training sources. The external evaluation measures correct Healthy/Unhealthy predictions by source category; it does not test disease-name classification.

## Download the source images

Download `Tomato_Leaf_Disease_Dataset_Pakistan.zip` from the [official release](https://data.mendeley.com/datasets/3mbnb82mxd/2). The recorded archive is 726,457,628 bytes with publisher SHA-256:

```text
2a2b36a8f972337f6d0e95e436835e55ffee5291fa07277157b40f0441f42f50
```

Alternatively, from the project folder with its full requirements installed:

```powershell
python inspect_matching.py
python download_matching.py
```

These tools use the [frozen official release metadata](source_metadata.json), inspect the ZIP directory and download only the 830 raw-image members. Each member's CRC is checked and each downloaded file receives a SHA-256 record. This selective download does not verify the complete archive's SHA-256 because the augmented members are not downloaded. Files are stored locally under the Git-ignored `data/3mbnb82mxd_unseen/` directory. The repository itself contains no source photographs.

## Audited external test

Before model inference, the audit excluded 357 raw entries from detected photo families overlapping earlier data and 12 additional duplicate/cropped/resized copies. The final cohort contains **461 images: 115 Healthy and 346 Unhealthy**. All augmented images were excluded.

The checks combine decoded-pixel hashes, perceptual/structural comparison and SIFT/homography/aligned-pixel checks for resized or cropped copies. The whole release includes reused photographs and cannot be called a clean independent unseen collection. Undetected overlap, acquisition dependence and reference-label errors can remain.

See the [full evaluation report](../../evaluation_unseen_matching/REPORT.md), [complete 830-row audit manifest](../../evaluation_unseen_matching/audit_manifest.csv), [predictions](../../evaluation_unseen_matching/predictions.csv) and [frozen protocol](../../evaluation_unseen_matching/protocol_before_predictions.json). Audit records preserve original paths as historical identifiers; downloading the images is unnecessary for recomputing saved metrics:

```powershell
python verify_results.py
```

Fresh overlap-audit reproduction additionally requires the earlier seven-source manifests/images, annotated parent photos and the local Agri-Vision4 audit referenced by the protocol. It also requires OpenCV (`python -m pip install -r audit-requirements.txt`). `prepare_matching.py`, `audit_matching_geometry.py` and `evaluate_matching.py` document the original sequence; completed evaluations are protected against overwrite. The saved source metadata, full audit decisions, geometric matches and predictions allow review without rerunning that sequence or downloading any photographs.
