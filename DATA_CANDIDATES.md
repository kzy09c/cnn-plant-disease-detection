# Dataset inspection notes

The dataset descriptions below were checked against primary release pages and downloaded metadata on 22 October 2026. No candidate is silently relabelled or added merely to increase image counts.

- **PlantVillage, Bangladesh, TLID, PlantDoc:** existing local releases and audited source manifests were available. Original bytes were checked against recorded SHA-256 hashes. Earlier data and models are preserved.
- **Tomato-Village multiclass variant:** 4,525 photographs/crops, 536 MiB, downloaded and checked against Git blob SHA values. Random training previews show detached leaves on plain paper. Although the authors describe field collection, this release does not provide the natural-background diversity that its name alone might suggest. Multiple crops from original photos cross the published folders; family grouping is required.
- **[GTLD](https://data.mendeley.com/datasets/2bdfjb99k5/1):** the inspected public release metadata contains only seven JPEG examples, one of each category. It was not used as a large training dataset or counted as substantial independent validation.
- **[Tomato Leaf Dataset, bpfd9cns5g V2](https://data.mendeley.com/datasets/bpfd9cns5g/2):** metadata contains 1,420 JPEG entries and 689 TXT annotations. Many entries have identical publisher checksums, consistent with original and exported copies. The metadata inspected did not supply a verified numerical class-to-name mapping. Empty detection annotations cannot be assumed to mean Healthy; the release was not merged without that mapping.
- **[Tomato-Disease, Zenodo 15868289](https://zenodo.org/records/15868289):** release metadata describes 2,212 tomato disease images with nine categories including Healthy. CC BY 4.0. Archive size 544,524,808 bytes; publisher MD5 `f41cf311fdad64d87f5d41a89be2ccfb`. Download and annotation inspection are tracked separately before inclusion. Captions and annotations, not model predictions, govern labels.

`data/sources/` retains the exact inspected metadata. Availability does not establish independent acquisition, reliable diagnosis or representative internet coverage.
