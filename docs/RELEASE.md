# Generator v1.0: repository and archive correspondence

Software v1.0.1 was published on 3 November 2026. It completes the P1A software distribution while retaining Generator v1.0 and the published dataset v1.0.0.

## Published software

- GitHub release: https://github.com/kodendaal/ShipSpace-GA/releases/tag/v1.0.1
- Release commit: `0839ba20688771099de47f09f0a19db4b96db057`
- Version DOI: https://doi.org/10.5281/zenodo.23913344
- Software concept DOI (all versions): https://doi.org/10.5281/zenodo.23196764
- Original software v1.0.0 DOI: https://doi.org/10.5281/zenodo.23196765
- Original v1.0.0 tag commit: `bc7c37a150da2ee730da6a6cf2760354e83cf618`

## Corresponding dataset and paper

- Dataset version: v1.0.0, unchanged by the software correction.
- Dataset version DOI: https://doi.org/10.5281/zenodo.23197299
- Dataset concept DOI (all versions): https://doi.org/10.5281/zenodo.23197298
- Paper: https://doi.org/10.48550/arXiv.2610.09585

The dataset record retains its original software v1.0.0 provenance. The software v1.0.1 link identifies the completed software distribution; it does not imply that the archived dataset was regenerated. Software and dataset versions are independent.

## Changes in software v1.0.1

- Restore the existing engineering package for calibrated mass, LCG/KG, adjacency, hydrostatics, trim and free-surface-corrected GM evaluation.
- Include calibration, native-geometry adapters, the CSV/JSONL runner and 11 existing engineering tests.
- Align the README, schema documentation, physics guide and paper citation.
- Use Generator v1.0 in current comments and output version labels.

Numerical generation rules, tensor layouts, hull masks and archived reference data are unchanged. See [VALIDATION.md](VALIDATION.md) for the integration checks and their limits. These checks do not constitute a rerun of the paper experiments.

## Archive verification

On 3 November 2026, the new Zenodo record and DataCite registry identified software v1.0.1 as a version of the existing software concept DOI. The GitHub tag resolved to the merged correction commit above.

The downloaded Zenodo ZIP contained all 68 expected repository files. Every file matched the prepared and reviewed repository; there were no missing, differing or extra files. Its MD5 matched the Zenodo manifest: `c5d48a45445f37c8da26f68147968948`.

## Citation and later documentation edits

Cite the version DOI for the software actually used and the dataset DOI for the data used. The concept DOIs identify the respective version families.

The software DOI was minted after publication. Later documentation commits add it to CITATION.cff and README and update this release record. Those commits are not part of the archived v1.0.1 source snapshot. Keep the published v1.0.0 and v1.0.1 tags and archived files fixed.

The code is MIT-licensed. Bundled data is CC BY 4.0, as stated in the repository README. The current Zenodo software record identifies both licenses; its description states their scopes. The separate dataset record remains CC BY 4.0.

## Software, dataset and paper links

The published Zenodo records were checked on 3 November 2026. The software record links to its GitHub v1.0.1 tag, the dataset and the paper. The dataset links to both the original software v1.0.0 and the completed software v1.0.1 distribution, and to the paper.

The reciprocal software-dataset relations preserve the existing convention: software is a supplement to the dataset, and the dataset is supplemented by the software. Each record's version-family relation is managed by Zenodo.
