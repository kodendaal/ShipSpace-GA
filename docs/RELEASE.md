# Generator v1.0: repository and archive correspondence

This correction completes the P1A repository with the existing engineering
package and aligned documentation. **Generator v1.0** remains the generator
name. **v1.0.1** is the proposed software packaging correction.

## Current correspondence

Checked against GitHub and DataCite's registered DOI metadata on 3 November 2026:

| Item | Identifier | Status |
|---|---|---|
| Generator | Generator v1.0 | Numerical rules retained |
| Existing GitHub release | `v1.0.0` → `bc7c37a150da2ee730da6a6cf2760354e83cf618` | Published 6 October 2026; preserve this tag |
| Main before this correction | `ef38c461f64f44630d9372ab72235a1986e3feb3` | Three later commits, changing README/CITATION only |
| Software v1.0.0 DOI | [10.5281/zenodo.23196765](https://doi.org/10.5281/zenodo.23196765) | Registered version v1.0.0; links to GitHub tag v1.0.0 |
| Software concept DOI | [10.5281/zenodo.23196764](https://doi.org/10.5281/zenodo.23196764) | All software versions; verified through the version record's IsVersionOf relation |
| Dataset v1.0.0 DOI | [10.5281/zenodo.23197299](https://doi.org/10.5281/zenodo.23197299) | Registered version v1.0.0; retain files and identifier |
| Dataset concept DOI | [10.5281/zenodo.23197298](https://doi.org/10.5281/zenodo.23197298) | All dataset versions; verified through the version record's IsVersionOf relation |
| P1A paper | [arXiv:2610.09585](https://arxiv.org/abs/2610.09585) | 2026 preprint |
| Proposed software correction | `v1.0.1` | Not tagged or archived yet |

A GitHub tag is a fixed source checkpoint. Later commits on `main` are normal
and do not update that checkpoint. The old tag should not be moved to include
this correction.

## Complete the release in order

1. Review and merge the P1A correction. Run the included 11 engineering tests.
   Record the exact final commit used for the release.
2. Open the existing software record in Zenodo and check whether ShipSpace-GA
   is still enabled in the GitHub integration. Use the software concept DOI
   above to keep the new archive in the existing version family.
3. Update release metadata, then create GitHub release **v1.0.1** from the
   final commit. Use the release-note text below. Keep Generator v1.0 as the
   generator name.
4. Archive it as a **new version of the existing software record**. If the
   repository is already linked, let the GitHub integration process the
   release. If the original record was a manual deposit, use **New version**
   on that record and upload the exact tagged source archive. Choose one route;
   do not create a second unrelated software deposit.
5. Check that the new Zenodo record's version, source archive and GitHub link
   match `v1.0.1`. Add its DOI to the GitHub release notes and update the current
   README citation links. Leave the dataset archive unchanged.

The software registry record links explicitly to GitHub `tree/v1.0.0`; the
software and dataset records also link to each other. This confirms the
registered correspondence. Zenodo file bytes and the current account
integration settings were not accessible, so archive-byte equality and the
current automatic-deposit setting were not checked. The concept DOIs above
come from registered relations, not inferred neighbouring record numbers.

`CITATION.cff` identifies candidate software v1.0.1 and the P1A paper. Its
top-level software DOI is absent until v1.0.1 receives its own identifier;
the v1.0.0 DOI must not be labelled as a newly archived version.
For a manual Zenodo version, reserve its DOI before making the final tag if
the DOI must be embedded in that tagged source. For automatic archiving, keep
the tag fixed and add the minted DOI to the release description and a later
documentation commit. That later commit is not itself the archived checkpoint.

## Release notes for v1.0.1

ShipSpace-GA v1.0.1 completes the P1A software distribution for Generator v1.0.

- Restore the existing `src/engineering/` package, including the P1A mass,
  adjacency, hydrostatics, trim and free-surface calculations, calibration,
  native-geometry adapters and CSV/JSONL runner.
- Include the 11 existing engineering tests and October input-validation fixes.
- Align the README, dataset schema, physics guide and paper citation.
- Use Generator v1.0 in current source labels and documentation. The producer
  and bundle labels on newly generated files are updated; numerical generation,
  tensor layouts, stored datasets and archived labels are unchanged.

This is a software packaging correction. The original v1.0.0 tag and published
50K dataset remain historical references. The included integration checks do
not constitute a rerun of the paper experiments.

At publication, append the final commit, new software version DOI, verified
software concept DOI, unchanged dataset DOI and validation result.

## Verification sources

- [Software DOI registry metadata](https://api.datacite.org/dois/10.5281/zenodo.23196765)
- [Dataset DOI registry metadata](https://api.datacite.org/dois/10.5281/zenodo.23197299)
- [Existing GitHub release](https://github.com/kodendaal/ShipSpace-GA/releases/tag/v1.0.0)

## Official guidance

- [Zenodo DOI versioning](https://support.zenodo.org/help/en-gb/1-upload-deposit/97-what-is-doi-versioning)
- [Manage versions](https://help.zenodo.org/docs/deposit/manage-versions/)
- [Archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/)
