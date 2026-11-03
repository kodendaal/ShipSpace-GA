# P1A integration checks — 3 November 2026

Base repository commit: `ef38c461f64f44630d9372ab72235a1986e3feb3`.
The restored engineering package includes the 24 October adapter corrections.

| Check | Result |
|---|---|
| Existing engineering suite | 11 tests passed |
| Original kernel/calibration provenance | All 11 original file hashes matched the recovered source archive |
| Packaged numerical implementation | Unchanged executable syntax trees; two version-name comment/docstring edits recorded separately |
| Existing generator/validation source | All 21 Python files retained the same executable syntax trees after excluding docstrings and the two output version-label constants |
| Hull masks and real-reference data | All tracked data files byte-identical to the base commit |
| CSV and JSONL batch commands | Synthetic fixture evaluated successfully with equilibrium enabled |
| Invalid batch input | Error row retained; exit code 1 |
| Citation | CFF 1.2.0 schema validation passed |
| Documentation | Relative file links resolve |

Validation environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and
NetworkX 3.7. PyTorch/PyG loading and a fresh full generator installation were
not exercised by these integration checks.

Run the retained suite from the repository root:

```bash
python -m unittest discover -s tests -p "test_engineering_dropin.py" -v
```

The batch smoke input uses the same synthetic fixture defined in that test
file. It is an integration check, not measured ship data. The paper experiments
and full published corpus were not rerun or regenerated.

The source comparison allows only these output-label changes in existing
executable code: `GENERATOR_VERSION` becomes `v1.0`, and `BUNDLE_SCHEMA` becomes
`shipspace_ga_v1_enc2_reference_bundle_1`. This renames new output metadata;
it does not modify archived files or tensor layouts.
