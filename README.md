# Fake Job Posting Detector

A Samsung Innovation Campus classical ML project with a shared feature transformer,
TF-IDF text features, four base learners (LR, Linear SVC, random forest and XGBoost),
RandomOverSampler inside each stacking fold, and a Streamlit single/batch interface.

## Reproduce
Python 3.12 is the tested target. Create/activate `.venv`, then:
```bash
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
The included `fake_job_postings.csv` is the public job-posting dataset used in the project;
verify its upstream distribution terms before reuse. Training runs locally and can be CPU/memory
intensive (four learners and five-fold stacking). No paid API or custom foundation-model training is used.

`train_model.py` removes normalized exact duplicates before a fixed stratified split, rejects
conflicting labels, and writes the fitted model and `artifacts/metrics.json` with actual held-out
metrics, confusion matrix, package version, dataset/model checksums and split fingerprints.
The UI serves only a model matching that manifest. The legacy bundled model is retained unchanged
and is not the model measured here; run training first to regenerate both the local model and manifest. Previous fixed README/UI scores and the
uncommitted DistilBERT comparison are not represented as reproducible results.

This is a single development holdout, not independent nested model selection. Exact-duplicate
removal does not rule out semantically related postings across the split. The output is a statistical
risk score, not a calibrated fraud probability or a factual determination of fraud. Metadata guessed
from pasted text can differ from training metadata; use the detailed form when possible.

## Tests
`python -m pip install -r requirements-dev.txt`, then `python -m pytest -q`.
Tests check split disjointness, conflicting duplicates, feature consistency and fold-local pipeline structure.
Only load model files you trust; joblib/pickle can execute code. No project license grant is implied.

## Measured local run

On the committed dataset, 315 normalized duplicate rows were removed before splitting 17,565
unique postings into 14,052 training and 3,513 test rows (171 fraudulent). Fraud-class precision
was 0.8725, recall 0.7602, F1 0.8125 and average precision 0.8809. The confusion matrix was
`[[3323,19],[41,130]]`. This run took 460 seconds in this environment; runtime varies by machine.
The full provenance and checksum are in `artifacts/metrics.json`; these are single-holdout results,
not a promise of real-world performance.

## Reproduction check

A second full local run of this training/evaluation code reproduced all seven reported metrics,
the confusion matrix, dataset checksum and split fingerprints exactly. Serialized model bytes
were different; byte-for-byte artifact reproducibility is not claimed. The newly trained binary
is not included in this PR. GitHub Actions runs offline unit tests, not full training.
