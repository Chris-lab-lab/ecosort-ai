# Training results

The current result is Robust v2:

- Architecture: MobileNetV3-Small
- Parameters: 1,520,931
- Training data: 1,800 balanced images (600 per class)
- Validation: 40 original-only images
- Independent test: 40 original-only images
- Test accuracy: 100.0%
- Test macro-F1: 1.000
- Confusion matrix (`empty`, `half-full`, `full`): `[[8,0,0],[0,16,0],[0,0,16]]`
- Controlled transformed-view raw accuracy: 99.6% across 280 views
- Controlled transformed-view consistency: 99.6%

See `IMPROVEMENT_REPORT.md` for the complete dataset audit, old/new comparison, condition-level metrics, limitations, and deployment notes. The 40-image test split is small and comes from the same broad collection process; new physical deployment-session testing is still required.
