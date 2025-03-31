# Resume-JD Matching Demo

This is a lightweight demo version of a resume-job matching project using spaCy transformer-based NER and SBERT embedding similarity.

## Structure
- `scripts/`: Main scripts for inference and evaluation
- `data/sample/`: A few sample resumes and JDs
- `results/`: Model output and top-K matching
- `configs/`: Config files for spaCy training or inference

## How to Run
1. Run `scripts/ner_predictor.py` to extract entities
2. Run `scripts/similarity_matcher.py` to compute SBERT similarity
3. Run `scripts/eval_metrics.py` to evaluate ranking

See `scripts/` for implementation details.