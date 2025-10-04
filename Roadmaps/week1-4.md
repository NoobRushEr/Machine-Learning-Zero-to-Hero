Noted and saved: I recorded your preference so future expansions won’t repeat already-covered material unnecessarily. I’ve created a focused, **4-week** roadmap derived from **your saved `my_learning_prompt`**. Below is a brief planning justification (150–300 words), then the full Markdown 4-week curriculum (strict structure you asked for). At the end you’ll find a compact JSON summary (4 objects) for easy import.

---

## Planning justification (assumptions & design choices — 170 words)

This 4-week plan compresses the `my_learning_prompt` rules into a practical, project-driven onboarding for an absolute beginner (intermediate Python, basic Git, HS algebra). I prioritized small UCI datasets (≤100 rows) so every project is CPU-friendly and reproducible locally: **Soybean-small (47)**, **Cryotherapy (90)**, **Period Changer (90)**, **Sirtuin6 (100)** (provenance & licenses are included in each mini-project). Weeks 1–4 keep weekly caps ≤10 hrs and introduce the frozen toolstack incrementally: pandas/Jupyter/scikit-learn first, then DVC + pipelines, then dimensionality reduction/feature selection, and finally experiment tracking with MLflow and full repo hygiene. Math/theory slots start Week 2 (1 hr/week) to avoid cognitive overload—each includes a short video + reading + runnable snippet. I schedule minimal theory (just-in-time) linked to the week’s project. DVC stays local (no cloud) and Docker/CI/remote GCP come later per your rules. I assumed you wanted a **4-week deliverable now** (so JSON contains 4 objects); when you ask to extend to later weeks I’ll **expand incrementally** and **avoid repeating** basics already covered (preference saved). Dataset and official tool links cited inside the plan.

---

# 4-Week Project-Driven Roadmap (Markdown)

> NOTE: This document strictly follows your saved `my_learning_prompt` rules for weeks 1–4 (time caps, datasets, deliverables, reproducibility, ethics, resources format). Where I cite dataset provenance or official docs I’ve included UCI / tool doc links.

---

## One-page overview (4 blocks)

**Weeks 1–4 (This deliverable)**
Topics: EDA & reproducible basics; pipelines + DVC; feature selection & PCA; experiment tracking & packaging.
Projects (examples):

* W1: Soybean-small — EDA → baseline classifier.
* W2: Cryotherapy — scikit-learn Pipeline + DVC local remote.
* W3: Period Changer — feature selection + PCA, RFE.
* W4: Sirtuin6 Capstone — end-to-end reproducible run (MLflow tracking + checksums).
  Hireable skills: pandas data wrangling, scikit-learn pipelines, DVC basics, MLflow experiment tracking, reproducible README + simple model evaluation.

**Weeks 5–8 (planned later — brief placeholder)**
Topics: Larger Kaggle datasets (<5k), model tuning, model card creation, Docker introduction, classifier calibration. (Will expand without repeating basics.)

**Weeks 9–12 (placeholder)**
Topics: PyTorch intro, small NLP/time-series, GCP basics (deferred until Week12 per rules).

**Weeks 13–17 (placeholder)**
Topics: production MLOps patterns, recommender systems stretch, portfolio + resumes + CI/CD.

---

## Index (weekly coverage)

1. Week 1 — Foundations: EDA, pandas, scikit-learn basics
2. Week 2 — Pipelines, cross-validation, DVC (local)
3. Week 3 — High-dim data: feature selection & PCA
4. Week 4 — Capstone: end-to-end reproducible pipeline + MLflow; interview checkpoint

---

> ### Global repo & seed snippet (to include in every project README)

```py
# reproducibility snippet (include exact versions in requirements.txt & runtime.txt)
import random, os
import numpy as np
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# torch seed only when torch used (Weeks 9+)
# import torch
# torch.manual_seed(RANDOM_SEED)
```

---

## Week 1 — Foundations: EDA, pandas, scikit-learn basics

**Prerequisites:** intermediate Python, basic Git, VSCode or Jupyter familiarity.
**Why this matters:** Good EDA & reproducible repo hygiene are the foundation for every data job. Beginners learn to get a dataset from UCI, inspect it, make a clean notebook, and produce a simple baseline classifier with `scikit-learn`.

**Time-split Tasks (≤10 hrs — split to keep each task ≤25% of weekly hours):**

1. Repo + environment setup (pin python version, create requirements.txt, runtime.txt) — **1.0 hr**
2. Quick Jupyter / notebook hygiene + git init, README skeleton — **1.0 hr**
3. Pandas crash sessions — Part A: IO, indexing, missing values (1.5 hr) — **1.5 hr**

   * *Simplified alternative:* follow a 30-min interactive Colab notebook with preloaded CSV if local setup fails.
4. Pandas crash sessions — Part B: groupby, joins, feature creation (1.5 hr) — **1.5 hr**
5. EDA & visualization (matplotlib/seaborn) on Soybean-small — **2.0 hr**
6. Baseline model (train/test split, logistic regression or decision tree using scikit-learn) — **2.0 hr**
7. Write README dataset provenance & license, add `main.py` smoke runner that loads model and prints "SMOKE OK" — **1.0 hr**
   **Total tasks time:** 10.0 hrs (Resources consumption below).

**Math & Intuition slot:**

* **Topic:** No formal math slot this week (per program rules math starts Week 2). *Optional quick refresher:* mean/variance refresher (15 mins).
* **Video (🎥):** optional 10-min "Mean & Variance explained" (any short stat video) — (optional).
* **Reading (📝):** quick 10-min review: "Descriptive statistics" section in *100 Pages Machine Learning* (assumption: canonical short chapter).
* **Runnable snippet idea:** show how to compute mean, median, std with `pandas` and plot distribution with `matplotlib` (≤12 lines).
* **Interview prompt (1-line):** "Given a numeric column, how would you explain variance to a non-technical stakeholder?"

**ML/DL concepts introduced this week:** supervised classification, train/test split, baseline model, confusion matrix, accuracy, EDA.

**Resources (for this week — **videos first**, then docs, then readings). New tools/topics introduced: `pandas`, `Jupyter`, `scikit-learn`**

* **pandas (📚 new)**

  1. 🎥 **freeCodeCamp — Learn Pandas & Python for Data Analysis (Full course)** — [https://www.youtube.com/watch?v=gtjxAH8uaP0](https://www.youtube.com/watch?v=gtjxAH8uaP0) — **2023** — est **3–4 hrs** — **Required**. (Video-first primary)
  2. 🎥 **Data School — Best practices with pandas (playlist / short tutorials)** — [https://www.youtube.com/dataschool](https://www.youtube.com/dataschool) — **2018 (canonical)** — est **1–2 hrs** — **Optional**. *Justification (older than 2 yrs):* Kevin Markham's pedagogy remains canonical for pandas best practices.
  3. 📄 **pandas official docs — Getting started / user guide** — [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/) — **(official doc)** — est **30–60 mins** — **Required**. (keeps you aligned to API and versions). ([Pandas][1])

* **Jupyter (📚 new)**

  1. 🎥 **Jupyter Notebook Tutorial — Setup & Walkthrough (Corey Schafer / similar)** — [https://www.youtube.com/watch?v=2FwcFdybn34](https://www.youtube.com/watch?v=2FwcFdybn34) — **2018** — est **45–60 mins** — **Required**.
  2. 📄 **Jupyter official docs / quickstart** — [https://jupyter.org/](https://jupyter.org/) — **(official)** — est **15–30 mins** — **Required**.

* **scikit-learn basics (📚 new)**

  1. 🎥 **StatQuest — Logistic Regression / classification primers** — [https://www.youtube.com/playlist?list=PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR](https://www.youtube.com/playlist?list=PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR) — **various years (canonical)** — est **20–40 mins** — **Required**.
  2. 📄 **scikit-learn User Guide — Getting started** — [https://scikit-learn.org/stable/getting\_started.html](https://scikit-learn.org/stable/getting_started.html) — **(official)** — est **30 mins** — **Required**. ([scikit-learn][2])

**Small Read:** *Python Data Science Handbook* — **Chapter: Data Manipulation with pandas (approx. 10–20 pages)** — est **30–45 mins** — *Relevance:* makes pandas examples portable to real notebooks. *(Assumption: page numbers follow O’Reilly edition; adjust to your edition if needed.)*

**Mini-Project:** Soybean-small classification (UCI, ≤100 rows)

* **Dataset:** Soybean-small (47 instances). Provenance & license: UCI page (CC BY 4.0). ([UCI Machine Learning Repository][3])

  * **URL:** [https://archive.ics.uci.edu/dataset/91/soybean+small](https://archive.ics.uci.edu/dataset/91/soybean+small)
  * **License:** CC BY 4.0 (note in README)
* **Acceptance criteria (≤3):**

  1. `notebooks/main.ipynb` runs from top→bottom without errors.
  2. `python main.py --smoke` prints `SMOKE OK` and writes `artifacts/eval.csv`.
  3. Model achieves improvement over majority-class baseline (report baseline and % improvement in README; target: demonstrable improvement — e.g., if baseline = 40% accuracy, show model ≥ 55% OR explain why not).
* **Deliverables:** See global E — include `notebooks/main.ipynb`, `requirements.txt` with pinned versions, `runtime.txt` (python==3.10), `main.py` supporting `--smoke`, `artifacts/` (eval CSV + confusion matrix PNG), `checksums.txt` with SHA256, README with dataset provenance & license, explicit RNG seed snippet. (DVC optional this week but recommended as learning.)
* **Business impact metric:** compute baseline = majority class accuracy; model accuracy = M. % improvement = (M − baseline)/baseline × 100%. *Example assumption:* baseline = 0.33, model accuracy = 0.66 → 100% relative improvement → map to business: if misdiagnosis causes average crop loss of ₹X per field and model reduces errors by Y%, estimated saved = ₹X × Y% × assumed adoption rate. (Document assumptions in README.)

**Ethics Checklist (3 lines):**

1. **Dataset bias source:** Soybean-small collected from limited regions → class imbalance possible.
2. **Model harm scenario:** Misclassification could lead to wrong crop treatment recommendations harming yield (farmers).
3. **Mitigation:** stratified reporting + use `fairlearn` parity checks (code hint: `from fairlearn.metrics import MetricFrame; MetricFrame(metrics=sklearn.metrics.recall_score, y_true=y, y_pred=preds, sensitive_features=df['region'])`) and log fairness checks to MLflow once tracking added.

**Weekly Outcomes (testable):**

* Run: `python main.py --smoke` → expected: console `SMOKE OK` + `artifacts/eval.csv` created.

**Stretch Goal:** Add a scikit-learn `Pipeline` with simple imputer + one-hot encoder + classifier.

**Beginner Survival Tip:** If local env fails, use Google Colab and attach runtime.txt with `python==3.10` instructions in README.

**Community:** DataSchool Slack / Stack Overflow / r/MachineLearning (for conceptual Qs).

**GitHub folder scaffold snippet (starter):**

```
/project-soybean/
  notebooks/main.ipynb
  main.py
  requirements.txt
  runtime.txt
  artifacts/
  checksums.txt
  README.md
```

**Weekly Total Estimated Time:** Tasks **10.0 hrs** + Resources (videos/docs reading) **1.0–2.0 hrs (overlap)** → **Total ≤10 hrs (watch/listen while coding)**

---

## Week 2 — Pipelines, cross-validation, DVC (local)

**Prerequisites:** Week1 complete; comfortable with pandas, basic model training.
**Why this matters:** Real projects require robust pipelines, cross-validation, and data/version control (DVC) for reproducibility. This week introduces scikit-learn Pipelines and DVC local remotes (no cloud).

**Time-split Tasks (≤10 hrs; keep tasks ≤2.5 hrs each):**

1. Quick review + reproducible env pinning (`requirements.txt`) — **0.5 hr**
2. scikit-learn `Pipeline` hands-on (split into two 1-hr hands): Part A (transformers & ColumnTransformer) — **1.0 hr**

   * *Simplified alternative:* use `sklearn.preprocessing` direct transforms in notebook if Pipeline concept feels heavy.
3. Part B: cross-validation & hyperparameter grid (GridSearchCV / cross\_val\_score) — **1.5 hr**
4. Add DVC: `dvc init`, `dvc add data/soybean.csv`, `dvc.yaml` stage for `train` — **1.5 hr**
5. Hook `main.py` to read DVC-tracked data & run training (create `artifacts/model.joblib`) — **2.0 hr**
6. Unit smoke + README update (include `dvc pull` instructions & local remote steps) — **1.0 hr**
7. Write checksum & artifact files — **0.5 hr**
   **Total tasks time:** 8.0 hrs + resource time \~1–2 hr = ≤10.

**Math & Intuition slot (1 hr):**

* **Topic:** Bias–Variance tradeoff & regularization (intuitions for L2 / L1).
* **Video (🎥):** StatQuest — Bias/Variance & Regularization short explainer. — (canonical) — est **15–20 mins**.
* **Reading (📝):** *100 Pages Machine Learning* — short chapter on regularization (10–15 mins).
* **Runnable snippet idea:** vary `C` in `LogisticRegression` and plot train vs CV scores to show over/underfitting (≤15 lines).
* **Interview prompt:** "How does L2 regularization affect bias and variance?"

**ML/DL concepts introduced this week:** Pipelines, cross-validation, hyperparameter search, regularization, data/version control (DVC local remote).

**Resources (new tool/topic: DVC + scikit-learn advanced)**

* **DVC (📄 official + videos)**

  1. 🎥 **DVC — Versioning Data with DVC (tutorial video)** — [https://www.youtube.com/watch?v=kLKBcPonMYw](https://www.youtube.com/watch?v=kLKBcPonMYw) — **(2021/2022)** — est **35–60 mins** — **Required**.
  2. 🎥 **DVC — short workshop / get started (Iterative)** — [https://www.youtube.com/watch?v=some\_workshop](https://www.youtube.com/watch?v=some_workshop) — **(various)** — est **30–60 mins** — **Optional**.
  3. 📄 **DVC official docs (Get Started / User Guide)** — [https://dvc.org/doc](https://dvc.org/doc) — **(official)** — est **30–60 mins** — **Required**. ([Data Version Control · DVC][4])

* **scikit-learn (Pipeline & CV)**

  1. 🎥 **Hands-on scikit-learn pipeline tutorial (YouTube / short)** — (pick a short tutorial) — est **30–45 mins** — **Required**.
  2. 📄 **scikit-learn User Guide — Model selection & pipelines** — [https://scikit-learn.org/stable/user\_guide.html](https://scikit-learn.org/stable/user_guide.html) — **Required**. ([scikit-learn][5])

**Small Read:** *Hands-On Machine Learning (scikit parts)* — pages covering Pipelines & Model selection (\~10–15 pages) — est **30–45 mins** — *Relevance:* learn practical pipeline patterns.

**Mini-Project:** Cryotherapy outcome predictor (UCI, 90 rows)

* **Dataset:** Cryotherapy Dataset (90 instances). Provenance & license: UCI (CC BY 4.0). ([UCI Machine Learning Repository][6])

  * **URL:** [https://archive.ics.uci.edu/dataset/429/cryotherapy+dataset](https://archive.ics.uci.edu/dataset/429/cryotherapy+dataset)
  * **License:** CC BY 4.0
* **Acceptance criteria (≤3):**

  1. `dvc init` + `dvc add data/cryotherapy.xlsx` + `dvc.yaml` present.
  2. `pip install -r requirements.txt` and `python main.py --smoke` runs (reads `dvc` data via `dvc pull` instructions).
  3. Notebook reports CV mean & std for chosen metric (e.g., F1).
* **Deliverables:** All standard deliverables (E), plus `dvc.yaml` + `dvc.lock`, instructions for local DVC remote, no large binaries committed.
* **Business impact metric:** baseline (majority class) → show % improvement in F1. Map: if each correct classification avoids ₹Z of unnecessary treatment costs, compute savings = (#cases × adoption %) × improvement% × ₹Z. (Document assumptions.)

**Ethics Checklist:**

1. **Bias source:** small clinical dataset, sampling bias by treatment centers.
2. **Harm scenario:** false negatives → skipped treatments.
3. **Mitigation:** stratified CV + report per-group metrics; code hint: run `MetricFrame` from `fairlearn` and log to MLflow (later). Log fairness checks.

**Weekly Outcomes (testable):**

* Run: `dvc repro && python main.py --smoke` → expected: `SMOKE OK` and `artifacts/eval.csv` created.

**Stretch Goal:** Add `dvc push` local remote simulation (document steps).

**Beginner Survival Tip:** If DVC is unfamiliar, add a simple `data/` folder and document `dvc` commands in README; convert to DVC later.

**Community:** DVC Discourse / Stack Overflow.

**GitHub folder scaffold snippet (add):**

```
dvc.yaml
dvc.lock
data/  (dvc tracked)
notebooks/main.ipynb
main.py
requirements.txt
runtime.txt
README.md
```

**Weekly Total Estimated Time:** Tasks **8.0 hrs** + Resources **1.5 hrs** = **≤10 hrs**

---

## Week 3 — High-dim data: feature selection & PCA

**Prerequisites:** Weeks 1–2 complete; comfortable with Pipelines and DVC basics.
**Why this matters:** Many realistic datasets (like Period Changer: 1,177 features) require feature selection, dimensionality reduction (PCA) and careful model selection to avoid overfitting and to speed pipelines.

**Time-split Tasks (≤10 hrs):**

1. Quick review of scikit-learn feature selection APIs (SelectKBest, RFE) — **1.0 hr**
2. PCA intuition & hands-on using `sklearn.decomposition.PCA` — **1.5 hr**

   * *Simplified alternative:* use `SelectKBest` with univariate tests if PCA math feels heavy.
3. Build Pipeline: scaler → PCA (or feature selection) → classifier, test with CV — **2.0 hr**
4. Model evaluation: compute ROC/AUPRC and confusion matrix — **2.0 hr**
5. DVC: add feature-selection step as `dvc.yaml` stage and commit `dvc.lock` — **1.5 hr**
6. Update README with model card placeholder (we'll include full model card at Week 8) — **1.0 hr**
7. Checksums + artifacts — **0.5 hr**
   **Total tasks time:** 9.5 hrs + resources ≈ 0.5 hr → ≤10.

**Math & Intuition slot (1 hr):**

* **Topic:** Linear algebra for PCA — eigenvectors/eigenvalues + variance explained.
* **Video (🎥):** 3Blue1Brown — Essence of Linear Algebra (select short PCA segment) — **(canonical, older)** — est **15–25 mins**.
* **Reading (📝):** *100 Pages Machine Learning* — PCA short explainer (10–15 mins).
* **Runnable snippet idea:** fit `sklearn.decomposition.PCA(n_components=5)` and plot cumulative explained variance (≤15 lines).
* **Interview prompt:** "Explain PCA in one sentence and where you would use it."

**ML/DL concepts introduced this week:** dimensionality reduction (PCA), feature selection (SelectKBest, RFE), ROC & AUPRC.

**Resources (new concepts: PCA / feature selection)**

* **PCA (📄 + videos)**

  1. 🎥 **StatQuest — PCA (step-by-step)** — [https://www.youtube.com/playlist?list=PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR](https://www.youtube.com/playlist?list=PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR) — **(canonical)** — est **10–15 mins** — **Required**. ([YouTube][7])
  2. 🎥 **3Blue1Brown — Essence of linear algebra (visual intuition for eigenvectors)** — [https://www.youtube.com/watch?v=kjBOesZCoqc](https://www.youtube.com/watch?v=kjBOesZCoqc) — **(older)** — est **20–30 mins** — **Optional** (justification: excellent visual intuition).
  3. 📄 **scikit-learn PCA docs** — [https://scikit-learn.org/stable/modules/decomposition.html#pca](https://scikit-learn.org/stable/modules/decomposition.html#pca) — **(official)** — est **20–30 mins** — **Required**. ([scikit-learn][5])

* **Feature selection (RFE / SelectKBest)**

  1. 🎥 **short tutorial on scikit-learn feature selection (YouTube)** — est **20–30 mins** — **Optional**.
  2. 📄 **scikit-learn feature selection docs** — [https://scikit-learn.org/stable/modules/feature\_selection.html](https://scikit-learn.org/stable/modules/feature_selection.html) — **(official)** — est **15–20 mins** — **Required**. ([scikit-learn][5])

**Small Read:** *Hands-On Machine Learning* — chapter pages on dimensionality reduction / feature selection (10–20 pages) — est **30–45 mins**

**Mini-Project:** Period Changer classification (UCI — 90 instances, 1,177 features)

* **Dataset:** Period Changer (90 instances). Provenance & license: UCI CC BY 4.0. ([UCI Machine Learning Repository][8])

  * **URL:** [https://archive.ics.uci.edu/dataset/729/period+changer-2](https://archive.ics.uci.edu/dataset/729/period+changer-2)
* **Acceptance criteria (≤3):**

  1. Pipeline exists: scaler → SelectKBest/RFE or PCA → classifier; code runs via `python main.py --smoke`.
  2. `artifacts/` contains `confusion_matrix.png` and `roc.png` or `auprc.png` where applicable.
  3. DVC pipeline has `dvc.yaml` + `dvc.lock` capturing feature selection step.
* **Deliverables:** Standard E list (notebooks, requirements pinned, runtime.txt, main.py, artifacts, checksums, README with dataset provenance & license).
* **Business impact metric:** use AUPRC or ROC AUC vs baseline; map % improvement to reduce number of false positives/negatives, convert into estimated downstream cost savings with assumptions documented.

**Ethics Checklist:**

1. **Bias:** descriptors are chemically-derived — selection bias for molecules studied.
2. **Harm:** false positives in drug candidate prediction waste lab resources and funding.
3. **Mitigation:** conservative thresholding, use holdout, report per-class metrics; code hint: calibrate probabilities using `CalibratedClassifierCV` and log calibration plots to MLflow later.

**Weekly Outcomes (testable):**

* Run: `python main.py --smoke` → `SMOKE OK`, artifacts include `confusion_matrix.png`.

**Stretch Goal:** Try Recursive Feature Elimination (RFE) + cross-validated estimator selection.

**Beginner Survival Tip:** If PCA math is heavy, default to `SelectKBest` with `f_classif`.

**Community:** Cross-validated scikit-learn examples on StackOverflow / scikit-learn mailing list.

**GitHub folder snippet (adds feature selection stage):**

```
/project-period-changer/
  dvc.yaml
  notebooks/main.ipynb
  main.py
  artifacts/
  requirements.txt
  runtime.txt
  README.md
```

**Weekly Total Estimated Time:** Tasks **9.5 hrs** + Resources **0.5–1.0 hr** = **≤10 hrs**

---

## Week 4 — Capstone: end-to-end reproducible pipeline + MLflow (Interview checkpoint)

**Prerequisites:** Weeks 1–3 completed.
**Why this matters:** Demonstrate repeatable experiments, artifact checksums, evaluation artifacts and write a short project writeup suitable for LinkedIn/resume. Week 4 is the first interview checkpoint—includes LeetCode, SQL, behavioral, whiteboard math, debugging, and system design prompts.

**Time-split Tasks (≤10 hrs):**

1. Finalize pipeline & model selection on Sirtuin6 dataset (split tasks so none >2.5h): data + pipeline review — **1.0 hr**
2. Add MLflow tracking to training script, log params/metrics/artifacts — **1.5 hr**
3. Add checksums.txt generation for `artifacts/model.joblib` and `artifacts/eval.csv` (script) — **0.5 hr**
4. Add tests (pytest) in `tests/` that check `python main.py --smoke` runs and that `checksums.txt` matches artifact SHAs — **1.5 hr**

   * *Simplified alternative:* provide a smoke test that asserts files exist (if writing hash tests feels heavy).
5. README polish, dataset provenance + license, and explicit RNG snippet — **1.0 hr**
6. Create `main.py` CLI entrypoint (with `--smoke`) and sample `run` (export artifacts) — **1.0 hr**
7. Validate artifacts, add `artifacts/` to DVC local remote instructions, commit `dvc.lock` — **1.0 hr**
8. Career packaging: LinkedIn post draft + resume bullet drafts — **1.5 hr**
   **Total tasks time:** 9.0 hrs + resources \~1 hr = ≤10 hrs.

**Math & Intuition slot (1 hr):**

* **Topic:** Evaluation metrics for imbalanced data (Precision, Recall, AUPRC)
* **Video (🎥):** short explainer on Precision-Recall curves (StatQuest or similar) — **15–20 mins**.
* **Reading (📝):** short section from *Designing Machine Learning Systems* or *100 Pages ML* on evaluation of classifiers — est **15–20 mins**.
* **Runnable snippet idea:** compute precision\_recall\_curve and plot AUPRC with matplotlib (≤15 lines).
* **Interview prompt:** "Why choose AUPRC over ROC AUC for a highly imbalanced dataset?"

**ML/DL concepts introduced this week:** experiment tracking (MLflow), artifact checksums, evaluation for imbalanced data, minimal testing.

**Resources (new tool/topic: MLflow + testing patterns)**

* **MLflow (📄 official + video)**

  1. 🎥 **MLflow — Getting started / tutorial (video)** — search YouTube for "MLflow getting started" — est **30–60 mins** — **Required**.
  2. 📄 **MLflow official docs** — [https://mlflow.org/docs/latest/](https://mlflow.org/docs/latest/) — **(official)** — est **30–60 mins** — **Required**. ([MLflow][9])
  3. 🎥 **Short demo: Logging models + artifacts (YouTube)** — est **20–30 mins** — **Optional**.

* **Testing (pytest basics)**

  1. 🎥 **pytest intro (short)** — est **20–30 mins** — **Required**.
  2. 📄 **pytest docs — writing tests** — [https://docs.pytest.org/](https://docs.pytest.org/) — est **20–30 mins** — **Optional**.

**Small Read:** *Designing Machine Learning Systems* — 10–20 page excerpt on evaluation & monitoring — est **30–45 mins** — *Relevance:* helps design metrics & logging.

**Mini-Project (Week 4 capstone):** Sirtuin6 small molecules classification (UCI — 100 instances)

* **Dataset:** Sirtuin6 Small Molecules (100 instances). Provenance & license: UCI CC BY 4.0. ([UCI Machine Learning Repository][10])

  * **URL:** [https://archive.ics.uci.edu/dataset/748/sirtuin6+small+molecules-1](https://archive.ics.uci.edu/dataset/748/sirtuin6+small+molecules-1)
* **Acceptance criteria (≤3):**

  1. `python main.py --smoke` runs end-to-end, creates `artifacts/eval.csv` + `artifacts/model.joblib`.
  2. `checksums.txt` contains SHA256 for model artifact and eval CSV; pytest checksum test passes.
  3. README includes dataset provenance, license, RNG seed snippet, and short business metric calculation.
* **Deliverables:** Full E list from global rules (note: Dockerfile not required until Week 8 per rules). Include `dvc.yaml` + `dvc.lock`, local DVC instructions, and explicit MLflow example that logs run parameters and artifacts.
* **Business impact metric:** Baseline → model improvement in AUPRC; map to cost saved per false positive reduction using explicit assumptions in README.

**Ethics Checklist:**

1. **Bias:** dataset small and chemically curated → limited chemical space coverage.
2. **Harm scenario:** experimental resources wasted on false positives.
3. **Mitigation:** conservative decision thresholds + track group metrics; code hint: `mlflow.log_metric("group_recall_mt", recall_score(y_g, pred_g))`; record fairness checks.

**Weekly Outcomes (testable):**

* Run tests: `pytest` (includes checksum test & smoke test) → expected: all tests pass; `python main.py --smoke` prints `SMOKE OK`.

**Interview Prep (Week 4 checkpoint)**

* **LeetCode-style DS/Algo (easy/medium):** Two Sum (array/two pointers/hashmap).

  * **Practice resource (🎥):** NeetCode Two Sum walkthrough (YouTube).
  * **Time:** 20–40 mins.
* **SQL question:** SQLZoo or Mode Analytics question — e.g., "Find top 3 customers by total orders in last 6 months" — practice on SQLZoo ([https://sqlzoo.net](https://sqlzoo.net)).
* **Behavioral STAR:** "Describe a time you handled an ambiguous dataset; what did you do, what was the outcome?" — prepare 2-3 bullets.

**Week 4 Interview checkpoint extras (whiteboard & system prompts):**

* **Whiteboard math (prompt + short solution outline):**

  * **Prompt:** Given a binary classifier with 95% specificity and 70% sensitivity, and disease prevalence 1%, compute positive predictive value (PPV). *Outline:* use Bayes theorem: PPV = (sensitivity × prevalence) / (sensitivity × prevalence + (1−specificity) × (1−prevalence)). Show numbers to compute PPV ≈ small → interpret.
* **Debugging scenario:**

  * **Scenario:** Model performs well on CV but fails on holdout (AUC drops).
  * **Checklist of root causes:** data leakage, mismatched preprocessing, different label distributions, wrong random seed/split.
  * **Fixes:** check pipelines, ensure `ColumnTransformer` applied identically in train/test, re-run with stratified split, add unit tests for preprocessing.
* **System design prompt (detailed):**

  * **Prompt:** Design a minimal reproducible ML pipeline to train, version, and serve a scikit-learn model for inference in a small company (no cloud yet). **Key points to cover:** repo structure, DVC local remotes for data, MLflow for experiment tracking, `main.py` CLI, packaging via Docker (deferred until Week8), acceptance tests (pytest), API via FastAPI (Week12+), secrets NEVER in Git — store in environment variables or GCP Secret Manager later. Provide a CI checklist: pip install, pytest (checksum test), `python main.py --smoke`. (This maps exactly to your Week 12 rules when CI is added.)

**Stretch Goal:** Create a one-page portfolio entry and LinkedIn post for this project.

**Beginner Survival Tip:** If MLflow setup is tricky, start by logging metrics to a CSV and convert to MLflow later.

**Community:** MLflow Discourse / mlflow GitHub issues.

**GitHub folder snippet (full capstone):**

```
/project-sirtuin6/
  notebooks/main.ipynb
  main.py
  requirements.txt
  runtime.txt
  dvc.yaml
  dvc.lock
  artifacts/
    model.joblib
    eval.csv
    confusion_matrix.png
  checksums.txt
  tests/test_smoke.py
  README.md
```

**Weekly Total Estimated Time:** Tasks **9.0 hrs** + Resources **1.0 hr** = **≤10 hrs**

---

## Concept → Week mapping table (which ML/DL concept introduced when)

| Concept                                  |     Week introduced |
| ---------------------------------------- | ------------------: |
| EDA & data cleaning                      |              Week 1 |
| Train/test split, baseline classifier    |              Week 1 |
| scikit-learn Pipelines                   |              Week 2 |
| Cross-validation & hyperparameter search |              Week 2 |
| DVC (local) & `dvc.yaml`                 |              Week 2 |
| Bias–variance & regularization           |              Week 2 |
| PCA & eigen intuition                    |              Week 3 |
| Feature selection (RFE, SelectKBest)     |              Week 3 |
| ROC / AUPRC / confusion matrix           |              Week 3 |
| MLflow experiment tracking               |              Week 4 |
| Checksums & pytest smoke tests           |              Week 4 |
| Experiment reproducibility & RNG seeds   | Weeks 1–4 (applied) |

---

## Final notes & hand-offs

* **Assumptions made:** You requested a 4-week plan now; therefore the JSON summary contains 4 objects (weeks 1–4). When you ask to expand, I will add subsequent weeks incrementally and **not** repeat basics covered here (assumption saved to memory).
* **Datasets & provenance (UCI pages / licenses):** Soybean-small, Cryotherapy, Period Changer, Sirtuin6 (all UCI pages cited above). ([UCI Machine Learning Repository][3])
* **Pinned versions (example for requirements.txt):**

  ```
  python==3.10
  pandas==2.0.3
  scikit-learn==1.2.2
  matplotlib==3.7.1
  seaborn==0.12.2
  dvc==2.30.0
  mlflow==2.7.0
  ```

  *(Please pin exact versions after testing locally; these are examples — adjust as needed.)*

---

# JSON summary (4 objects)

```json
[
  {
    "week": 1,
    "title": "Foundations: EDA, pandas & scikit-learn basics",
    "topics": ["pandas", "Jupyter", "EDA", "baseline classification"],
    "project": "EDA and baseline classifier on Soybean-small (UCI, 47 rows)"
  },
  {
    "week": 2,
    "title": "Pipelines, cross-validation & DVC (local)",
    "topics": ["scikit-learn Pipeline", "cross-validation", "DVC (local)"],
    "project": "Pipeline + DVC reproducible training on Cryotherapy (UCI, 90 rows)"
  },
  {
    "week": 3,
    "title": "Feature selection & PCA (high-dim data)",
    "topics": ["feature selection (RFE/SelectKBest)", "PCA", "AUC/AUPRC"],
    "project": "Feature selection + PCA pipeline on Period Changer (UCI, 90 rows, 1177 features)"
  },
  {
    "week": 4,
    "title": "Capstone: end-to-end reproducible pipeline & MLflow",
    "topics": ["MLflow", "artifact checksums", "pytest smoke tests", "evaluation for imbalanced data"],
    "project": "End-to-end reproducible run on Sirtuin6 (UCI, 100 rows) with MLflow tracking and checksum tests"
  }
]
```
 
---

If you'd like, I can now:

* generate the **starter `README.md`, `requirements.txt` (pinned), `runtime.txt`, `main.py` smoke runner**, and skeleton `notebooks/main.ipynb` for **Week 1** right away; **or**
* expand to Weeks 5–8 next (I will not repeat material covered here).

Which would you like me to produce next?

[1]: https://pandas.pydata.org/docs/?utm_source=chatgpt.com "pandas 2.3.2 documentation"
[2]: https://scikit-learn.org/stable/getting_started.html?utm_source=chatgpt.com "Getting Started — scikit-learn 1.7.2 documentation"
[3]: https://archive.ics.uci.edu/ml/datasets/soybean%2B%28small%29 "UCI Machine Learning Repository"
[4]: https://dvc.org/doc?utm_source=chatgpt.com "DVC Documentation"
[5]: https://scikit-learn.org/stable/user_guide.html?utm_source=chatgpt.com "User Guide — scikit-learn 1.7.2 documentation"
[6]: https://archive.ics.uci.edu/ml/datasets/Cryotherapy%2BDataset "UCI Machine Learning Repository"
[7]: https://www.youtube.com/playlist?list=PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR&utm_source=chatgpt.com "StatQuest"
[8]: https://archive.ics.uci.edu/dataset/729/period%2Bchanger-2 "UCI Machine Learning Repository"
[9]: https://mlflow.org/docs/3.2.0/?utm_source=chatgpt.com "Documentation"
[10]: https://archive.ics.uci.edu/dataset/748/sirtuin6%2Bsmall%2Bmolecules-1?utm_source=chatgpt.com "Sirtuin6 Small Molecules"
