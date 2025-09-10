# 17-Week Project-Driven Curriculum — now with explicit Math & Theory slots

> Notes up front:
> *Math approach:* practical + interview-focused. Each math slot is 45–60 minutes: short video + 1 small runnable code snippet or notebook cell that demonstrates the idea (no heavy proofs). Where appropriate, I include a *stretch* reference to deeper math (marked **Stretch**).
> *Seeds & reproducibility:* every project must include seed snippets and reproducibility commands (per your master prompt).
> *Deliverables, DVC, MLflow, Docker, CI, Model Card, ethics checklist, business metrics, tests* are enforced exactly as before.
> *Datasets:* Weeks 1–4 UCI ≤100 rows; Weeks 5–8 Kaggle <5k; Weeks 9+ small public CPU-friendly sets.

---

## One-page overview (4 blocks) — with math emphasis

* **Weeks 1–4 (Foundations + Math Foundations)**
  Topics: pandas, EDA, scikit-learn pipelines, basic statistics & probability, linear algebra intuition.
  Projects: Lens classifier (UCI), Fertility EDA, Cryotherapy pipeline.
  Hireable skills: reproducible notebooks, baseline modeling, reproducible pipelines, applied statistics for ML.

* **Weeks 5–8 (MLOps intro + model engineering + applied optimization)**
  Topics: DVC, MLflow, Docker, model evaluation, basic optimization & regularization intuition.
  Projects: Titanic (Kaggle) end-to-end with DVC/MLflow + Docker & Model Card.
  Hireable skills: experiment tracking, data versioning, containerized reproducible pipelines.

* **Weeks 9–12 (DL intro, NLP, time-series + optimization math)**
  Topics: PyTorch basics, small NLP, time-series, gradient descent intuition, loss surfaces.
  Projects: Mini-IMDB small subset, time-series forecasting small dataset, CI + PR simulation.
  Hireable skills: basic deep learning, transfer learning (small), CI for ML.

* **Weeks 13–17 (Serving, recommender, GCP + deployments & monitoring math)**
  Topics: FastAPI serving, containerized APIs, recommender basics, deployment to GCP, monitoring & drift detection math (distribution comparisons).
  Projects: API for model, recommender stretch, final capstone + portfolio + two resumes.
  Hireable skills: serving and deployment, API testing, MLOps pipelines, portfolio-ready projects.

---

# Weekly plan (Weeks 1 → 17)

> Each week includes: **Week X — Title**, **Prerequisites**, **Why this matters**, **Time-split Tasks**, **Math & Intuition slot**, **ML/DL concepts introduced this week**, **Resources**, **Small Read**, **Mini-Project (dataset + acceptance criteria + deliverables + business metric)**, **Ethics checklist**, **Weekly Outcome**, **Stretch Goal**, **Beginner Survival Tip**, **Community**, **GitHub folder snippet**, **Weekly Total Estimated Time**.

> I keep totals within allowed caps. If any single task exceeds 25% of the week’s hours I provide a simplified alternative immediately below that task.

---

## Week 1 — Tooling & pandas + Jupyter basics (tiny UCI)

**Prerequisites:** intermediate Python, basic Git.
**Why this matters:** Core data wrangling and reproducible notebook setup.
**Time-split Tasks (≤10 hrs):**

* `2.0 hrs:` Project scaffold, virtualenv, `runtime.txt` (`python==3.10`), `requirements.txt` pinned.
* `2.5 hrs:` pandas quick practice: load UCI Lenses, `df.info()`, impute, basic transforms.
* `2.0 hrs:` Notebook writeup + export; `main.py --smoke` stub that loads data & prints summary.
* `1.5 hrs:` Baseline: `DummyClassifier` majority baseline → save `artifacts/eval.csv`.
* `1.0 hrs:` README: dataset provenance (UCI link) + seed snippet.
  **Math & Intuition slot:** *Not scheduled this week* (we start Week 2).
  **ML/DL concepts introduced this week:** data representation, baseline concept, evaluation basics (accuracy).
  **Resources (per-tool — 3 items):**
* 🎥 pandas quickstart (short video or official 10-minute guide), 2023–24, est 30m, **Required**.
* 📄 pandas user guide (official), est 45m, **Required**.
* 📄 Jupyter docs (official), est 20m, **Optional**.
  **Small Read:** *Python Data Science Handbook* pages **35–50** (≈30–40 mins) — pandas basics.
  **Mini-Project:** UCI Lenses (24 rows).
* Acceptance criteria: `notebooks/main.ipynb` runs, `artifacts/eval.csv` written, `requirements.txt` pinned, `runtime.txt` present.
* Deliverables (Week 1 minimum): `notebooks/main.ipynb`, `requirements.txt`, `runtime.txt`, `main.py` (with `--smoke`), `artifacts/`, `checksums.txt` (sha256 of model artifact or eval), README with dataset URL & license note.
* Business metric example: baseline accuracy → improvement mapping (document assumptions).
  **Ethics checklist:** dataset small sample bias; possible harm for minority subgroups; mitigation via stratified reporting + fairlearn hint in README.
  **Weekly Outcome:** `python main.py --smoke` prints `SMOKE OK` and writes `artifacts/eval.csv`.
  **Stretch Goal:** add scikit-learn `Pipeline` example.
  **Beginner Survival Tip:** if environment issues, use Google Colab and attach `runtime.txt`.
  **Community:** r/datascience / UCI mailing lists.
  **GitHub snippet:** `week01_lenses/` with expected files.
  **Weekly Total Estimated Time:** Tasks 9.0 hrs + Resources 1.0 hr = **10.0 hrs**.

---

## Week 2 — EDA & Visualization + Math: descriptive stats & basic probability

**Prerequisites:** Week 1.
**Why this matters:** Communicate insights and ground probabilistic thinking used in evaluation and thresholds.
**Time-split Tasks (≤10 hrs):**

* `2.0 hrs:` matplotlib & seaborn quick examples; practice plots.
* `2.5 hrs:` EDA on UCI Fertility (≤100 rows): distributions, missingness, pairwise plots, correlations.
* `1.5 hrs:` Save artifacts (`confusion_matrix.png`, `roc.png` if applicable).
* `2.0 hrs:` Notebook + tests for data sanity.
* `0.5 hrs:` Update README with interpretations.
  **Math & Intuition slot (1.0 hr — included in tasks above):**
* Topic: **Descriptive statistics & probability basics** — mean, variance, standard error, probability basics (events, independence), and how they connect to sampling error and confidence in small datasets.
* Format: 20–30 min video/read + 30–40 min runnable notebook snippet showing sample mean variance and bootstrapped CI (small code).
  **ML/DL concepts introduced this week:** sampling variability, confidence in metrics, interpretation of ROC/PR at a conceptual level.
  **Resources (for new tools/topics):**
* 🎥 short Descriptive Stats video (Khan/StatQuest), est 30m, **Required**.
* 📄 matplotlib quickstart (official), est 30m, **Required**.
* 🎥 seaborn tutorial (gallery), est 45m, **Optional**.
  **Small Read:** *Python Data Science Handbook* pages **70–90** (≈40 mins) — visualization.
  **Mini-Project:** UCI Fertility (≤100 rows).
* Acceptance criteria: artifacts saved, tests pass, README interpretations present.
* Deliverables: same repo-style files plus `artifacts/*.png`.
* Business metric: e.g., feature X associated with outcome Y — map effect size to business impact (document assumptions).
  **Ethics checklist:** small-sample bias; harm via misinterpretation; mitigation: report confidence intervals + log to MLflow.
  **Weekly Outcome:** `pytest` passes data tests; notebook exports artifacts.
  **Stretch Goal:** bootstrap CI for metric.
  **Beginner Survival Tip:** if pairplots slow, subsample to 50 rows.
  **Community:** Kaggle EDA discussion.
  **GitHub snippet additions:** `artifacts/*.png`, `tests/test_data.py`.
  **Weekly Total Estimated Time:** Tasks 9.5 hrs + Resources 0.5 hr = **10.0 hrs**.

---

## Week 3 — scikit-learn pipelines & math: linear algebra intuition

**Prerequisites:** Weeks 1–2.
**Why this matters:** Pipelines prepare code for production; linear algebra underpins many ML models.
**Time-split Tasks (≤10 hrs):**

* `2.5 hrs:` Build `Pipeline` & `ColumnTransformer` for preprocessing.
* `2.5 hrs:` Train baseline LogisticRegression + RandomForest on UCI Cryotherapy (≤100 rows).
  *If RF takes >25% CPU → Simplified alternative:* use `n_estimators=10` or only LogisticRegression.
* `2.0 hrs:` Implement `main.py` smoke entry and `checksums.txt`.
* `1.0 hrs:` Baseline vs final comparison table.
* `1.0 hrs:` README update & seeds.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Linear algebra intuition** — vectors, dot product, matrix multiplications, and how they relate to linear models and feature combinations; basic eigenvalue intuition (what PCA finds). Notebook: small numeric examples showing dot products, projection, covariance matrix & top eigenvector via `numpy.linalg.eig`.
  **ML/DL concepts introduced this week:** linear models, decision boundaries, PCA intuition, matrix representation of datasets.
  **Resources:** scikit-learn pipelines docs, short Linear Algebra intuition video (Gilbert Strang intro / 3Blue1Brown), scikit examples.
  **Small Read:** *Hands-On ML* (scikit parts) pages **20–40** (\~40 mins) — linear models & pipelines.
  **Mini-Project:** UCI Cryotherapy (≤100 rows). Deliverables: full repo items + `checksums.txt`.
  **Ethics checklist:** sampling bias; model harm; mitigation with `fairlearn` hint.
  **Weekly Outcome:** `python main.py --smoke` runs, outputs artifacts, checksums match test.
  **Stretch Goal:** small PCA-based feature pipeline.
  **Beginner Survival Tip:** if matrix ops are confusing, run visual examples with small 2×2 matrices.
  **Community:** scikit-learn discussions.
  **GitHub snippet additions:** pipelines examples.
  **Weekly Total Estimated Time:** Tasks 9.5 hrs + Resources 0.5 hr = **10.0 hrs**.

---

## Week 4 — Capstone #1 (Foundations capstone) + math: probability for model evaluation

**Prerequisites:** Wks1–3.
**Why this matters:** Consolidate foundational skills and math for interview readiness.
**Time-split Tasks (≤10 hrs):**

* `3.0 hrs:` Consolidate project, ensure `pip install -r requirements.txt && python main.py --smoke` works.
* `3.0 hrs:` Produce `artifacts/` (eval CSV, confusion matrix PNG).
* `2.0 hrs:` README polish, Model Card placeholder, checksums.
* `1.0 hrs:` LinkedIn post + one resume bullet.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Probability for model evaluation** — hypothesis testing intuition (p-value concept), sampling distributions for metrics; quick run: bootstrap metric CI and interpret p-values for comparing two models.
  **ML/DL concepts introduced this week (review + interview prep):** statistical comparison of models; bias/variance recap.
  **Resources:** short readings on bootstrap & model comparison.
  **Small Read:** *100 Pages ML* pages **1–15** (\~30 mins).
  **Mini-Project (Capstone 1):** full small UCI project deliverables.
  **Acceptance criteria:** Notebook reproducible `--smoke`, checksums pass, README includes dataset license & seed snippet.
  **Ethics checklist:** bias source, harm, mitigation hint.
  **Weekly Outcome:** Reviewer runs `python main.py --smoke` and reproduces artifact & checksum.
  **Interview Prep (Week 4 checkpoint included):** whiteboard math prompt: PCA eigenvalue power iteration (already included earlier), debugging & design prompts per master spec.
  **Stretch Goal:** include bootstrap-based significance test for metric improvement.
  **Beginner Survival Tip:** if short on time, provide minimal reproducible repo with clear README and smoke run.
  **Community:** Kaggle feedback threads.
  **GitHub snippet:** full capstone structure.
  **Weekly Total Estimated Time:** Tasks 9.0 hrs + Resources 1.0 hr = **10.0 hrs**.

---

## Week 5 — DVC & Kaggle import (Titanic) + math: regularization & overfitting intuition

**Prerequisites:** Wk4 capstone.
**Why this matters:** Data versioning for reproducibility; understanding regularization reduces overfitting in practice.
**Time-split Tasks (13 hrs):**

* `2.0 hrs:` DVC init & local remote setup.
* `3.0 hrs:` Download Kaggle Titanic, `dvc add` data, create `dvc.yaml` stages.
* `3.0 hrs:` Baseline `LogisticRegression` in pipeline with `params.yaml`.
* `3.0 hrs:` README `dvc pull` instructions, tests.
* `2.0 hrs:` MLflow local quick log of baseline.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Regularization & overfitting intuition** — L1 vs L2 idea, geometric intuition, and effect on coefficients; quick code: compare coeff magnitudes with/without `C` regularization in LogisticRegression.
  **ML/DL concepts introduced this week:** regularization, train/validation selection, DVC pipeline equivalents.
  **Resources:** DVC quickstart, Kaggle Titanic page, brief video on regularization.
  **Small Read:** *Hands-On ML* pages **60–75** (\~40 mins) — model evaluation & regularization.
  **Mini-Project:** Titanic (Kaggle <5k) with DVC local remote.
  **Acceptance criteria:** `dvc.yaml` + `dvc.lock` present, `dvc pull` instructions in README, smoke run writes artifacts and MLflow logged run.
  **Ethics checklist:** sample bias from manifest data; mitigation via stratified metrics.
  **Weekly Outcome:** `dvc repro` runs; `python main.py --smoke` logs MLflow run.
  **Stretch Goal:** add parameterized DVC pipeline with experiments.
  **Beginner Survival Tip:** if DVC push fails, commit steps and add `data_download.sh`.
  **Community:** DVC community, Kaggle Titanic threads.
  **GitHub snippet:** add `dvc.yaml`, `.dvc/` local remote config.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 6 — Feature engineering & model selection + math: probability distributions & hypothesis testing

**Prerequisites:** Week 5.
**Why this matters:** Features drive model performance; statistical tests help choose features confidently.
**Time-split Tasks (13 hrs):**

* `3.0 hrs:` Feature engineering lab for Titanic (title extraction, family size, cabin features).
* `3.0 hrs:` Encoding pipeline + `RandomizedSearchCV` light param search.
* `3.0 hrs:` Evaluate baseline vs tuned, create comparison CSV.
* `2.0 hrs:` Log to MLflow & save artifacts.
* `2.0 hrs:` README updates & resume bullet draft.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Probability distributions & hypothesis testing for features** — t-test vs non-parametric test, p-value interpretation, effect size; code: quick t-test comparing a numeric feature across labels.
  **ML/DL concepts introduced:** model selection, cross-validation, significance testing for feature importance.
  **Resources:** scikit-learn model selection docs, brief stats tutorial.
  **Small Read:** *Designing ML Systems* pages **10–25** (\~30 mins).
  **Mini-Project:** Titanic feature engineering & selection. Acceptance criteria: comparison CSV, MLflow logs, `checksums.txt`.
  **Ethics checklist:** features that proxy sensitive attributes — mitigation: test fairness metrics.
  **Weekly Outcome:** best model logged in MLflow; comparison CSV created.
  **Stretch Goal:** SHAP-based feature explanations (Stretch).
  **Beginner Survival Tip:** if CV is slow, use 3-fold CV and smaller param grid.
  **Community:** Kaggle forums.
  **GitHub snippet:** `params.yaml`, MLflow instructions.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 7 — MLflow experiments & math: cross-validation & bias-variance curves

**Prerequisites:** Wk6.
**Why this matters:** Track experiments and understand bias/variance tradeoff practically.
**Time-split Tasks (13 hrs):**

* `3.5 hrs:` Set up MLflow tracking for multiple experiments; log params & artifacts.
* `3.0 hrs:` Automate experiments with DVC stages or scripts.
* `3.0 hrs:` Analyze runs to pick best model and create report.
* `2.5 hrs:` Add tests and README updates.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Cross-validation intuition & bias-variance curves** — show train/val curves, under/overfitting diagnosis with plots; code: learning curves for model complexity.
  **ML/DL concepts introduced:** validation techniques, learning curves, model capacity.
  **Resources:** MLflow docs, DVC pipelines docs, scikit model selection docs.
  **Small Read:** *Designing ML Systems* pages **26–40** (\~35 mins).
  **Mini-Project:** Titanic experiments tracked with MLflow; artifact comparison.
  **Ethics checklist:** ensure logs don't include PII; mitigation: scrub logs.
  **Weekly Outcome:** MLflow UI / artifact folder shows runs and best model chosen.
  **Stretch Goal:** MLflow Model Registry local demo.
  **Beginner Survival Tip:** if MLflow UI heavy, keep runs stored locally and inspect CSV metrics.
  **Community:** MLflow discussions.
  **GitHub snippet:** `mlflow/` notes.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 8 — Docker + Model Card + Capstone #2 + math: optimization & gradient basics

**Prerequisites:** Wk7.
**Why this matters:** Containerization + model documentation; optimization understanding helps hyperparameter choices and DL transition.
**Time-split Tasks (15 hrs):**

* `3.0 hrs:` Write `Dockerfile` for `python main.py --smoke`.
* `3.0 hrs:` Build and test container locally (`docker run` smoke). *If build >25%:* simplified alt — keep slim base image and a `--smoke` target stage.
* `3.0 hrs:` Create full **Model Card** (required fields).
* `3.0 hrs:` Capstone packaging: Dockerfile, Model Card, artifacts, checksums.
* `2.0 hrs:` README Docker usage & resume/LinkedIn draft.
* `1.0 hrs:` Add small CI placeholder for future (Week 12).
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Optimization & gradient basics** — gradient descent intuition, learning rate tradeoffs, convex vs non-convex surfaces; code: a tiny gradient descent fit for quadratic \~ demonstration. **Stretch:** brief mention of momentum and Adam.
  **ML/DL concepts introduced:** optimization, convergence, learning rate significance — sets stage for PyTorch.
  **Resources:** Docker docs, Model Card templates, short optimization tutorial.
  **Small Read:** *100 Pages ML* pages **16–30** (\~30 mins).
  **Mini-Project (Week 8 Capstone):** Titanic full E2E packaged with Docker + Model Card. Acceptance: docker build/run smoke, Model Card present, checksums valid.
  **Ethics checklist:** data license noted in README, bias items logged, fairlearn hint.
  **Weekly Outcome:** `docker build -t project .` and `docker run --rm project python main.py --smoke` produce artifacts.
  **Stretch Goal:** push container to GHCR (document only).
  **Beginner Survival Tip:** if Docker unavailable, provide detailed `venv` run script and Dockerfile as future step.
  **Community:** Docker forums.
  **GitHub snippet:** `Dockerfile`, `model_card.md`.
  **Weekly Total Estimated Time:** Tasks 15.0 hrs + Resources 0.5 hr = **15.5 hrs** — **Adjustment:** reduce one 0.5-hr optional resource so total **= 15.0 hrs** to respect the cap.

---

## Week 9 — PyTorch basics + Intro to NLP + math: derivatives & chain rule intuition

**Prerequisites:** Wk8.
**Why this matters:** Core DL skills and calculus intuition underpin backprop and training decisions.
**Time-split Tasks (13 hrs):**

* `2.5 hrs:` PyTorch quickstart: tensors, autograd, basic training loop.
* `3.0 hrs:` Prepare small NLP subset (IMDb/SST2 small) and tokenization pipeline.
* `3.0 hrs:` Train tiny PyTorch classification model (embedding + small LSTM or linear classifier on TF-IDF if heavy). *If training >25% CPU:* fallback to TF-IDF + LogisticRegression.
* `2.0 hrs:` Log MLflow run, save model artifact.
* `2.5 hrs:` Notebook writeup & README updates.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Derivatives & chain rule intuition** — show a scalar chain rule example + how gradients propagate through layers; code: autodiff on a toy function with PyTorch `.backward()`.
  **ML/DL concepts introduced:** tensors, autograd, loss functions, backprop intuition.
  **Resources:** PyTorch quickstart, Hugging Face dataset guide (for data loading).
  **Small Read:** *Hands-On ML* pages **100–120** (\~40 mins) — transition commentary.
  **Mini-Project:** Mini-IMDB small subset — acceptance: smoke run completes, artifact saved.
  **Ethics checklist:** content sensitivity; mitigation: filter & human flagging.
  **Weekly Outcome:** `python main.py --smoke` executes small train and writes artifact under 20 minutes on CPU.
  **Stretch Goal:** small DistilBERT fine-tune (if GPU available).
  **Beginner Survival Tip:** if PyTorch unfamiliar, use scikit-learn with embeddings precomputed.
  **Community:** PyTorch & Hugging Face forums.
  **GitHub snippet:** add `torch` requirement, small training script.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 10 — Small NLP fine-tuning & math: loss landscapes & regularization in NN

**Prerequisites:** Week 9.
**Why this matters:** Practical transfer learning and understanding regularization in DL.
**Time-split Tasks (13 hrs):**

* `3.0 hrs:` Tokenization & preprocessing pipeline for small subset.
* `3.5 hrs:` Fine-tune small model or train simple NN; log to MLflow. *If >25% CPU:* use feature extraction + scikit pipeline.
* `2.5 hrs:` Error analysis & explainability (confusion by class).
* `2.0 hrs:` Save artifacts, update Model Card.
* `2.0 hrs:` LinkedIn/resume bullet drafts.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Loss landscapes & regularization in neural nets** — why regularization helps, L2 weight decay vs dropout intuition; notebook demo with/without dropout or weight decay on small net.
  **ML/DL concepts introduced:** transfer learning basics, embedding fine-tuning, regularization mechanisms in NNs.
  **Resources:** Hugging Face tutorials, PyTorch docs.
  **Small Read:** *Designing ML Systems* pages **41–55** (\~35 mins).
  **Mini-Project:** small NLP fine-tune project deliverables.
  **Ethics checklist:** content moderation & bias.
  **Weekly Outcome:** `python main.py --smoke` produces artifacts and MLflow logs.
  **Stretch Goal:** dataset augmentation or adversarial robustness test.
  **Beginner Survival Tip:** reduce dataset size to keep CPU runtimes low.
  **Community:** Hugging Face forums.
  **GitHub snippet:** include `transformers` optionally (only if CPU budget allows).
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 11 — Time-series forecasting basics + math: stationarity & autocorrelation intuition

**Prerequisites:** Wks 1–10.
**Why this matters:** Forecasting common in business use-cases; math helps choose methods.
**Time-split Tasks (13 hrs):**

* `3.0 hrs:` Time-series primer (train/test with rolling windows).
* `3.5 hrs:` Build baseline naive & small ML regressor on lag features (CPU-friendly).
* `3.0 hrs:` Implement evaluation (MAE, RMSE, MAPE) and save plots.
* `2.0 hrs:` Document reproducibility & small-proxy for heavy models.
* `1.5 hrs:` Update README & Model Card.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Stationarity & autocorrelation** — ACF/PACF intuition, why differencing matters; code: compute ACF and show seasonal patterns.
  **ML/DL concepts introduced:** time-series features, windowing, forecasting error metrics.
  **Resources:** short time-series tutorial, scikit guide for regression-based forecasts.
  **Small Read:** *Hands-On ML* pages **140–160** (\~40 mins).
  **Mini-Project:** small time-series dataset; deliverables per master prompt.
  **Ethics checklist:** decisions from forecasts can cause resource misallocation; mitigation: include uncertainty intervals.
  **Weekly Outcome:** `python main.py --smoke` creates forecast CSV + plots.
  **Stretch Goal:** simple LSTM or quantile regression (if CPU budget allows).
  **Beginner Survival Tip:** if LSTM is heavy, use lag features + RandomForestRegressor.
  **Community:** r/MachineLearning time-series threads.
  **GitHub snippet:** add `notebooks/timeseries.ipynb`.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 12 — CI + PR simulation + Capstone #3 + math: bias-variance math & confidence intervals

**Prerequisites:** Week 11.
**Why this matters:** CI and code review are production expectations; math helps interpret model uncertainty.
**Time-split Tasks (15 hrs):**

* `4.0 hrs:` Create `tests/` including checksum test (assert SHA256), data sanity tests, and API test placeholder.
* `3.5 hrs:` Add `.github/workflows/ci.yml` to run `pip install -r requirements.txt`, `pytest`, and `python main.py --smoke`.
* `3.5 hrs:` PR simulation: create PR checklist, simulate code review & fix.
* `2.5 hrs:` Capstone packaging: integrate DVC, MLflow, Dockerfile, Model Card (choose time-series or NLP).
* `1.5 hrs:` Final README polish for CI/DVC/MLflow.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Bias-variance math & confidence intervals for metrics** — compute bootstrap CIs for metrics and interpret when differences are meaningful. Notebook demo included.
  **ML/DL concepts introduced:** formalizing uncertainty, reproducible CI/CD for ML.
  **Resources:** GitHub Actions docs, CI best practices.
  **Small Read:** *Designing ML Systems* pages **56–75** (\~40 mins).
  **Mini-Project (Week12 Capstone):** Full project with CI, tests (checksum test required), DVC local remote, and PR simulation notes.
  **Acceptance criteria:** CI steps (local script or GH Actions), `pytest` passes and smoke run executed by CI.
  **Ethics checklist:** data leakage detection tests included.
  **Weekly Outcome:** CI script passes locally; `pytest` tests (checksum) succeed.
  **Interview Prep (Week 12 checkpoint included).**
  **Stretch Goal:** scheduled nightly smoke tests in CI.
  **Beginner Survival Tip:** if GH Actions blocked, provide `ci_local.sh` replicating CI steps.
  **Community:** GitHub Actions discussions.
  **GitHub snippet:** `.github/workflows/ci.yml`, `tests/` folder.
  **Weekly Total Estimated Time:** Tasks 15.0 hrs + Resources 1.0 hr = **16.0 hrs** — **Adjustment:** trim 1.0 hr optional resource to meet cap **15.0 hrs**.

---

## Week 13 — FastAPI basics (serving) + math: data/schema validation & uncertainty propagation

**Prerequisites:** Week 12.
**Why this matters:** Serve model predictions robustly and handle input uncertainty.
**Time-split Tasks (13 hrs):**

* `3.0 hrs:` FastAPI minimal app: load model artifact & predict.
* `3.0 hrs:` Define Pydantic input schema & add sample inference.
* `3.0 hrs:` Add `tests/test_api.py` asserting JSON schema & sample output.
* `2.5 hrs:` Add logging, deterministic inference wrapper (seed handling).
* `1.5 hrs:` README API usage + example curl commands.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Uncertainty propagation & predictive intervals** — conceptually how input uncertainty affects output; demo a Monte Carlo dropout inference (conceptual/simple code demo).
  **ML/DL concepts introduced:** deterministic inference, schema contracts, uncertainty-aware outputs.
  **Resources:** FastAPI tutorial & Pydantic docs.
  **Small Read:** *Python Data Science Handbook* pages **120–140** (\~40 mins) on production patterns.
  **Mini-Project:** add FastAPI wrapper for a Week-12 model; deliver tests & README.
  **Ethics checklist:** do not log PII; add privacy note.
  **Weekly Outcome:** `pytest` including API tests passes.
  **Stretch Goal:** add rate-limiting; add healthchecks.
  **Beginner Survival Tip:** if FastAPI is new, stub with Flask but keep API contract consistent.
  **Community:** FastAPI forum.
  **GitHub snippet:** `app/` with `main.py`, `requirements.txt`.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 14 — Containerized serving & API tests + math: monitoring statistics (drift detection basics)

**Prerequisites:** Week 13.
**Why this matters:** Containers + tests prepare for cloud deployment; monitoring math helps detect drift early.
**Time-split Tasks (13 hrs):**

* `3.0 hrs:` Multi-stage `Dockerfile` for API with healthcheck.
* `3.0 hrs:` `docker run` + curl smoke tests for `/health` & `/predict`.
* `3.0 hrs:` CI step to build container and run `pytest tests/test_api.py`.
* `2.0 hrs:` Add checksum test that runs inside container (CI step).
* `2.0 hrs:` Document container usage & compose dev setup.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Monitoring stats for drift detection** — compare distributions (KL divergence / KS test idea) and conceptually detect drift; small notebook demonstrating KS test on a numeric feature pre/post.
  **ML/DL concepts introduced:** sanity checks, monitoring metrics, drift detection basics.
  **Resources:** Docker & CI docs.
  **Small Read:** *Designing ML Systems* pages **76–95** (\~40 mins).
  **Mini-Project:** containerized API that passes tests in CI.
  **Ethics checklist:** ensure logs are scrubbed & deploy secrets managed.
  **Weekly Outcome:** container image builds, container smoke tests pass in CI.
  **Stretch Goal:** basic Evidently AI drift check (optional).
  **Beginner Survival Tip:** if CI can't run docker, mock inference function in tests.
  **Community:** Docker & monitoring communities.
  **GitHub snippet:** `Dockerfile` for API, updated `ci.yml`.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 15 — Recommender fundamentals (stretch) + math: similarity metrics & evaluation

**Prerequisites:** Weeks 1–14.
**Why this matters:** Recommender basics are often used in product teams; math helps pick similarity measures & evaluation metrics.
**Time-split Tasks (13 hrs):**

* `3.0 hrs:` Popularity baseline + item-based collaborative filter using co-occurrence.
* `3.5 hrs:` Build simple recommender on small dataset (or subset) with evaluation. *If heavy:* fallback to popularity + simple item similarity on small subset.
* `3.0 hrs:` API endpoint for `recommend(user_id)`.
* `1.5 hrs:` Save evaluation CSV (precision\@k, recall\@k).
* `2.0 hrs:` README + Model Card note (stretch).
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Similarity metrics & ranking evaluation** — cosine similarity intuition, recall\@k/precision\@k definitions and interpretation; demo computing similarity matrix and ranking.
  **ML/DL concepts introduced:** collaborative filtering intuition, ranking metrics.
  **Resources:** recommender primer & small dataset tutorials.
  **Small Read:** *Hands-On ML* pages **200–220** (\~40 mins).
  **Mini-Project:** small recommender project (stretch); deliver artifacts & tests.
  **Ethics checklist:** popularity bias and echo chambers; mitigation: diversity-aware re-ranking.
  **Weekly Outcome:** `python main.py --smoke` outputs top-K recommendations CSV for sample users.
  **Stretch Goal:** matrix factorization using SVD (Stretch).
  **Beginner Survival Tip:** if dataset is big, sample 1k users/items.
  **Community:** Movielens/recommender forums.
  **GitHub snippet:** `recommender/` module.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 16 — Recommender mini-capstone & API stress tests + math: evaluation significance & A/B test basics

**Prerequisites:** Week 15.
**Why this matters:** End-to-end recommender + testing demonstrates real product capability.
**Time-split Tasks (13 hrs):**

* `3.0 hrs:` Finalize recommender model and artifact.
* `3.0 hrs:` Implement API and `tests/test_api.py` asserting schema & outputs.
* `3.0 hrs:` Add containerization & CI build + container-based tests.
* `2.0 hrs:` Create evaluation report & business mapping.
* `2.0 hrs:` LinkedIn post + resume bullet drafts.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Evaluation significance & A/B test basics** — how to interpret lift and evaluate statistical significance for ranking metrics; notebook showing basic comparison and bootstrap CIs for precision\@k.
  **ML/DL concepts introduced:** ranking evaluation, A/B testing basics for models.
  **Resources:** short A/B testing primers & recommender evaluation docs.
  **Small Read:** *Designing ML Systems* pages **96–115** (\~40 mins).
  **Mini-Project Acceptance criteria:** API passes tests; Docker image builds; CI runs smoke tests.
  **Ethics checklist:** personalization fairness, cold-start fairness; mitigation: fallback policies & logging.
  **Weekly Outcome:** CI run passes; recommendations validated.
  **Stretch Goal:** small simulated A/B test (Stretch).
  **Beginner Survival Tip:** if calculations are heavy, compute metrics on a sample set.
  **Community:** Recommender system groups.
  **GitHub snippet:** updated `ci.yml` to include recommender tests.
  **Weekly Total Estimated Time:** Tasks 12.0 hrs + Resources 1.0 hr = **13.0 hrs**.

---

## Week 17 — Final portfolio, two resumes, GCP deployment steps & Capstone #4 + math: monitoring & drift statistics (final)

**Prerequisites:** All previous weeks completed; CI and tests in place.
**Why this matters:** Portfolio & deployment show hiring readiness and production understanding.
**Time-split Tasks (15 hrs):**

* `3.0 hrs:` Build portfolio site (Notion or GitHub Pages) linking all repos, abstracts, LinkedIn posts.
* `3.5 hrs:` Produce **two role-specific resumes** (Data Scientist and ML Engineer) tailored to projects (metrics + action verbs).
* `3.0 hrs:` Prepare final capstone Docker image for GCP deployment and instructions. Ensure **never commit secrets** — include GitHub Secrets guide for `GOOGLE_APPLICATION_CREDENTIALS`.
* `3.0 hrs:` Provide GCP deployment steps: build & push to Container Registry, deploy to Cloud Run (or GCE/GKE options), sample `gcloud` commands, and where to set secrets (GitHub Actions).
* `2.5 hrs:` Final README polish, Model Cards, CI checks, final checksums, and stakeholder 1-page pitch.
  **Math & Intuition slot (1.0 hr included):**
* Topic: **Monitoring & drift statistics** — recap KS test, KL divergence intuition, population stability index (PSI) concept; include small notebook to compute PSI between training and sample production feature distributions.
  **ML/DL concepts introduced:** monitoring metrics, drift detection, rollback strategies.
  **Resources:** GCP quickstart & Cloud Run deployment docs, GitHub Secrets docs.
  **Small Read:** *Python Data Science Handbook* pages **200–220** (\~40 mins) on packaging & deployment.
  **Capstone (Week17 final):** full project docker image + deploy guide + portfolio + 2 resumes. Acceptance: reviewer can follow README to build image & find deploy guide; resumes ready; portfolio links valid.
  **Ethics checklist:** final model card includes ethics & bias summary + mitigations.
  **Weekly Outcome:** deliver portfolio URL and two resume files (or markdown) and `deploy_gcp.md` with commands (example included).
  **Stretch Goal:** configure Cloud Monitoring alert (documented).
  **Beginner Survival Tip:** If GCP access blocked, provide exact steps and scripts to run later; host portfolio on Notion in meantime.
  **Community:** GCP community.
  **GitHub snippet:** `deploy_gcp.md`, `portfolio/` link, `resumes/` folder.
  **Weekly Total Estimated Time:** Tasks 15.0 hrs + Resources 1.0 hr = **16.0 hrs** — **Adjustment:** reduce 1.0 hr optional resource detail to meet cap **15.0 hrs**.

---

# Interview Prep checkpoints (Weeks 4, 8, 12, 17) — math & the rest (recap)

At each checkpoint you still get:

* 1 whiteboard math prompt (now explicitly math-heavy where appropriate).
* 1 debugging scenario checklist.
* 1 system design prompt.
  I kept the earlier examples; math prompts now align with the math topics covered leading up to those checkpoints (e.g., Week 8 optimization prompt, Week 12 bias-variance hypothesis testing prompt, Week 17 monitoring & drift prompt).

---

# Where each ML/DL concept is introduced (quick map)

* **Data wrangling / EDA** — Weeks 1–2 (pandas, plotting)
* **Baselines & evaluation metrics** — Weeks 1–4 (DummyClassifier, accuracy, ROC/AUPRC)
* **Preprocessing & pipelines** — Weeks 3–4 (ColumnTransformer, Pipelines)
* **Feature engineering & selection** — Weeks 5–6
* **Model selection & CV** — Weeks 6–7 (Grid/Random search, learning curves)
* **Experiment tracking & versioning** — Weeks 6–8 (MLflow, DVC)
* **Containerization & Model Cards** — Weeks 8 (Docker + Model Card)
* **PyTorch & DL basics** — Weeks 9–10 (tensors, autograd, backprop intuition)
* **NLP transfer learning** — Weeks 9–10 (small fine-tuning or classical fallback)
* **Time-series forecasting** — Weeks 11 (windowing, metrics)
* **CI & PR workflows** — Week 12 (tests, CI, checksum tests)
* **Serving & APIs** — Weeks 13–14 (FastAPI, schema tests, containerized serving)
* **Recommenders (stretch)** — Weeks 15–16 (collab filtering, ranking metrics)
* **Deployment & monitoring** — Week 17 (GCP deploy docs, drift detection)

Math topics are distributed with the weekly slots:

* Week 2: descriptive stats & probability
* Week 3: linear algebra intuition
* Week 4: probability for evaluation & bootstrap CI
* Week 5: regularization & overfitting intuition
* Week 6: probability distributions & hypothesis testing
* Week 7: cross-validation & bias-variance curves
* Week 8: optimization & gradient basics
* Week 9: derivatives & chain rule intuition
* Week 10: loss landscapes & NN regularization
* Week 11: stationarity & autocorrelation
* Week 12: bias-variance math & CIs
* Week 13: uncertainty propagation & predictive intervals
* Week 14: drift detection stats (KS / divergence)
* Week 15: similarity metrics & ranking evaluation
* Week 16: A/B test basics & significance for ranking metrics
* Week 17: monitoring & drift statistics recap (PSI/KL/KS)

---

# Small Reads (weekly) — consistent with your 4 books

Each week’s "Small Read" picks a 10–20 page slice from one of these required books:

* *100 Pages Machine Learning*
* *Hands-On Machine Learning with Scikit-Learn & TensorFlow* (only scikit parts)
* *Designing Machine Learning Systems*
* *Python Data Science Handbook*
  I included an explicit page range each week above (kept to 30–45 mins reading time).

---

# Notes about time budgets & simplifications

* I integrated the 1-hour weekly math slot starting Week 2 while keeping every week within its allowed cap by trimming small optional resource time or redistributing minor task hours where necessary.
* If any heavy training or operation would exceed \~1 hour CPU, I included a **Small-proxy** alternative (subsample / fewer epochs / classical fallback). These are marked in the weeks where applicable.

---

# Final JSON summary (17 objects) — exact format requested

```json
[
  {"week":1,"title":"Intro: pandas & Jupyter","topics":["pandas","Jupyter","Baseline modeling"],"project":"UCI Lenses: EDA + Dummy baseline"},
  {"week":2,"title":"EDA & Visualization + Math: Descriptive Stats","topics":["EDA","matplotlib","probability"],"project":"UCI Fertility: EDA + basic statistics"},
  {"week":3,"title":"scikit-learn pipelines + Math: Linear Algebra Intuition","topics":["Pipelines","Preprocessing","Linear algebra"],"project":"UCI Cryotherapy: pipeline + baseline"},
  {"week":4,"title":"Capstone #1 + Math: Probability for Evaluation","topics":["Reproducibility","Evaluation","Bootstrap"],"project":"Foundations capstone: small UCI end-to-end"},
  {"week":5,"title":"DVC & Kaggle import + Math: Regularization","topics":["DVC","Kaggle","Regularization"],"project":"Titanic: DVC pipeline + local remote"},
  {"week":6,"title":"Feature Engineering & Model Selection + Math: Hypothesis Testing","topics":["Feature engineering","Model selection","Hypothesis testing"],"project":"Titanic: features + tuned model"},
  {"week":7,"title":"MLflow experiments + Math: CV & Bias-Variance","topics":["MLflow","Experiments","Learning curves"],"project":"Titanic: MLflow tracking + reproducibility"},
  {"week":8,"title":"Docker & Model Card + Math: Optimization Basics","topics":["Docker","Model Card","Optimization"],"project":"Titanic: Dockerfile + Model Card"},
  {"week":9,"title":"PyTorch basics & NLP intro + Math: Derivatives","topics":["PyTorch","NLP basics","Autograd"],"project":"Mini-IMDB/SST2 subset: PyTorch classifier"},
  {"week":10,"title":"Small NLP fine-tuning + Math: Loss Landscapes","topics":["Fine-tuning","Evaluation","Regularization"],"project":"NLP small subset: fine-tune or TF-IDF fallback"},
  {"week":11,"title":"Time-series basics + Math: Stationarity","topics":["Time-series","Forecasting","Evaluation"],"project":"Small time-series: naive vs model forecast"},
  {"week":12,"title":"CI & PR simulation + Math: Bias-Variance & CIs","topics":["CI","Tests","PR workflow"],"project":"Capstone with CI: reproducibility + tests"},
  {"week":13,"title":"FastAPI basics + Math: Uncertainty Propagation","topics":["FastAPI","API schema","Uncertainty"],"project":"Wrap model with FastAPI + API tests"},
  {"week":14,"title":"Containerized serving & API tests + Math: Drift Detection","topics":["Docker for API","API tests","Monitoring"],"project":"Containerized API with healthcheck & tests"},
  {"week":15,"title":"Recommender fundamentals + Math: Similarity & Ranking","topics":["Recommenders","Evaluation","API"],"project":"Small recommender: popularity + CF baseline"},
  {"week":16,"title":"Recommender mini-capstone + Math: A/B Basics","topics":["Serving","CI for API","Evaluation"],"project":"Recommender: API + CI + artifacts"},
  {"week":17,"title":"Portfolio, resumes & GCP deploy + Math: Monitoring & PSI","topics":["Portfolio","GCP deploys","Monitoring"],"project":"Final capstone: Docker -> GCP deploy + portfolio"}
]
```

---

If you want, I can now (pick one — I will do it right away):

1. Expand each week’s **Resources** into 3–4 explicit links per new tool/topic (with emoji, publish year, est time, tag Required/Optional/Stretch).
2. Generate the **Week 8 capstone repo scaffold** (files: `requirements.txt` with pinned versions, `runtime.txt`, `Dockerfile`, `main.py` smoke stub, `notebooks/main.ipynb` outline, `README.md`) that you can copy and run locally.
3. Produce two **resume drafts** (Data Scientist & ML Engineer) tailored to the projects and a short portfolio Notion/GitHub Pages template.

Choose 1, 2, or 3 (or say “all”), and I’ll generate it immediately.
