# GPT-5-High


# 16-Week AI/ML/DL/MLOps Roadmap (12–14 hrs/week)

This plan takes you from zero to job-ready for roles like Data Scientist, ML/AI Engineer, or AI Architect. You’ll learn Python, math, ML/DL, and MLOps by building and deploying real projects every week. Everything uses free resources and emphasizes reproducibility, ethics, and portfolio-building.

---

## 4-Week Block Overview (Condensed)

- Weeks 1–4: Foundations + first end-to-end classical ML
  - Skills: Python, Git/GitHub, NumPy/Pandas, plotting, linear algebra/prob/prob, scikit-learn, metrics, cross-validation, basic DVC.
  - Job tasks: Clean and explore data, train/evaluate baseline models, version code/data, publish reproducible notebooks and reports.
  - Capstone 1: Reproducible scikit-learn pipeline with DVC and thorough evaluation.

- Weeks 5–8: Intermediate ML + deployment basics
  - Skills: Feature engineering, regularization, trees/ensembles, hyperparameter tuning (Optuna), experiment tracking (MLflow), Docker, FastAPI, CI/CD basics.
  - Job tasks: Build robust pipelines, track experiments, package and containerize models, deploy a simple API, add tests and CI.
  - Capstone 2: Deployed ML microservice (Docker + FastAPI) with CI on GitHub Actions and a free hosting platform.

- Weeks 9–12: Deep Learning (PyTorch) + deployed DL service
  - Skills: PyTorch tensors, autograd, MLP/CNN, transfer learning, Transformers for NLP, GPUs on Colab, TorchScript/ONNX, MLflow model registry.
  - Job tasks: Train and fine-tune DL models, deploy an inference service, monitor performance/drift.
  - Capstone 3: Deployed DL model (image or text) with logging/monitoring and model card.

- Weeks 13–16: Production MLOps + LLM/RAG + final production-grade capstone
  - Skills: Orchestration (Prefect), data validation (Great Expectations), monitoring (Evidently), A/B testing, interpretability (SHAP), vector search (FAISS), RAG, autoscaling basics.
  - Job tasks: Build/operate ML pipelines, monitor models, design rollouts, build retrieval-augmented apps.
  - Capstone 4: Full production-grade system with DVC, MLflow, Prefect, Docker, CI/CD, monitoring, docs, and model card.

Compute and deployment (free options you’ll use): Google Colab (enable GPU in Runtime → Change runtime type), Kaggle Notebooks (GPU on), Hugging Face Spaces (Gradio), Render/Fly.io (FastAPI), GitHub Actions (CI), MLflow local server, DVC.

Tools: 
- scikit-learn: consistent API and gold standard for classical ML (https://scikit-learn.org/)
- PyTorch: pythonic, widely used in industry for DL (https://pytorch.org/get-started/locally/)
- FastAPI: fast, typed web API for ML services (https://fastapi.tiangolo.com/)
- DVC: Git-like data/pipeline versioning (https://dvc.org/)
- MLflow: experiment tracking + model registry (https://mlflow.org/)
- Prefect: simple modern orchestration (https://docs.prefect.io/)
- Evidently: open-source monitoring and drift (https://docs.evidentlyai.com/)
- Optuna: efficient hyperparameter optimization (https://optuna.org/)

---

# Weekly Breakdown

## Week 1 — Python, Git, and Data Basics
- Topics to Cover
  - Prerequisites: none.
  - Python syntax, control flow, functions, data structures
  - Jupyter/Colab, NumPy, Pandas, Matplotlib/Seaborn
  - Shell/CLI, Git/GitHub basics, virtual environments
- Estimated time (hours)
  - 13h
- Tasks
  - Environment setup: Python 3.11, VS Code, Jupyter, Git, GitHub account; test Colab (1.5h)
  - Python basics (variables, lists/dicts, loops, functions) using your playlist (3h)
  - NumPy arrays, slicing, broadcasting (2h)
  - Pandas: reading CSVs, describe(), groupby, merges, plotting (2h)
  - Git/GitHub basics: init, add/commit, branches, push/pull, README (2h)
  - Mini-project build session (2.5h)
- Resources (Free Only)
  - Python tutorial: https://docs.python.org/3/tutorial/
  - Google Colab: https://colab.research.google.com/
  - NumPy: https://numpy.org/doc/stable/user/quickstart.html
  - Pandas: https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
  - Matplotlib: https://matplotlib.org/stable/tutorials/introductory/pyplot.html
  - Seaborn: https://seaborn.pydata.org/tutorial.html
  - Git: https://git-scm.com/docs/gittutorial
  - GitHub Hello World: https://docs.github.com/en/get-started/quickstart/hello-world
  - Dataset (Iris, UCI): https://archive.ics.uci.edu/dataset/53/iris
- My resources
  - Python basics playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ
  - Books (reference): Hands-On ML; Designing ML Systems; Hundred-Page ML Book
- Mini Project / Hands-on Assignment
  - Project: Iris EDA and data cleaning
  - Dataset: UCI Iris (above)
  - Deliverables: 
    - GitHub repo with Jupyter notebook
    - README (problem, dataset link, how to run)
    - Short write-up (what you found + simple ethics note)
    - Basic tests: function to check missing values; assert column types
  - Acceptance criteria: 
    - Notebook loads dataset, computes descriptive stats, 3+ plots, saves clean CSV
    - GitHub repo is reproducible (requirements.txt), results explained
  - Scoring rubric: correctness 40%, reproducibility/README 20%, evaluation 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Record dataset source and license; avoid PII; use random seeds where applicable
  - Commit often with meaningful messages; store data file sizes responsibly
- Weekly Outcomes
  - Set up Python/Colab/VS Code; use Git/GitHub; run EDA with NumPy/Pandas/Seaborn
- Stretch goals / Next steps
  - Try Kaggle Notebooks: https://www.kaggle.com/code
  - Add Makefile with simple commands (make setup, make eda)

---

## Week 2 — Math for ML I + First Regression
- Topics to Cover
  - Prerequisites: Week 1 basics in Python/Pandas/Git
  - Linear algebra (vectors/matrices, dot product), calculus (derivative intuition), probability (random variables)
  - NumPy vectorization; scikit-learn basics; unit tests (pytest)
- Estimated time (hours)
  - 12.5h
- Tasks
  - Math primer: LinAlg 2h, Calculus 1h, Probability 1h (4h)
  - NumPy vectorization/broadcasting exercises (2h)
  - scikit-learn: train_test_split, LinearRegression, metrics (2h)
  - Testing: setup pytest; write 2–3 unit tests (1h)
  - Mini-project implementation (3.5h)
- Resources (Free Only)
  - scikit-learn tutorial: https://scikit-learn.org/stable/tutorial/basic/tutorial.html
  - Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
  - pytest: https://docs.pytest.org/en/stable/getting-started.html
  - Diabetes dataset docs: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
- My resources
  - Linear algebra playlist: https://www.youtube.com/playlist?list=PLRDl2inPrWQW1QSWhBU0ki-jq_uElkh2a
  - Calculus playlist: https://www.youtube.com/playlist?list=PLRDl2inPrWQVu2OvnTvtkRpJ-wz-URMJx
  - Probability playlist: https://www.youtube.com/playlist?list=PLRDl2inPrWQWwJ1mh4tCUxlLfZ76C1zge
  - 100 Days of ML: https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH
- Mini Project / Hands-on Assignment
  - Project: Linear Regression from scratch vs scikit-learn
  - Dataset: scikit-learn Diabetes
  - Tasks: implement univariate gradient descent in NumPy, then compare with LinearRegression; compute MSE/R2
  - Deliverables: GitHub repo; notebook; README; write-up; pytest for loss decreasing and shapes
  - Acceptance criteria: from-scratch model converges; sklearn baseline reported; explain bias/variance intuition
  - Rubric: same as Week 1
- Ethics & Best Practices
  - Track random seeds; include requirements.txt; document assumptions
- Weekly Outcomes
  - Implement simple GD, evaluate regression, write basic tests
- Stretch goals / Next steps
  - Add k-fold cross-validation using scikit-learn

---

## Week 3 — Classification, Metrics, and Cross-Validation
- Topics to Cover
  - Prerequisites: Week 2 regression and metrics
  - Logistic regression, classification metrics (accuracy, precision/recall, F1, ROC-AUC)
  - Scaling (StandardScaler), Pipelines, cross-validation and grid search
- Estimated time (hours)
  - 13h
- Tasks
  - Study: classification metrics and confusion matrix (1.5h)
  - Logistic regression with scikit-learn; scaling and pipelines (2h)
  - Cross-validation/grid search (2h)
  - Feature engineering for tabular data (2h)
  - Titanic Kaggle: EDA + modeling + submit (5.5h)
- Resources (Free Only)
  - Classification tutorial: https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
  - Pipelines: https://scikit-learn.org/stable/modules/compose.html
  - Kaggle Titanic: https://www.kaggle.com/c/titanic
- My resources
  - 100 Days of ML: https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH
  - Hands-On ML (reference)
- Mini Project / Hands-on Assignment
  - Project: Titanic survival classifier
  - Dataset: Kaggle Titanic
  - Deliverables: repo + notebook + README, cross-val results, log metrics, short ethics write-up (data bias and feature leakage)
  - Acceptance criteria: >0.78 public LB with simple model; use Pipeline; include cross-validation
  - Rubric: same schema
- Ethics & Best Practices
  - Avoid target leakage; document imputation strategy; reproducible seeds; do not store Kaggle API token in repo
- Weekly Outcomes
  - Train logistic regression with pipelines; evaluate with CV; submit to Kaggle
- Stretch goals / Next steps
  - Add PolynomialFeatures; compare with tree-based baseline

---

## Week 4 — Capstone 1: Reproducible Classical ML Pipeline
- Topics to Cover
  - Prerequisites: Weeks 1–3
  - DVC basics (data versioning), scikit-learn pipelines, model evaluation report
- Estimated time (hours)
  - 13h
- Tasks
  - DVC intro: install, dvc init, dvc add data (2h)
  - Build end-to-end pipeline on California Housing (7.5h)
    - EDA, feature engineering
    - Pipeline with ColumnTransformer
    - CV and grid search
    - Save model with joblib; record metrics
    - DVC stage for preprocess/train
  - Write report + README + refactor repo structure (2h)
  - Self-assessment & mock interview prep (1.5h)
- Resources (Free Only)
  - DVC start: https://dvc.org/doc/start
  - California Housing: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
  - Joblib: https://joblib.readthedocs.io/en/latest/
- My resources
  - Designing Machine Learning Systems (reference)
- Mini Project / Hands-on Assignment
  - Capstone: California Housing price prediction
  - Deliverables: repo; notebook; modular scripts (src/); DVC pipeline (dvc.yaml); requirements.txt; metrics.json; README; write-up; unit tests for data transformations
  - Acceptance criteria: RMSE reported via cross-val; dvc repro works end-to-end
  - Rubric: correctness 35%, reproducibility/DVC 25%, evaluation 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Version raw/processed data; lock random seeds; document limitations
- Weekly Outcomes
  - Create reproducible ML project with DVC and scikit-learn
- Stretch goals / Next steps
  - Add pre-commit hooks (black/ruff), badge for CI (to be added in Week 8)

- Checkpoint: Self-Assessment & Interview Prep
  - Self-check: Explain bias-variance; implement/train/evaluate reg/logistic; run k-fold CV; use DVC to version data
  - Quizzes/exercises: scikit-learn tutorial exercises + Kaggle micro-courses (Python, Pandas, ML)
  - Mock interview Qs:
    - What is cross-validation and why use it?
    - Difference between precision, recall, and when each matters?
    - How do you prevent leakage?
  - Portfolio: Publish Weeks 1–4 projects on GitHub; pin repositories; add project summaries to LinkedIn

---

## Week 5 — Feature Engineering, Regularization, and DVC Pipelines
- Topics to Cover
  - Prerequisites: Capstone 1
  - Feature engineering (encoding, scaling, interactions), regularization (L1/L2, ElasticNet), DVC pipelines with multiple stages, MLflow intro
- Estimated time (hours)
  - 13h
- Tasks
  - Study feature engineering & regularization (3h)
  - scikit-learn Pipeline/ColumnTransformer practice (1.5h)
  - DVC pipelines (multiple stages) (2h)
  - MLflow quickstart for experiment tracking (1.5h)
  - Mini-project: Bike Sharing Demand (5h)
- Resources (Free Only)
  - Feature engineering: https://scikit-learn.org/stable/modules/preprocessing.html
  - Regularization: https://scikit-learn.org/stable/modules/linear_model.html
  - DVC pipelines: https://dvc.org/doc/user-guide/pipelines
  - MLflow quickstart: https://mlflow.org/docs/latest/getting-started/quickstart-tracking.html
  - Kaggle Bike Sharing: https://www.kaggle.com/c/bike-sharing-demand
- My resources
  - Hands-On ML (reference)
  - 100 Days of ML playlist
- Mini Project / Hands-on Assignment
  - Project: Predict bike rentals with regularized linear models
  - Deliverables: repo; notebook; src/; dvc.yaml with preprocess->train->evaluate; MLflow runs with hyperparams; README; tests for preprocessing; write-up
  - Acceptance criteria: Baseline vs regularized model comparison; DVC pipeline runs; MLflow logs (params/metrics/artifacts)
  - Rubric: correctness 35%, reproducibility/DVC/MLflow 25%, evaluation 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Explain feature choices; log feature importance; ensure metrics use CV; version data and models
- Weekly Outcomes
  - Build multi-stage DVC pipeline; track experiments with MLflow; apply regularization
- Stretch goals / Next steps
  - Try ElasticNet with nested CV; log runs comparison in MLflow

---

## Week 6 — Trees, Ensembles, Imbalanced Data + Docker Intro
- Topics to Cover
  - Prerequisites: Week 5 pipelines and MLflow
  - Decision trees, Random Forests, Gradient Boosting (XGBoost), handling imbalanced data (class weights, SMOTE)
  - Docker basics for reproducible runtime
- Estimated time (hours)
  - 12.5h
- Tasks
  - Study trees/ensembles (2.5h)
  - Imbalanced learning techniques (1.5h)
  - Docker install + build/run basics (3h)
  - Mini-project: Credit Card Fraud detection (5.5h)
- Resources (Free Only)
  - Tree-based methods: https://scikit-learn.org/stable/modules/ensemble.html
  - imbalanced-learn: https://imbalanced-learn.org/stable/
  - XGBoost docs: https://xgboost.readthedocs.io/en/stable/
  - Docker install: https://docs.docker.com/get-docker/
  - Kaggle Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- My resources
  - Hands-On ML (reference)
  - 100 Days of ML playlist
- Mini Project / Hands-on Assignment
  - Project: Fraud detection with RF/XGBoost + Dockerized inference
  - Deliverables: repo; notebook; src/; MLflow runs; Dockerfile to run predict.py with JSON input; README; tests for metrics and API schema; write-up
  - Acceptance criteria: PR-AUC reported; class imbalance handled; docker build and docker run works locally
  - Rubric: correctness 35%, reproducibility/Docker 25%, evaluation 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Address false positives/negatives trade-offs; document class imbalance handling; include model threshold rationale
- Weekly Outcomes
  - Train ensemble models; handle imbalance; containerize inference
- Stretch goals / Next steps
  - Add SHAP feature importance for tree model

---

## Week 7 — Hyperparameter Tuning, Packaging, and FastAPI Serving
- Topics to Cover
  - Prerequisites: Week 6 Docker + MLflow
  - Advanced CV, Optuna HPO, MLflow tracking advanced, packaging code (setup.cfg/pyproject), FastAPI for inference
- Estimated time (hours)
  - 12.5h
- Tasks
  - Advanced CV + Optuna intro (2.5h)
  - MLflow tracking improvements (tags, artifacts) (1.5h)
  - Package training/inference as a module; CLI with Typer or argparse (2h)
  - Build FastAPI service + request/response schema (2h)
  - Mini-project: Churn prediction E2E (4.5h)
- Resources (Free Only)
  - Optuna: https://optuna.org/ and tutorial https://optuna.readthedocs.io/en/stable/tutorial/index.html
  - FastAPI: https://fastapi.tiangolo.com/
  - Typer: https://typer.tiangolo.com/
  - Telco Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
  - MLflow tracking API: https://mlflow.org/docs/latest/python_api/mlflow.html
- My resources
  - Designing Machine Learning Systems (reference)
- Mini Project / Hands-on Assignment
  - Project: Churn classifier with Optuna + FastAPI
  - Deliverables: repo; src package; Optuna study; MLflow logs; FastAPI app with POST /predict; tests (pytest) for data validation and one endpoint test; README; write-up
  - Acceptance criteria: Achieve >0.80 ROC-AUC; API runs locally (uvicorn) and returns valid JSON; CLI predict works
  - Rubric: correctness 35%, packaging/API 25%, evaluation 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Input validation on API; log model version; explain how to handle PII; include rate limiting suggestion
- Weekly Outcomes
  - Tune models with Optuna; package code; serve with FastAPI
- Stretch goals / Next steps
  - Add pydantic schemas for strict validation; cache model on startup

---

## Week 8 — Capstone 2: Deployed ML Microservice + CI/CD
- Topics to Cover
  - Prerequisites: Weeks 5–7
  - CI with GitHub Actions, deploy Dockerized FastAPI to Render or Fly.io, basic logging/monitoring
- Estimated time (hours)
  - 13.5h
- Tasks
  - Refactor project (from Week 7 or 6) to production layout (1.5h)
  - Add CI (pytest + lint + type-check) via GitHub Actions (2h)
  - Container hardening (small base image, pinned versions) (1h)
  - Deploy to Render (Docker) or Hugging Face Spaces (Gradio alternative) (3h)
  - Add request logging, healthcheck, basic metrics endpoint (/metrics or log latency) (2h)
  - Document deployment, add Makefile and scripts (2h)
  - Assessment + portfolio updates (2h)
- Resources (Free Only)
  - GitHub Actions Python CI: https://github.com/actions/starter-workflows/blob/main/ci/python-package.yml
  - Render (Docker): https://render.com/docs/docker
  - Fly.io: https://fly.io/docs/
  - Logging in Python: https://docs.python.org/3/howto/logging.html
- My resources
  - Designing Machine Learning Systems (reference)
- Mini Project / Hands-on Assignment
  - Capstone: Deploy churn/fraud service publicly
  - Deliverables: repo; Dockerfile; FastAPI app; CI workflow; deployed URL; README with run/deploy steps; write-up; simple load test script; model card (first version)
  - Acceptance criteria: Public endpoint live; CI passes on PR; logs capture request count and latency; model card included
  - Rubric: correctness 25%, deployment/CI 30%, reproducibility 20%, ethics/model card 15%, code/tests 10%
- Ethics & Best Practices
  - No sensitive data in logs; redact fields; display version and deprecation policy
- Weekly Outcomes
  - Ship a public ML API with CI and basic monitoring
- Stretch goals / Next steps
  - Add auto-deploy on tag; simple rate limiting with middleware

- Checkpoint: Self-Assessment & Interview Prep
  - Self-check: Build/train/tune a model, containerize, deploy, and set up CI from scratch
  - Mock interview Qs:
    - Explain how you’d design an ML pipeline for churn prediction end-to-end
    - What’s the difference between PR-AUC and ROC-AUC?
    - How do you structure a FastAPI prediction service?
  - Portfolio: Add deployed link to README/LinkedIn; include short demo video/gifs

---

## Week 9 — PyTorch Foundations and MLPs
- Topics to Cover
  - Prerequisites: Capstone 2 deployment experience
  - PyTorch tensors, datasets/dataloaders, autograd, training loop, optimizers
- Estimated time (hours)
  - 13h
- Tasks
  - PyTorch quickstart + Colab GPU setup (3h)
  - Autograd and manual training loop (2h)
  - Optimizers, schedulers, early stopping pattern (2h)
  - Mini-project: Fashion-MNIST MLP (6h)
- Resources (Free Only)
  - PyTorch Quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
  - Dataset/DataLoader: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
  - Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
  - Enable GPU in Colab: Runtime → Change runtime type → GPU
- My resources
  - 100 Days of DL: https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn
  - Hands-On ML (DL chapters as reference)
- Mini Project / Hands-on Assignment
  - Project: Train an MLP on Fashion-MNIST with PyTorch
  - Deliverables: repo; notebook; training loop with metrics; MLflow run; save best model; README; tests for dataset shapes and forward pass
  - Acceptance criteria: >88% test accuracy; runs on Colab GPU; seeds fixed; learning curve plotted
  - Rubric: correctness 40%, reproducibility 20%, evaluation 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Log hyperparams; control randomness (torch.manual_seed); ensure data license compliance
- Weekly Outcomes
  - Implement full DL training loop; track results; save/load models
- Stretch goals / Next steps
  - Export to ONNX and verify inference

---

## Week 10 — CNNs and Transfer Learning
- Topics to Cover
  - Prerequisites: Week 9 PyTorch basics
  - CNNs (conv/pool), data augmentation, transfer learning with torchvision models, TorchScript/ONNX
- Estimated time (hours)
  - 13h
- Tasks
  - Study CNNs and transf. learning (2.5h)
  - Implement augmentation with torchvision transforms (1h)
  - Fine-tune ResNet18 on CIFAR-10 (4h)
  - Export model to TorchScript or ONNX (1h)
  - Serve with FastAPI + Docker (4.5h)
- Resources (Free Only)
  - CNN tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
  - Transfer learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
  - TorchScript: https://pytorch.org/docs/stable/jit.html
  - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- My resources
  - 100 Days of DL playlist
- Mini Project / Hands-on Assignment
  - Project: CIFAR-10 image classifier + API
  - Deliverables: repo; notebook; training script; MLflow runs; FastAPI service; Dockerfile; README; tests for inference shape/type; model card v1
  - Acceptance criteria: >85% accuracy; Dockerized inference works locally; TorchScript/ONNX artifact included; model card filled
  - Rubric: correctness 35%, model artifacts 20%, deployment 25%, ethics/model card 10%, code/tests 10%
- Ethics & Best Practices
  - Note dataset biases; document misuse risks; control data augmentation randomness
- Weekly Outcomes
  - Fine-tune CNN; export model; serve via API in Docker
- Stretch goals / Next steps
  - Try TorchServe: https://pytorch.org/serve/

---

## Week 11 — NLP with Transformers
- Topics to Cover
  - Prerequisites: Week 10 PyTorch/tensors
  - Tokenization, Transformer basics, Hugging Face Datasets + Transformers Trainer API
- Estimated time (hours)
  - 13h
- Tasks
  - HF Datasets + tokenizers (2h)
  - Transformer fine-tuning with Trainer (2h)
  - Metrics for text classification (accuracy/F1) (1h)
  - Mini-project: DistilBERT on IMDB (6h)
  - Serve demo (Gradio or FastAPI) (2h)
- Resources (Free Only)
  - HF Datasets: https://huggingface.co/docs/datasets
  - Transformers course: https://huggingface.co/learn/nlp-course/chapter1/1
  - IMDB dataset: https://huggingface.co/datasets/imdb
  - Gradio: https://www.gradio.app/
- My resources
  - 100 Days of DL playlist
- Mini Project / Hands-on Assignment
  - Project: Sentiment classifier with DistilBERT
  - Deliverables: repo; training notebook; metrics; MLflow logs (use custom logging if needed); Gradio or FastAPI demo; Dockerfile; README; tests for tokenizer and inference; model card v1
  - Acceptance criteria: >0.90 accuracy on IMDB test; demo runs; model card includes data/ethics/risk
  - Rubric: correctness 35%, deployment/demo 25%, reproducibility 20%, ethics/model card 10%, code/tests 10%
- Ethics & Best Practices
  - Address toxicity/bias risks; include disclaimer; verify license for pretrained model
- Weekly Outcomes
  - Fine-tune BERT-class model and demo it
- Stretch goals / Next steps
  - Quantize model for faster inference (onnxruntime or bitsandbytes)

---

## Week 12 — Capstone 3: Deployed DL Service with Monitoring
- Topics to Cover
  - Prerequisites: Weeks 9–11
  - MLflow model registry basics, monitoring/logging, drift detection (Evidently), simple persistence of logs
- Estimated time (hours)
  - 13h
- Tasks
  - Choose your DL model (CIFAR-10 or IMDB) for production hardening (0.5h)
  - Refactor training/eval; register model version (MLflow or tag) (2h)
  - Deploy service (Render/HF Spaces) (3h)
  - Add monitoring: log inputs/outputs/latency; nightly drift report with Evidently (4h)
  - Write model card v2 + report + README (2h)
  - Assessment and mock interview (1.5h)
- Resources (Free Only)
  - MLflow models: https://mlflow.org/docs/latest/models.html
  - Evidently: https://docs.evidentlyai.com/
  - SQLite quickstart (for logs): https://sqlite.org/quickstart.html
  - HF Spaces: https://huggingface.co/spaces
- My resources
  - Hands-On ML; Designing ML Systems (reference)
- Mini Project / Hands-on Assignment
  - Capstone: DL model with monitoring
  - Deliverables: repo; deployed service; logging (file or SQLite); Evidently drift report artifact; CI tests; model card v2; README
  - Acceptance criteria: Live demo; logged predictions; drift report generated; clear rollback plan in README
  - Rubric: correctness 25%, deployment/monitoring 30%, reproducibility 20%, ethics/model card 15%, code/tests 10%
- Ethics & Best Practices
  - Redact inputs in logs if sensitive; document drift response plan; note training/inference mismatch risks
- Weekly Outcomes
  - Operate a deployed DL service with simple monitoring and model registry/tagging
- Stretch goals / Next steps
  - Add Prometheus + Grafana for metrics (optional)

- Checkpoint: Self-Assessment & Interview Prep
  - Self-check: Implement DL training loop; export model; deploy; add monitoring; write a model card
  - Mock interview Qs:
    - Explain transfer learning and when to freeze vs finetune
    - How would you detect data drift in production?
    - Compare TorchScript vs ONNX
  - Portfolio: Add Capstone 3 to LinkedIn/Resume with metrics and live demo

---

## Week 13 — Orchestration and Data Validation
- Topics to Cover
  - Prerequisites: Capstone 3
  - Prefect orchestration, Great Expectations data validation, artifact storage patterns
- Estimated time (hours)
  - 13h
- Tasks
  - Prefect flows and blocks (3h)
  - Great Expectations suite (2h)
  - Artifact management patterns (local/S3-compatible) (1h)
  - Mini-project: NYC Taxi fare pipeline (7h)
- Resources (Free Only)
  - Prefect: https://docs.prefect.io/
  - Great Expectations: https://docs.greatexpectations.io/
  - NYC Taxi Trip Duration: https://www.kaggle.com/c/nyc-taxi-trip-duration
  - MinIO (optional local S3): https://min.io/
- My resources
  - Designing Machine Learning Systems (reference)
  - 100 Days of ML playlist
- Mini Project / Hands-on Assignment
  - Project: Orchestrated pipeline (ingest→validate→feature→train→evaluate→register)
  - Deliverables: repo; prefect flow; GE validation suite; MLflow logs; DVC for data; README; tests for tasks; write-up
  - Acceptance criteria: prefect deployment runs end-to-end; failed validation stops training; metrics reported
  - Rubric: correctness 30%, orchestration/validation 30%, reproducibility 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Validate schema and ranges; document how bad data is handled; maintain data/version lineage
- Weekly Outcomes
  - Build a scheduled, validated ML pipeline with Prefect + GE
- Stretch goals / Next steps
  - Parameterize flows and schedule daily runs

---

## Week 14 — Monitoring, Interpretability, and A/B Testing
- Topics to Cover
  - Prerequisites: Week 13 pipelines
  - Model monitoring (Evidently), interpretability (SHAP), AB testing/canary and rollback strategies
- Estimated time (hours)
  - 13h
- Tasks
  - Monitoring dashboard/report with Evidently (2h)
  - Interpretability with SHAP on tree/GBM model (2.5h)
  - Design A/B test plan; canary rollout and rollback procedures (1.5h)
  - Mini-project: Instrument prior API with monitoring + SHAP report (6h)
- Resources (Free Only)
  - Evidently: https://docs.evidentlyai.com/
  - SHAP: https://shap.readthedocs.io/en/latest/
  - Canary/A/B blog (concepts): https://martinfowler.com/bliki/CanaryRelease.html
- My resources
  - Designing Machine Learning Systems; Hundred-Page ML Book (reference)
- Mini Project / Hands-on Assignment
  - Project: Monitoring + Interpretability Pack
  - Deliverables: repo updates; Evidently dashboards; SHAP plots; README with A/B plan; tests for monitoring job
  - Acceptance criteria: Monitoring job runs and outputs report; SHAP explanation included; documented rollback plan
  - Rubric: correctness 30%, monitoring/interpretability 30%, reproducibility 20%, ethics 10%, code/tests 10%
- Ethics & Best Practices
  - Discuss fairness metrics; avoid exposure of sensitive attributes; include human-in-the-loop review suggestion
- Weekly Outcomes
  - Add monitoring and interpretability to an existing service; plan safe rollouts
- Stretch goals / Next steps
  - Add automated alerts (email/Slack webhook) when drift exceeds threshold

---

## Week 15 — LLMs, Embeddings, and RAG
- Topics to Cover
  - Prerequisites: Week 11 Transformers basics
  - Text embeddings (Sentence Transformers), FAISS vector search, building a RAG pipeline (retrieve → generate)
- Estimated time (hours)
  - 12.5h
- Tasks
  - Embeddings with sentence-transformers (3h)
  - FAISS index build/search (2h)
  - Mini-project: RAG over your project READMEs or scikit-learn docs subset (5.5h)
  - Deploy demo to Hugging Face Spaces (2h)
- Resources (Free Only)
  - Sentence Transformers: https://www.sbert.net/
  - FAISS: https://faiss.ai/
  - FLAN-T5 small (local inference): https://huggingface.co/google/flan-t5-small
  - Simple RAG tutorial (HF): https://huggingface.co/blog/retrieval-augmented-generation
  - HF Spaces: https://huggingface.co/spaces
- My resources
  - 100 Days of DL playlist
- Mini Project / Hands-on Assignment
  - Project: Local RAG assistant over small doc set
  - Deliverables: repo; notebook; index build script; Gradio UI; Dockerfile (optional); README; tests for retrieval quality (top-k recall on synthetic Q/A)
  - Acceptance criteria: End-to-end RAG answers questions grounded in docs; retrieval eval (at least synthetic)
  - Rubric: correctness 35%, retrieval quality 20%, deployment/demo 20%, reproducibility 15%, ethics 10%
- Ethics & Best Practices
  - Cite sources; display confidence/limitations; block prompt injection basics
- Weekly Outcomes
  - Build a small RAG app with local embedding + generation model
- Stretch goals / Next steps
  - Add reranking (cross-encoder) for better retrieval

---

## Week 16 — Capstone 4: Production-Grade ML System
- Topics to Cover
  - Prerequisites: Weeks 13–15
  - Full E2E MLOps: DVC + Prefect + MLflow + Docker + FastAPI + CI/CD + monitoring + model card + autoscaling basics
- Estimated time (hours)
  - 13h
- Tasks
  - Choose problem (churn/taxi/image/text). Define clear KPIs and SLAs (0.5h)
  - Build pipeline: DVC data versioning, Prefect orchestration, MLflow experiment/registry (5h)
  - API service: FastAPI + Docker; add logging/metrics; model version endpoint (3h)
  - CI/CD: GitHub Actions (tests, lint, docker build); deploy to Render/Fly.io/HF Spaces (2h)
  - Monitoring: Evidently nightly job + alerts; model card v3; documentation (2.5h)
- Resources (Free Only)
  - CI/CD refresher: https://github.com/actions/starter-workflows
  - Render Deploy Guide: https://render.com/docs/deploy-an-app
  - Autoscaling basics (Render): https://render.com/docs/scaling (concepts)
- My resources
  - All three books (reference)
- Mini Project / Hands-on Assignment
  - Capstone: Production-grade ML system
  - Deliverables: mono-repo with:
    - src package, tests, Makefile, requirements.txt
    - DVC pipeline and data remote (can be local/remote)
    - Prefect flow (ingest/validate/train/eval/register/deploy step optional)
    - MLflow runs + model registry/tagging
    - FastAPI + Dockerfile
    - GitHub Actions CI
    - Deployment URL
    - Evidently monitoring job and sample report
    - Model card v3 + comprehensive README (architecture diagram, ops runbook)
  - Acceptance criteria: One-click (make all or dvc repro + prefect deployment) to retrain; CI green; API live; monitoring report produced
  - Rubric: correctness 25%, ops/infrastructure 35%, reproducibility 20%, ethics/model card 10%, code/tests 10%
- Ethics & Best Practices
  - Clear data usage policy; PII handling; fairness note; incident response and rollback; reproducibility checklist
- Weekly Outcomes
  - Deliver a job-ready, production-style ML project
- Stretch goals / Next steps
  - Canary rollout with two model versions; autoscaling experiment; add IaC (Terraform) for infra (optional)

- Final Interview-Readiness Checklist (tasks you should be able to do)
  - Implement logistic regression and softmax from scratch with NumPy
  - Explain/implement k-fold and nested CV
  - Feature engineering with ColumnTransformer/Pipelines
  - Tune hyperparameters with Optuna; compare runs
  - Package code; write CLI; write pytest unit tests and simple property tests
  - Dockerize a FastAPI model service; add health/metrics endpoints
  - Set up GitHub Actions for tests/lint/build
  - Use DVC to version data and define pipelines
  - Run MLflow tracking; register models; load by version
  - Train and fine-tune a CNN and a Transformer (DistilBERT) and export to TorchScript/ONNX
  - Deploy APIs to Render/Fly.io or demos to Hugging Face Spaces
  - Add Evidently monitoring and interpretability with SHAP
  - Design an A/B test and rollback plan
  - Build a Prefect-orchestrated pipeline with validation (Great Expectations)
  - Build a simple RAG system with embeddings + FAISS
  - Write a comprehensive model card; discuss fairness, drift, and failure modes

---

# Weekly Ethics & Reproducibility Checklist (use weekly)
- Document dataset source and license; note PII presence
- Set and log random seeds; record package versions (requirements.txt or pip freeze)
- Version code (Git) and data/models (DVC/MLflow)
- Include evaluation metrics and confidence intervals where possible
- Explain limitations, potential biases, and misuse risks
- Provide instructions to reproduce (README + steps/commands)
- Include tests for critical code paths (data transforms, model I/O)
- Avoid committing secrets; use environment variables

---

# Free Compute/Deploy Notes (use throughout)
- Google Colab (GPU): Runtime → Change runtime type → Hardware accelerator = GPU
- Kaggle Notebooks (GPU): “Settings” → Accelerator → GPU
- Hugging Face Spaces: free hosting for Gradio apps; push via Git repo
- Render/Fly.io: free tiers for small FastAPI services; link your GitHub repo
- Local MLflow UI: run `mlflow ui` and open http://127.0.0.1:5000
- DVC remotes: local filesystem or free cloud storage you configure; don’t push large data to GitHub

---

# Portfolio Project Suggestions (8–10 strong items)
- Iris EDA (Week 1)
- Diabetes linear regression from scratch vs sklearn (Week 2)
- Titanic classification with pipelines and CV (Week 3)
- California Housing DVC pipeline (Capstone 1)
- Bike Sharing with DVC + MLflow (Week 5)
- Fraud detection with Dockerized inference (Week 6)
- Churn prediction with Optuna + FastAPI (Week 7/8 deployed)
- CNN CIFAR-10 classifier + API (Week 10)
- DistilBERT sentiment + demo (Week 11/12 deployed)
- RAG mini-app (Week 15)
- Final Production-grade system (Week 16)

Publish each with:
- Clear README (problem, data, approach, results, how to run)
- Screenshots/GIFs or demo links
- Model card (for deployed/advanced projects)
- Issues/todos for future improvements

---

## JSON Summary (machine-readable)

```json
[
  {"week": 1, "title": "Python, Git, and Data Basics", "top_topics": ["Python & Jupyter", "NumPy/Pandas", "Git/GitHub"], "project": "Iris EDA and data cleaning with plots and simple tests"},
  {"week": 2, "title": "Math for ML I + First Regression", "top_topics": ["LinAlg/Calc/Prob basics", "NumPy vectorization", "Linear Regression"], "project": "Linear regression from scratch vs scikit-learn on Diabetes"},
  {"week": 3, "title": "Classification, Metrics, and Cross-Validation", "top_topics": ["Logistic Regression", "Metrics & CV", "Pipelines"], "project": "Titanic classifier with CV and Kaggle submission"},
  {"week": 4, "title": "Capstone 1: Reproducible Classical ML Pipeline", "top_topics": ["DVC basics", "Pipelines", "Evaluation reports"], "project": "California Housing pipeline with DVC stages and metrics"},
  {"week": 5, "title": "Feature Engineering, Regularization, and DVC Pipelines", "top_topics": ["Feature Engineering", "Regularization", "MLflow intro"], "project": "Bike Sharing demand with DVC stages and MLflow runs"},
  {"week": 6, "title": "Trees, Ensembles, Imbalanced Data + Docker Intro", "top_topics": ["Random Forest/XGBoost", "Imbalanced learning", "Docker"], "project": "Credit Card Fraud detection with Dockerized inference"},
  {"week": 7, "title": "Hyperparameter Tuning, Packaging, and FastAPI Serving", "top_topics": ["Optuna HPO", "Packaging", "FastAPI"], "project": "Churn prediction with Optuna and FastAPI API"},
  {"week": 8, "title": "Capstone 2: Deployed ML Microservice + CI/CD", "top_topics": ["CI/CD", "Docker deploy", "Logging/Monitoring"], "project": "Public FastAPI ML service with CI on GitHub Actions"},
  {"week": 9, "title": "PyTorch Foundations and MLPs", "top_topics": ["Tensors & Autograd", "Training Loops", "MLP"], "project": "Fashion-MNIST MLP with MLflow tracking"},
  {"week": 10, "title": "CNNs and Transfer Learning", "top_topics": ["CNNs", "Data Augmentation", "Transfer Learning"], "project": "CIFAR-10 ResNet fine-tune + TorchScript/ONNX + API"},
  {"week": 11, "title": "NLP with Transformers", "top_topics": ["Tokenization", "Transformers Trainer", "Text Metrics"], "project": "DistilBERT IMDB sentiment with Gradio/FastAPI demo"},
  {"week": 12, "title": "Capstone 3: Deployed DL Service with Monitoring", "top_topics": ["Model Registry", "Monitoring", "Drift Detection"], "project": "Deployed DL model with logs and Evidently drift report"},
  {"week": 13, "title": "Orchestration and Data Validation", "top_topics": ["Prefect", "Great Expectations", "Artifacts"], "project": "NYC Taxi pipeline with validation and MLflow tracking"},
  {"week": 14, "title": "Monitoring, Interpretability, and A/B Testing", "top_topics": ["Evidently", "SHAP", "A/B Testing"], "project": "Instrument API with monitoring dashboard and SHAP report"},
  {"week": 15, "title": "LLMs, Embeddings, and RAG", "top_topics": ["Embeddings", "FAISS", "RAG"], "project": "RAG app over docs using SentenceTransformers + FLAN-T5"},
  {"week": 16, "title": "Capstone 4: Production-Grade ML System", "top_topics": ["DVC+Prefect+MLflow", "Docker+FastAPI+CI/CD", "Monitoring+Model Card"], "project": "Full E2E production ML system with deployment and monitoring"}
]
```