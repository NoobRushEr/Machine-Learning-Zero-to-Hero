# quen-3

# 🚀 16-Week Master Plan: From Zero to AI/ML Engineer

---

## 🔍 1-Page Roadmap Overview (Every 4 Weeks)

| **Weeks** | **Core Skills Achieved** | **Real-World Relevance** |
|---------|--------------------------|--------------------------|
| **Weeks 1–4**<br>Foundations of Python, Math, and Data | Learn Python programming, essential linear algebra, calculus, probability, and data manipulation with pandas. Build first data analysis projects. | Core for all roles: writing clean code, understanding data, exploratory analysis — foundational for ML pipelines and debugging. |
| **Weeks 5–8**<br>Classical ML & Model Evaluation | Train and evaluate regression, classification, and clustering models using scikit-learn. Understand bias-variance, cross-validation, hyperparameter tuning. | Daily work of Data Scientists and ML Engineers: building reliable models, avoiding overfitting, interpreting results. |
| **Weeks 9–12**<br>Deep Learning & Neural Networks | Implement feedforward, convolutional (CNN), and recurrent (RNN) networks in PyTorch. Work with images and text. Use transfer learning. | Required for AI Engineers and Architects: vision, NLP, model design, and deployment in production systems. |
| **Weeks 13–16**<br>MLOps & Full ML Lifecycle | Version control (Git + DVC), experiment tracking (MLflow), containerization (Docker), deployment (FastAPI + Render), monitoring, CI/CD basics. | Critical for ML Engineers and MLOps roles: deploy models, track performance, automate workflows, ensure reproducibility. |

By the end of **Week 16**, you will have:
- ✅ Built **8–10 portfolio projects** on GitHub
- ✅ Deployed at least **2 models** to the cloud
- ✅ Mastered **end-to-end ML lifecycle** from idea to deployment
- ✅ Gained hands-on experience with **real datasets, ethics, testing, and documentation**
- ✅ Be **job-ready** for roles like **Data Scientist, ML Engineer, AI Engineer**

---

# 📘 Full 16-Week Study Plan (12–14 hrs/week)

> **Prerequisites:** None. Starts from zero.

> **Compute Platform Suggestions:** Use [Google Colab (free)](https://colab.research.google.com) for all notebook-based work. Enable GPU under `Runtime > Change runtime type`. For deployment, use [Render (free tier)](https://render.com) or [Fly.io (free credits)](https://fly.io).

> **Version Control:** Use **GitHub** for all projects. Initialize every project with a `.gitignore`, `README.md`, and license.

> **Project Deliverables (Every Week):**
- ✅ GitHub repo
- ✅ Jupyter/Python notebook or script
- ✅ `README.md`: problem, dataset link, how to run
- ✅ Short write-up: results + ethical considerations
- ✅ Evaluation metrics or tests
- Later: Dockerfile, model card, CI setup

---

## Week 1 → Week 16 Breakdown

---

### **Week 1: Introduction to Python Programming**

**Topics to Cover**
- Variables, data types, loops, conditionals
- Functions, error handling
- Basic file I/O
- Introduction to Jupyter Notebooks

**Estimated time (hours):** 13

**Tasks**
- [2 hrs] Install Python via [Anaconda](https://www.anaconda.com/products/distribution) or use Google Colab
- [3 hrs] Learn basics: variables, strings, lists, dictionaries
- [3 hrs] Practice loops (`for`, `while`) and conditionals (`if-elif-else`)
- [2 hrs] Write functions and handle exceptions
- [3 hrs] Mini-project: Build a CLI quiz app or expense tracker

**Resources (Free Only)**
- [Python for Beginners – Real Python](https://realpython.com/start-here/)
- [Python Tutorial – W3Schools](https://www.w3schools.com/python/)

**My resources**  
(Leave blank unless specified)

**Mini Project / Hands-on Assignment**  
**Project:** Personal Expense Tracker CLI  
**Dataset:** None (user input)  
**Deliverables:**
- GitHub repo with `expense_tracker.py`
- `README.md`: describe functionality and how to run
- Write-up: explain design choices and limitations
- Simple test: assert sum matches expected total

**Rubric (100 pts):**
- Correctness (40%)
- Reproducibility/README (20%)
- Evaluation/Test (20%)
- Ethics/Discussion (10%)
- Code Quality (10%)

**Ethics & Best Practices**
- [ ] Avoid storing sensitive data without encryption
- [ ] Document assumptions clearly
- [ ] Use descriptive variable names
- [ ] Handle user errors gracefully

**Weekly Outcomes**
- Can write basic Python scripts
- Can define functions and handle errors
- Can create a simple command-line program

**Stretch goals / Next steps**
- Add CSV export functionality
- Create a menu interface

**Prerequisites:** None

---

### **Week 2: Data Structures & Libraries in Python**

**Topics to Cover**
- NumPy arrays and operations
- Pandas Series and DataFrames
- Reading/writing CSV, JSON
- Basic data cleaning

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn NumPy: arrays, indexing, math operations
- [4 hrs] Learn Pandas: load data, filter, group, aggregate
- [3 hrs] Clean a messy dataset (handle NaN, duplicates)
- [3 hrs] Mini-project: Analyze a dataset (e.g., Titanic)

**Resources (Free Only)**
- [NumPy Quickstart – Official Docs](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Tutorial – Kaggle Learn](https://www.kaggle.com/learn/pandas)

**Mini Project / Hands-on Assignment**  
**Project:** Titanic Survival Data Analysis  
**Dataset:** [Kaggle: Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
**Deliverables:**
- GitHub repo with `titanic_analysis.ipynb`
- `README.md`: summary, dataset link, how to run
- Write-up: survival rates by gender/class, missing data impact
- Metrics: % missing values handled, number of visualizations

**Rubric:**
- Correctness (40%)
- README (20%)
- Evaluation (20%)
- Ethics (10%)
- Code Quality (10%)

**Ethics & Best Practices**
- [ ] Consider bias in historical data (e.g., class/gender)
- [ ] Avoid implying causation from correlation
- [ ] Cite data source
- [ ] Use consistent formatting

**Weekly Outcomes**
- Can manipulate structured data using Pandas
- Can clean real-world datasets
- Can compute basic statistics

**Stretch goals / Next steps**
- Add seaborn visualizations
- Export cleaned data to CSV

**Prerequisites:** Week 1 (Python basics)

---

### **Week 3: Essential Math for ML – Linear Algebra & Calculus**

**Topics to Cover**
- Vectors, matrices, dot product
- Matrix multiplication, transpose, inverse
- Derivatives, partial derivatives, gradient
- Role of gradients in ML

**Estimated time (hours):** 13

**Tasks**
- [4 hrs] Study vectors, matrices, operations (addition, multiplication)
- [4 hrs] Learn derivatives and partial derivatives
- [2 hrs] Visualize gradients using Python
- [3 hrs] Mini-project: Implement gradient of a 2D function

**Resources (Free Only)**
- [Essence of Linear Algebra – 3Blue1Brown (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Calculus for ML – MIT OCW Notes](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/)

**Mini Project / Hands-on Assignment**  
**Project:** Gradient Calculator for f(x,y) = x² + 2y²  
**Dataset:** None (synthetic function)  
**Deliverables:**
- GitHub repo with `gradient_calc.py`
- Notebook showing contour plots and gradient vectors
- `README.md`: math background, how to run
- Write-up: explain why gradient matters in ML
- Test: compare numerical vs analytical gradient

**Rubric:**
- Correctness (40%)
- Visualization/README (20%)
- Explanation (20%)
- Ethics (10%)
- Code/Test (10%)

**Ethics & Best Practices**
- [ ] Clearly distinguish mathematical truth from real-world inference
- [ ] Avoid misrepresenting gradient descent as always convergent
- [ ] Credit visualizations and sources

**Weekly Outcomes**
- Can compute gradients manually and numerically
- Understand matrix operations used in neural nets
- Can visualize 2D functions and gradients

**Stretch goals / Next steps**
- Implement finite difference approximation
- Plot optimization path

**Prerequisites:** Week 2 (Pandas/NumPy)

---

### **Week 4: Probability, Statistics & EDA**

**Topics to Cover**
- Mean, median, variance, standard deviation
- Distributions: normal, binomial
- Correlation, covariance
- Hypothesis testing (intuition)
- Exploratory Data Analysis (EDA)

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn descriptive statistics and distributions
- [3 hrs] Practice EDA on a dataset (e.g., Iris or Wine Quality)
- [3 hrs] Plot histograms, boxplots, correlation matrices
- [4 hrs] Mini-project: Full EDA report

**Resources (Free Only)**
- [Statistics for Data Science – StatQuest (YouTube)](https://www.youtube.com/c/joshstarmer)
- [Exploratory Data Analysis – Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.00-introduction-to-machine-learning.html)

**Mini Project / Hands-on Assignment**  
**Project:** Wine Quality EDA  
**Dataset:** [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)  
**Deliverables:**
- GitHub repo with `eda_wine.ipynb`
- `README.md`: summary, dataset, how to run
- Write-up: key insights, potential biases (e.g., subjective ratings)
- Visualizations: at least 5 plots
- Tests: assert no missing values after cleaning

**Rubric:**
- Insightfulness (40%)
- Visualization/README (20%)
- Data Cleaning (20%)
- Ethics (10%)
- Code Quality (10%)

**Ethics & Best Practices**
- [ ] Acknowledge subjectivity in "quality" labels
- [ ] Avoid stereotyping based on region or type
- [ ] Report limitations of sample size

**Weekly Outcomes**
- Can perform full EDA
- Can interpret distributions and correlations
- Understand basics of statistical inference

**Stretch goals / Next steps**
- Perform t-test between red/white wines
- Use Seaborn for styled plots

**Prerequisites:** Weeks 1–3

---

### ✅ **Checkpoint Capstone: Weeks 1–4**

**Capstone Project:** End-to-End Data Analysis Report  
**Dataset:** [Kaggle: Melanoma Classification Metadata](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) (use metadata CSV only)  
**Tasks:**
- Load and clean data
- Compute summary stats
- Visualize age distribution, sex, benign vs malignant
- Write ethical considerations (e.g., skin tone bias in diagnosis)
- Publish report as notebook + GitHub README

**Self-Assessment Checklist:**
- [ ] Can write Python functions
- [ ] Can use Pandas for data cleaning
- [ ] Can compute mean, std, correlation
- [ ] Can plot with Matplotlib/Seaborn
- [ ] Can explain gradient and matrix multiplication

**Interview Prep:**
- "Explain the difference between variance and standard deviation."
- "How would you handle missing data?"
- "What is a normal distribution?"

**Portfolio Tip:** Pin this repo on GitHub. Add to LinkedIn: “Completed foundational data science project.”

---

### **Week 5: Introduction to Machine Learning & Scikit-Learn**

**Topics to Cover**
- What is ML? Supervised vs unsupervised
- Regression vs classification
- Train/test split
- scikit-learn API (`fit`, `predict`)
- Linear regression

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn ML types and workflow
- [3 hrs] Study scikit-learn: `LinearRegression`, `train_test_split`
- [3 hrs] Train model on Boston housing (or California housing)
- [4 hrs] Mini-project: Predict house prices

**Why scikit-learn?** Uniform API, excellent docs, industry standard for classical ML.  
**Tutorial:** [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

**Resources (Free Only)**
- [Google’s ML Crash Course – Intro to ML](https://developers.google.com/machine-learning/crash-course/ml-intro)
- [Scikit-learn Tutorial – Kaggle](https://www.kaggle.com/learn/intro-to-machine-learning)

**Mini Project / Hands-on Assignment**  
**Project:** House Price Prediction  
**Dataset:** [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)  
**Deliverables:**
- GitHub repo with `house_price_model.ipynb`
- `README.md`: problem, metrics (RMSE), how to run
- Write-up: feature importance, ethical issues (e.g., redlining risk)
- Metrics: RMSE, R²
- Test: model trains without error

**Rubric:**
- Correctness (40%)
- README (20%)
- Evaluation (20%)
- Ethics (10%)
- Code Quality (10%)

**Ethics & Best Practices**
- [ ] Audit for proxy variables (e.g., location → race)
- [ ] Avoid deploying without fairness checks
- [ ] Use `random_state` for reproducibility

**Weekly Outcomes**
- Can train a linear regression model
- Can split data and evaluate performance
- Understands scikit-learn pipeline

**Stretch goals / Next steps**
- Try polynomial features
- Compare with baseline (mean predictor)

**Prerequisites:** Weeks 1–4

---

### **Week 6: Classification & Model Evaluation**

**Topics to Cover**
- Logistic regression
- Confusion matrix, accuracy, precision, recall, F1
- ROC curve, AUC
- Multiclass classification
- Overfitting/underfitting intuition

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn logistic regression and sigmoid
- [3 hrs] Compute confusion matrix and metrics
- [3 hrs] Train on Iris or Breast Cancer dataset
- [4 hrs] Mini-project: Medical diagnosis classifier

**Resources (Free Only)**
- [Logistic Regression – StatQuest (YouTube)](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [Model Evaluation – Scikit-learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html)

**Mini Project / Hands-on Assignment**  
**Project:** Breast Cancer Diagnosis Classifier  
**Dataset:** [UCI Breast Cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
**Deliverables:**
- GitHub repo with `cancer_classifier.ipynb`
- `README.md`: metrics (F1, AUC), how to run
- Write-up: clinical implications, false negative risks
- Metrics: F1, AUC, confusion matrix
- Test: `assert 0 <= f1 <= 1`

**Rubric:**
- Correctness (40%)
- Metrics (20%)
- Ethics Discussion (20%)
- README (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Prioritize recall (avoid false negatives)
- [ ] Do not claim diagnostic capability
- [ ] Note dataset limitations (e.g., demographics)

**Weekly Outcomes**
- Can train a classifier
- Can compute precision, recall, F1
- Can interpret ROC curves

**Stretch goals / Next steps**
- Try SVM or Random Forest
- Plot ROC curve

**Prerequisites:** Week 5

---

### **Week 7: Tree-Based Models & Hyperparameter Tuning**

**Topics to Cover**
- Decision trees, Random Forest, Gradient Boosting
- Feature importance
- Grid search, cross-validation
- Bias-variance tradeoff

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn Random Forest and XGBoost intuition
- [3 hrs] Use `GridSearchCV` for hyperparameter tuning
- [3 hrs] Apply to Titanic survival prediction
- [4 hrs] Mini-project: Optimized classifier

**Why XGBoost?** High performance, widely used in competitions.  
**Tutorial:** [XGBoost Python Guide](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)

**Resources (Free Only)**
- [Random Forest – StatQuest (YouTube)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [Hyperparameter Tuning – Scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html)

**Mini Project / Hands-on Assignment**  
**Project:** Optimized Titanic Predictor  
**Dataset:** [Kaggle: Titanic](https://www.kaggle.com/c/titanic/data)  
**Deliverables:**
- GitHub repo with `titanic_tuned.ipynb`
- `README.md`: best params, CV score
- Write-up: feature importance, overfitting risks
- Metrics: CV accuracy, F1
- Test: grid search completes

**Rubric:**
- CV Setup (40%)
- Tuning (20%)
- Interpretation (20%)
- Ethics (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Avoid deterministic fate narratives
- [ ] Acknowledge historical bias in survival
- [ ] Use cross-validation to avoid overfitting

**Weekly Outcomes**
- Can train ensemble models
- Can tune hyperparameters with CV
- Can interpret feature importance

**Stretch goals / Next steps**
- Try `RandomizedSearchCV`
- Use SHAP for explanation

**Prerequisites:** Weeks 5–6

---

### **Week 8: Unsupervised Learning & Clustering**

**Topics to Cover**
- K-Means clustering
- Elbow method, silhouette score
- Principal Component Analysis (PCA)
- Dimensionality reduction

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn K-Means and PCA
- [3 hrs] Apply to customer segmentation (e.g., Mall Customers)
- [3 hrs] Visualize clusters and explained variance
- [4 hrs] Mini-project: Customer Segmentation

**Resources (Free Only)**
- [K-Means – StatQuest (YouTube)](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [PCA – Scikit-learn Guide](https://scikit-learn.org/stable/modules/decomposition.html#pca)

**Mini Project / Hands-on Assignment**  
**Project:** Mall Customer Segmentation  
**Dataset:** [Kaggle: Mall Customers](https://www.kaggle.com/vjchub/mall-customers)  
**Deliverables:**
- GitHub repo with `clustering.ipynb`
- `README.md`: number of clusters, PCA components
- Write-up: business use cases, privacy concerns
- Metrics: silhouette score, inertia
- Test: `assert n_clusters > 1`

**Rubric:**
- Clustering Quality (40%)
- PCA Use (20%)
- Interpretation (20%)
- Ethics (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Avoid labeling clusters with stereotypes
- [ ] Anonymize synthetic customer data
- [ ] Explain limitations of clustering

**Weekly Outcomes**
- Can perform K-Means and PCA
- Can evaluate clustering quality
- Can reduce dimensionality

**Stretch goals / Next steps**
- Try DBSCAN
- Cluster on original vs PCA-transformed data

**Prerequisites:** Weeks 5–7

---

### ✅ **Checkpoint Capstone: Weeks 5–8**

**Capstone Project:** Predictive Modeling Competition Entry  
**Dataset:** [Kaggle: Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)  
**Tasks:**
- Clean data, engineer features
- Train Random Forest/XGBoost
- Tune with CV
- Submit to Kaggle
- Write model card: performance, bias, limitations

**Self-Assessment Checklist:**
- [ ] Can train regression/classification models
- [ ] Can evaluate with F1, AUC, RMSE
- [ ] Can tune hyperparameters
- [ ] Can cluster and reduce dimensions
- [ ] Can write model evaluation report

**Interview Prep:**
- "Explain bias-variance tradeoff."
- "When to use Random Forest vs Logistic Regression?"
- "How does PCA work?"

**Portfolio Tip:** Add Kaggle profile link to resume. Highlight top 50% score if achieved.

---

### **Week 9: Introduction to Deep Learning & PyTorch**

**Topics to Cover**
- Neurons, activation functions (ReLU, sigmoid)
- Feedforward networks
- Loss functions (MSE, Cross-Entropy)
- Gradient descent, backpropagation
- PyTorch tensors and autograd

**Estimated time (hours):** 13

**Why PyTorch?** Flexible, dynamic computation graph, dominant in research and production.  
**Tutorial:** [PyTorch Beginner Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)

**Tasks**
- [3 hrs] Learn neural network basics
- [3 hrs] Study PyTorch: tensors, `nn.Module`, `autograd`
- [3 hrs] Build a simple network for regression
- [4 hrs] Mini-project: Train on synthetic data

**Resources (Free Only)**
- [Neural Networks – 3Blue1Brown (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [PyTorch Tutorial – Official](https://pytorch.org/tutorials/)

**Mini Project / Hands-on Assignment**  
**Project:** Neural Network for f(x) = x³  
**Dataset:** Generate synthetic data: `x` in [-2,2], `y = x**3`  
**Deliverables:**
- GitHub repo with `nn_cubic.py`
- `README.md`: architecture, loss curve
- Write-up: compare with linear model
- Metrics: MSE, training loss plot
- Test: model predicts within 10% error

**Rubric:**
- Correctness (40%)
- Training Curve (20%)
- Comparison (20%)
- Ethics (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Avoid claiming human-like intelligence
- [ ] Document synthetic nature of data
- [ ] Use `torch.manual_seed()` for reproducibility

**Weekly Outcomes**
- Can build a feedforward network
- Can train with PyTorch
- Understands backpropagation

**Stretch goals / Next steps**
- Add momentum or Adam optimizer
- Visualize learned function

**Prerequisites:** Weeks 1–8

---

### **Week 10: Convolutional Neural Networks (CNNs)**

**Topics to Cover**
- Image representation (channels, pixels)
- Convolution, pooling, ReLU
- CNN architecture (LeNet, AlexNet)
- Transfer learning (ResNet)
- Data augmentation

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn CNN layers and flow
- [3 hrs] Use `torchvision` to load CIFAR-10
- [3 hrs] Train a small CNN
- [4 hrs] Mini-project: Image classifier

**Resources (Free Only)**
- [CNNs – 3Blue1Brown (YouTube)](https://www.youtube.com/watch?v=YRhxdVk_sIs)
- [TorchVision Models – Official](https://pytorch.org/vision/stable/models.html)

**Mini Project / Hands-on Assignment**  
**Project:** CIFAR-10 Image Classifier  
**Dataset:** [CIFAR-10 via TorchVision](https://www.cs.toronto.edu/~kriz/cifar.html)  
**Deliverables:**
- GitHub repo with `cnn_cifar.py`
- `README.md`: accuracy, architecture
- Write-up: bias in image datasets (e.g., object recognition disparities)
- Metrics: Test accuracy (>50%)
- Test: model runs inference on single image

**Rubric:**
- Accuracy (40%)
- Architecture (20%)
- Ethics (20%)
- README (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Discuss dataset diversity (e.g., animals vs people)
- [ ] Avoid facial recognition without consent
- [ ] Use data augmentation to improve robustness

**Weekly Outcomes**
- Can train a CNN on images
- Can use transfer learning
- Understands convolution operation

**Stretch goals / Next steps**
- Fine-tune ResNet18
- Use TensorBoard for logging

**Prerequisites:** Week 9

---

### **Week 11: Natural Language Processing (NLP) with RNNs**

**Topics to Cover**
- Text preprocessing (tokenization, padding)
- Word embeddings (Word2Vec, GloVe)
- RNNs, LSTMs
- Sentiment analysis
- Hugging Face `datasets`

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn tokenization and embeddings
- [3 hrs] Use LSTM for sentiment classification
- [3 hrs] Load IMDB dataset via Hugging Face
- [4 hrs] Mini-project: Movie review classifier

**Resources (Free Only)**
- [NLP with RNNs – TensorFlow Tutorial](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
- [Hugging Face Datasets – Quickstart](https://huggingface.co/docs/datasets/quickstart)

**Mini Project / Hands-on Assignment**  
**Project:** IMDB Sentiment Classifier  
**Dataset:** [Hugging Face: IMDB](https://huggingface.co/datasets/imdb)  
**Deliverables:**
- GitHub repo with `lstm_sentiment.py`
- `README.md`: accuracy, confusion matrix
- Write-up: risks of sentiment analysis (e.g., sarcasm, cultural bias)
- Metrics: Test accuracy (>80%)
- Test: predict on custom sentence

**Rubric:**
- Accuracy (40%)
- Preprocessing (20%)
- Ethics (20%)
- README (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Avoid deploying for surveillance
- [ ] Acknowledge sarcasm/idiom limitations
- [ ] Use balanced datasets

**Weekly Outcomes**
- Can preprocess text data
- Can train LSTM for classification
- Can use Hugging Face datasets

**Stretch goals / Next steps**
- Try BERT (Week 12)
- Use attention mechanism

**Prerequisites:** Weeks 9–10

---

### **Week 12: Transformers & Modern NLP**

**Topics to Cover**
- Attention mechanism
- Transformer architecture
- BERT, DistilBERT
- Fine-tuning with `transformers`
- Model cards

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn attention and Transformers
- [3 hrs] Use `transformers` library to load DistilBERT
- [3 hrs] Fine-tune on SST-2 (sentiment)
- [4 hrs] Mini-project: BERT sentiment classifier

**Resources (Free Only)**
- [Transformers Explained – Jay Alammar (Blog)](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Course – Free](https://huggingface.co/course/)

**Mini Project / Hands-on Assignment**  
**Project:** BERT for Sentiment Analysis  
**Dataset:** [Hugging Face: SST-2](https://huggingface.co/datasets/glue/viewer/sst2/train)  
**Deliverables:**
- GitHub repo with `bert_sentiment.py`
- `README.md`: accuracy, model name
- **Model Card:** training data, limitations, ethical risks
- Metrics: Accuracy on test set
- Test: inference on new sentence

**Rubric:**
- Performance (40%)
- Model Card (20%)
- Ethics (20%)
- README (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Include model card with bias assessment
- [ ] Avoid fine-tuning on toxic data
- [ ] Cite model license (e.g., MIT, Apache)

**Weekly Outcomes**
- Can fine-tune BERT
- Can create a model card
- Understands attention

**Stretch goals / Next steps**
- Deploy model with FastAPI
- Try zero-shot classification

**Prerequisites:** Week 11

---

### ✅ **Checkpoint Capstone: Weeks 9–12**

**Capstone Project:** Multimodal AI App  
**Options:**
- Image classifier with FastAPI endpoint
- Sentiment analyzer with web UI (Gradio)
**Deliverables:**
- GitHub repo with model, tests, Dockerfile
- `README.md` and model card
- Deploy on Render or Fly.io
- Write-up: scalability, ethical risks

**Self-Assessment Checklist:**
- [ ] Can train CNN and RNN
- [ ] Can fine-tune BERT
- [ ] Can document model responsibly
- [ ] Can preprocess images/text

**Interview Prep:**
- "Explain self-attention."
- "What is transfer learning?"
- "How does BERT work?"

**Portfolio Tip:** Add live demo link (e.g., `your-app.onrender.com`). Record a 1-min Loom video demo.

---

### **Week 13: MLOps I – Versioning, Experiment Tracking**

**Topics to Cover**
- Git for code versioning
- DVC for data/model versioning
- MLflow for experiment tracking
- Logging parameters, metrics, artifacts

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn Git basics (`commit`, `push`, branches)
- [3 hrs] Use DVC to version a dataset
- [3 hrs] Log experiments with MLflow
- [4 hrs] Mini-project: Track model experiments

**Resources (Free Only)**
- [DVC Tutorial – Official](https://dvc.org/doc/start)
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)

**Mini Project / Hands-on Assignment**  
**Project:** MLflow Experiment Tracker for Wine Quality  
**Dataset:** UCI Wine Quality  
**Deliverables:**
- GitHub repo with DVC-tracked data
- MLflow UI screenshot with runs
- `README.md`: how to reproduce
- Write-up: benefits of tracking
- Test: `mlflow ui` runs locally

**Rubric:**
- DVC Setup (30%)
- MLflow Logging (30%)
- Reproducibility (20%)
- Ethics (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Track data sources and licenses
- [ ] Avoid logging PII
- [ ] Use `.gitignore` for secrets

**Weekly Outcomes**
- Can version data with DVC
- Can log experiments with MLflow
- Can reproduce prior runs

**Stretch goals / Next steps**
- Use MLflow Models to register
- Add metrics to CI

**Prerequisites:** Weeks 1–12

---

### **Week 14: MLOps II – Docker & Deployment**

**Topics to Cover**
- Docker basics (images, containers)
- Write Dockerfile
- FastAPI for model serving
- Deploy to Render or Fly.io

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Learn Docker: `Dockerfile`, `docker build/run`
- [3 hrs] Wrap model in FastAPI endpoint
- [3 hrs] Build image and test locally
- [4 hrs] Mini-project: Deploy sentiment model

**Resources (Free Only)**
- [Docker Get Started](https://docs.docker.com/get-started/)
- [FastAPI Tutorial – Official](https://fastapi.tiangolo.com/tutorial/)

**Mini Project / Hands-on Assignment**  
**Project:** Deploy BERT Sentiment API  
**Deliverables:**
- GitHub repo with `Dockerfile`, `main.py` (FastAPI)
- Live endpoint (e.g., `your-api.onrender.com`)
- `README.md`: API docs, how to build
- Write-up: latency, scalability
- Test: `curl` request returns JSON

**Rubric:**
- Docker (30%)
- API (30%)
- Deployment (20%)
- README (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Rate-limit API to prevent abuse
- [ ] Avoid logging user inputs
- [ ] Use HTTPS in production

**Weekly Outcomes**
- Can containerize a model
- Can serve via REST API
- Can deploy to cloud

**Stretch goals / Next steps**
- Add health check endpoint
- Use environment variables

**Prerequisites:** Week 13

---

### **Week 15: MLOps III – Monitoring, Logging & CI/CD**

**Topics to Cover**
- Structured logging (Python `logging`)
- Model monitoring (prediction drift)
- GitHub Actions for CI
- Run tests on push

**Estimated time (hours):** 13

**Tasks**
- [3 hrs] Add logging to API
- [3 hrs] Simulate data drift
- [3 hrs] Write GitHub Action to run tests
- [4 hrs] Mini-project: CI-pipeline with tests

**Resources (Free Only)**
- [GitHub Actions – Official](https://docs.github.com/en/actions)
- [Python Logging – Real Python](https://realpython.com/python-logging/)

**Mini Project / Hands-on Assignment**  
**Project:** CI Pipeline for ML Model  
**Deliverables:**
- GitHub repo with `.github/workflows/test.yml`
- Logs showing drift detection
- `README.md`: CI/CD diagram
- Write-up: failure scenarios
- Test: `pytest` passes on push

**Rubric:**
- CI Setup (40%)
- Logging (20%)
- Drift Detection (20%)
- README (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Monitor for bias drift
- [ ] Alert on performance drop
- [ ] Audit logs for misuse

**Weekly Outcomes**
- Can set up CI/CD
- Can log model behavior
- Can detect drift

**Stretch goals / Next steps**
- Add model retraining trigger
- Use Prometheus/Grafana (optional)

**Prerequisites:** Weeks 13–14

---

### **Week 16: End-to-End MLOps Pipeline**

**Topics to Cover**
- Full pipeline: data → model → deploy → monitor
- Autoscaling basics
- Cost-aware deployment
- Portfolio finalization

**Estimated time (hours):** 13

**Tasks**
- [4 hrs] Build full pipeline (use prior projects)
- [3 hrs] Document all projects
- [3 hrs] Optimize Docker image
- [3 hrs] Finalize portfolio

**Mini Project / Hands-on Assignment**  
**Project:** End-to-End ML System  
**Example:** News classifier: scrape → clean → train → deploy → monitor  
**Deliverables:**
- GitHub repo with full pipeline
- Diagram (Mermaid or draw.io)
- Model card, README, tests
- Live demo link
- Write-up: lessons learned

**Rubric:**
- Completeness (40%)
- Documentation (20%)
- Deployment (20%)
- Ethics (10%)
- Code (10%)

**Ethics & Best Practices**
- [ ] Include data provenance
- [ ] Plan for model retirement
- [ ] Make system auditable

**Weekly Outcomes**
- Can build full ML pipeline
- Can deploy and monitor
- Ready for job applications

**Stretch goals / Next steps**
- Add A/B testing
- Write technical blog post

**Prerequisites:** Weeks 13–15

---

### ✅ **Final Capstone: Portfolio & Job Readiness**

**Portfolio Projects (8–10):**
1. Expense Tracker (Week 1)
2. Wine Quality EDA (Week 4)
3. House Price Model (Week 5)
4. Breast Cancer Classifier (Week 6)
5. Mall Clustering (Week 8)
6. CIFAR-10 CNN (Week 10)
7. BERT Sentiment (Week 12)
8. FastAPI Deployment (Week 14)
9. CI/CD Pipeline (Week 15)
10. End-to-End MLOps (Week 16)

**Resume Tips:**
- Use action verbs: "Built", "Deployed", "Trained"
- Quantify: "Improved accuracy by 15%"
- Link to GitHub and live demos

**Interview Readiness Checklist (20–30 Tasks):**
- [ ] Write logistic regression from scratch
- [ ] Explain backpropagation
- [ ] Dockerize a FastAPI ML service
- [ ] Design A/B test for model rollout
- [ ] Handle missing data in Pandas
- [ ] Compute F1 score manually
- [ ] Prevent overfitting
- [ ] Explain bias-variance tradeoff
- [ ] Describe ResNet architecture
- [ ] Fine-tune BERT
- [ ] Use MLflow to track experiments
- [ ] Write a model card
- [ ] Set up GitHub Actions for testing
- [ ] Monitor prediction drift
- [ ] Deploy model on Render
- [ ] Load data with DVC
- [ ] Use PCA for dimensionality reduction
- [ ] Interpret confusion matrix
- [ ] Handle class imbalance
- [ ] Explain attention mechanism
- [ ] Secure API with rate limiting
- [ ] Write clean, modular Python
- [ ] Use virtual environments
- [ ] Explain cross-validation
- [ ] Discuss AI ethics in healthcare

**Community Learning:**
- Join: r/MachineLearning, Kaggle Discussions, Hugging Face Discord
- Get feedback: post repos, ask for code review
- Contribute: answer beginner questions

---

## 🧩 Machine-Readable Summary (JSON)

```json
[
  {
    "week": 1,
    "title": "Introduction to Python Programming",
    "top_topics": ["Python basics", "Functions", "File I/O"],
    "project": "Build a CLI expense tracker with user input handling"
  },
  {
    "week": 2,
    "title": "Data Structures & Libraries in Python",
    "top_topics": ["Pandas", "NumPy", "Data cleaning"],
    "project": "Analyze Titanic survival data with Pandas and visualization"
  },
  {
    "week": 3,
    "title": "Essential Math for ML – Linear Algebra & Calculus",
    "top_topics": ["Vectors", "Matrices", "Gradients"],
    "project": "Compute and visualize gradient of a 2D function"
  },
  {
    "week": 4,
    "title": "Probability, Statistics & EDA",
    "top_topics": ["Distributions", "Correlation", "EDA"],
    "project": "Perform full EDA on wine quality dataset"
  },
  {
    "week": 5,
    "title": "Introduction to Machine Learning & Scikit-Learn",
    "top_topics": ["Linear regression", "Train/test split", "Scikit-learn"],
    "project": "Predict house prices using California housing data"
  },
  {
    "week": 6,
    "title": "Classification & Model Evaluation",
    "top_topics": ["Logistic regression", "Confusion matrix", "ROC"],
    "project": "Classify breast cancer with precision and recall analysis"
  },
  {
    "week": 7,
    "title": "Tree-Based Models & Hyperparameter Tuning",
    "top_topics": ["Random Forest", "Grid search", "Cross-validation"],
    "project": "Optimize Titanic survival predictor with CV"
  },
  {
    "week": 8,
    "title": "Unsupervised Learning & Clustering",
    "top_topics": ["K-Means", "PCA", "Silhouette score"],
    "project": "Segment mall customers using clustering"
  },
  {
    "week": 9,
    "title": "Introduction to Deep Learning & PyTorch",
    "top_topics": ["Neural networks", "Backpropagation", "PyTorch"],
    "project": "Train a neural network on synthetic cubic data"
  },
  {
    "week": 10,
    "title": "Convolutional Neural Networks (CNNs)",
    "top_topics": ["Convolution", "Pooling", "Transfer learning"],
    "project": "Classify CIFAR-10 images using a CNN"
  },
  {
    "week": 11,
    "title": "Natural Language Processing (NLP) with RNNs",
    "top_topics": ["LSTM", "Tokenization", "Sentiment analysis"],
    "project": "Build an LSTM for IMDB movie review sentiment"
  },
  {
    "week": 12,
    "title": "Transformers & Modern NLP",
    "top_topics": ["Transformers", "BERT", "Model cards"],
    "project": "Fine-tune BERT for sentiment analysis with model card"
  },
  {
    "week": 13,
    "title": "MLOps I – Versioning, Experiment Tracking",
    "top_topics": ["DVC", "MLflow", "Git"],
    "project": "Track wine quality experiments with MLflow and DVC"
  },
  {
    "week": 14,
    "title": "MLOps II – Docker & Deployment",
    "top_topics": ["Docker", "FastAPI", "Deployment"],
    "project": "Deploy BERT sentiment model via FastAPI on Render"
  },
  {
    "week": 15,
    "title": "MLOps III – Monitoring, Logging & CI/CD",
    "top_topics": ["Logging", "Drift detection", "GitHub Actions"],
    "project": "Set up CI pipeline with tests and logging"
  },
  {
    "week": 16,
    "title": "End-to-End MLOps Pipeline",
    "top_topics": ["Full pipeline", "Autoscaling", "Portfolio"],
    "project": "Build and document end-to-end ML system"
  }
]
```

--- 

✅ **You're now ready to begin.**  
Clone this plan, start Week 1, and commit daily.  
**In 16 weeks, you’ll be job-ready.**