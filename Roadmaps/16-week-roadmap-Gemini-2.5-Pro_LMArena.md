# Gemini-2.5-pro

Of course. Here is a detailed, step-by-step 16-week roadmap to master AI, ML, DL, and MLOps from scratch, designed to make you job-ready.

---

# 16-Week AI/ML/DL/MLOps Mastery Roadmap

This roadmap is designed for an absolute beginner and requires 12-14 hours of dedicated study per week. By the end, you will have a portfolio of 8-10 projects and the skills to build, deploy, and manage machine learning systems in production.

## Condensed Roadmap Overview

### **Block 1: Weeks 1-4 — Foundations of Programming, Math & Core ML**
*   **Goal:** Build the fundamental toolkit. Go from zero to programming in Python, understanding key mathematical concepts, performing data analysis, and training your first predictive model.
*   **Job Relevance:** These are the non-negotiable prerequisite skills for any data-related role (Analyst, Scientist, Engineer). You'll be able to answer basic screening questions about Python, data manipulation, and linear regression.
*   **Capstone Project:** Predict house prices using Linear Regression, focusing on a full EDA-to-model pipeline.

### **Block 2: Weeks 5-8 — Mastering Classical Machine Learning & Intro to Deep Learning**
*   **Goal:** Learn to build robust and powerful classical ML models. You will master model evaluation, feature engineering, tree-based models, and unsupervised learning, and build your first neural network from scratch.
*   **Job Relevance:** By the end of this block, you can tackle a wide range of tabular data problems, a core task for Data Scientists and junior ML Engineers. You'll be able to discuss the trade-offs between different models like Random Forest and Gradient Boosting.
*   **Capstone Project:** Classify customer churn using a tuned Gradient Boosting model, comparing it against other algorithms.

### **Block 3: Weeks 9-12 — Specialization in Deep Learning (Vision & NLP)**
*   **Goal:** Dive deep into neural networks. You'll use modern frameworks (PyTorch/TensorFlow) to build Convolutional Neural Networks (CNNs) for image classification and Recurrent Neural Networks (RNNs) and Transformers for text analysis.
*   **Job Relevance:** This block equips you with the skills for AI Engineer and specialized ML roles. You'll be able to build image recognition systems and basic NLP models, which are highly sought-after skills.
*   **Capstone Project:** Build an image classifier for a multi-class problem (e.g., Fashion-MNIST) using transfer learning and build a sentiment analysis model using Transformers.

### **Block 4: Weeks 13-16 — MLOps: Productionizing & Managing ML Systems**
*   **Goal:** Learn to take a model from a research notebook to a production-ready, automated system. You'll master experiment tracking, data versioning, containerization with Docker, CI/CD with GitHub Actions, and deployment & monitoring.
*   **Job Relevance:** This is the critical final step to becoming a top-tier ML Engineer, AI Engineer, or AI Architect. These skills are in high demand and are often the deciding factor in hiring decisions for senior roles.
*   **Capstone Project:** Deploy your best model as a scalable API, fully automated with CI/CD, versioning, and basic monitoring.

---

## Full Weekly Breakdown

### **BLOCK 1: FOUNDATIONS (WEEKS 1-4)**

### **Week 1: Python Fundamentals & Environment Setup**
*   **Prerequisites:** None. This is the starting point.
*   **Topics to Cover:**
    *   Python basics: variables, data types (strings, integers, floats, booleans).
    *   Python data structures: lists, tuples, dictionaries, sets.
    *   Control flow: if/else statements, for/while loops.
    *   Functions and modules.
    *   Setting up a development environment: Anaconda, VS Code, Jupyter Notebooks.
    *   Introduction to Git and GitHub for version control.
*   **Estimated time (13 hours):**
    *   **Tasks (10 hours):**
        *   Install Anaconda and VS Code. (1 hr)
        *   Complete a Python basics course (e.g., freeCodeCamp). (6 hrs)
        *   Set up a GitHub account and learn basic Git commands (`init`, `add`, `commit`, `push`). (3 hrs)
    *   **Project (3 hours):**
        *   Build the mini-project.
*   **Resources (Free Only):**
    *   **Python:** [freeCodeCamp - Python for Everybody (Full Course)](https://www.youtube.com/watch?v=8DvywoWv6fI)
    *   **Environment:** [VS Code for Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial), [Jupyter Notebook for Beginners](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
    *   **Git/GitHub:** [Git & GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk)
*   **My resources:**
    *   *Hands-On Machine Learning* (Book): Read Chapter 1 (The ML Landscape) for motivation.
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Simple Number Guessing Game.
    *   **Description:** Create a Python script where the computer thinks of a random number and the user has to guess it. Provide feedback ('too high', 'too low').
    *   **Deliverables:**
        1.  **GitHub Repo:** A public repository named `python-guessing-game`.
        2.  **Code:** A well-commented `game.py` file.
        3.  **README.md:** Explains what the project is, its features, and how to run it (`python game.py`).
    *   **Rubric:** Correctness (40%), Code Quality/Comments (30%), README (30%).
*   **Ethics & Best Practices:**
    *   **Reproducibility:** Your README should allow anyone to run your code.
    *   **Code Comments:** Write comments to explain *why* you did something, not just *what* you did.
*   **Weekly Outcomes:**
    *   Write basic Python scripts using functions, loops, and conditionals.
    *   Use Git to version control a simple project on GitHub.
    *   Set up and use a local Python development environment.
*   **Stretch goals / Next steps:**
    *   Add difficulty levels to your game (e.g., change the number range).
    *   Learn about Python virtual environments (`venv` or `conda create`).

---

### **Week 2: Essential Math & Data Libraries**
*   **Prerequisites:** Week 1 (Basic Python).
*   **Topics to Cover:**
    *   **Linear Algebra basics:** Vectors, matrices, dot products, matrix multiplication.
    *   **Calculus basics:** Derivatives (gradients), chain rule concept.
    *   **Probability & Statistics basics:** Mean, median, variance, standard deviation, probability distributions (Normal).
    *   **NumPy:** Arrays, vectorization, broadcasting.
    *   **Pandas:** DataFrames, Series, reading data (CSVs), data selection/indexing (`.loc`, `.iloc`).
*   **Estimated time (14 hours):**
    *   **Tasks (10 hours):**
        *   Watch introductory videos on Linear Algebra, Calculus, and Probability. (4 hrs)
        *   Complete a NumPy tutorial and practice array operations. (3 hrs)
        *   Complete a Pandas tutorial: loading data, selecting rows/columns, basic descriptive stats. (3 hrs)
    *   **Project (4 hours):**
        *   Work through the dataset and build the project notebook.
*   **Resources (Free Only):**
    *   **Math:** [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), [StatQuest: Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUIcdlgu78M36-6zBsK42Fv-)
    *   **NumPy:** [Official NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html), [freeCodeCamp - NumPy Crash Course](https://www.youtube.com/watch?v=9JUAPpyGkF4)
    *   **Pandas:** [Official Pandas "Getting Started" tutorials](https://pandas.pydata.org/docs/getting_started/index.html), [Pandas Tutorial by Corey Schafer](https://www.youtube.com/playlist?list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS)
*   **My resources:**
    *   Linear algebra playlist — [https://www.youtube.com/playlist?list=PLRDl2inPrWQW1QSWhBU0ki-jq_uElkh2a](https://www.youtube.com/playlist?list=PLRDl2inPrWQW1QSWhBU0ki-jq_uElkh2a)
    *   Probability playlist — [https://www.youtube.com/playlist?list=PLRDl2inPrWQWwJ1mh4tCUxlLfZ76C1zge](https://www.youtube.com/playlist?list=PLRDl2inPrWQWwJ1mh4tCUxlLfZ76C1zge)
    *   Calculus playlist — [https://www.youtube.com/playlist?list=PLRDl2inPrWQVu2OvnTvtkRpJ-wz-URMJx](https://www.youtube.com/playlist?list=PLRDl2inPrWQVu2OvnTvtkRpJ-wz-URMJx)
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Titanic Dataset Analysis.
    *   **Description:** Use Pandas and NumPy to load and perform a preliminary analysis of the famous Titanic dataset. Answer basic questions like "What was the survival rate by gender?" and "What was the average age of passengers?".
    *   **Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
    *   **Deliverables:**
        1.  **GitHub Repo:** A new repo for this project.
        2.  **Jupyter Notebook:** A notebook (`analysis.ipynb`) showing your code, outputs, and comments.
        3.  **README.md:** Link to the dataset, summary of findings, and instructions to run the notebook.
    *   **Rubric:** Correctness (40%), Code Quality (20%), Insights/Analysis (30%), README (10%).
*   **Ethics & Best Practices:**
    *   **Data Provenance:** Always cite your data source. Who collected it? When? For what purpose?
    *   **Initial Bias Check:** Look at the dataset's summary statistics. Are there any obvious imbalances (e.g., more passengers of one class)? Note these down.
*   **Weekly Outcomes:**
    *   Load and inspect a CSV dataset using Pandas.
    *   Perform basic data manipulation and aggregation (slicing, grouping).
    *   Use NumPy for numerical operations.
    *   Explain the role of vectors, matrices, and derivatives in ML at a high level.
*   **Stretch goals / Next steps:**
    *   Calculate correlation matrices in Pandas (`.corr()`).
    *   Handle missing values in the 'Age' column (e.g., fill with the mean or median).

---

### **Week 3: Data Visualization & Exploratory Data Analysis (EDA)**
*   **Prerequisites:** Week 1-2 (Python, Pandas).
*   **Topics to Cover:**
    *   The importance of EDA.
    *   **Matplotlib:** Creating basic plots (line, bar, scatter).
    *   **Seaborn:** Statistical plots (histograms, box plots, heatmaps).
    *   Principles of effective data visualization.
    *   Combining Pandas with visualization libraries for insights.
    *   Formulating and testing hypotheses with data.
*   **Estimated time (13 hours):**
    *   **Tasks (9 hours):**
        *   Complete a Matplotlib tutorial. (3 hrs)
        *   Complete a Seaborn tutorial. (3 hrs)
        *   Read articles/blogs on the EDA process. (3 hrs)
    *   **Project (4 hours):**
        *   Apply EDA techniques to a new dataset.
*   **Resources (Free Only):**
    *   **Matplotlib:** [Official Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html), [Matplotlib Tutorial by Corey Schafer](https://www.youtube.com/playlist?list=PL-osiE80TeTvipOqomVEeZ1HRrcEvt-ZG)
    *   **Seaborn:** [Official Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html), [Seaborn Crash Course](https://www.youtube.com/watch?v=6GUZXDef2U0)
    *   **EDA:** [A Comprehensive Guide to EDA](https://www.kdnuggets.com/2021/04/comprehensive-guide-exploratory-data-analysis.html)
*   **My resources:**
    *   *Hands-On Machine Learning* (Book): Read Chapter 2 for an example end-to-end project, focusing on the EDA parts.
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** EDA on the Iris Dataset.
    *   **Description:** Perform a full exploratory data analysis on the classic Iris flower dataset. Visualize relationships between features (sepal length, petal width, etc.) and the target species.
    *   **Dataset:** [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) (or load directly from Scikit-learn/Seaborn).
    *   **Deliverables:**
        1.  **GitHub Repo:** A public repository for your EDA projects.
        2.  **Jupyter Notebook:** A detailed notebook with sections for each analysis step (e.g., Data Loading, Univariate Analysis, Bivariate Analysis), including plots and written interpretations.
        3.  **README.md:** Describe the EDA process, summarize key findings/visualizations, and state any hypotheses you formed.
    *   **Rubric:** Quality of Visualizations (30%), Depth of Analysis/Interpretation (40%), Notebook Structure/Clarity (20%), README (10%).
*   **Ethics & Best Practices:**
    *   **Misleading Visualizations:** Be aware of how axis scales, colors, and chart types can mislead. Always label your axes clearly.
    *   **Correlation vs. Causation:** Remind yourself that a strong correlation seen in a plot does not imply one variable causes the other.
*   **Weekly Outcomes:**
    *   Create a variety of plots (histograms, scatter plots, box plots, heatmaps) to understand a dataset.
    *   Interpret visualizations to identify patterns, outliers, and relationships in data.
    *   Structure and document an EDA process in a Jupyter Notebook.
*   **Stretch goals / Next steps:**
    *   Try creating interactive plots using a library like [Plotly Express](https://plotly.com/python/plotly-express/).

---

### **Week 4: First ML Model & Foundations Capstone**
*   **Prerequisites:** Weeks 1-3 (Python, Pandas, EDA).
*   **Topics to Cover:**
    *   Introduction to Machine Learning: Supervised vs. Unsupervised Learning.
    *   The ML workflow: problem framing, data prep, model training, evaluation.
    *   **Scikit-learn:** A tour of the API (`fit`, `predict`, `transform`).
    *   **Linear Regression:** Theory, intuition, and implementation.
    *   **Model Evaluation:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared.
    *   Train-test split for validating model performance.
*   **Estimated time (14 hours):**
    *   **Tasks (6 hours):**
        *   Learn the theory of Linear Regression. (2 hrs)
        *   Go through the Scikit-learn documentation and tutorials on the `LinearRegression` model. (4 hrs)
    *   **Project (8 hours):**
        *   Complete the capstone project.
*   **Resources (Free Only):**
    *   **Theory:** [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
    *   **Scikit-learn:** [Official Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html), [Intro to Scikit-learn Video](https://www.youtube.com/watch?v=0B5eIE_1vpU)
    *   **Full Workflow:** [Kaggle - Intro to Machine Learning course](https://www.kaggle.com/learn/intro-to-machine-learning) (especially Lessons 1-5).
*   **My resources:**
    *   Statistical Learning with Python by Stanford learning — [https://www.youtube.com/playlist?list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ](https://www.youtube.com/playlist?list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ) (Watch Lectures on Linear Regression).
    *   100 Days of ML playlist — [https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) (Find videos on Linear Regression).
*   **Capstone Project 1: Boston Housing Price Prediction**
    *   **Description:** Build an end-to-end machine learning project. Start with the Boston Housing dataset, perform EDA, preprocess the data, train a Linear Regression model, evaluate its performance, and interpret the results.
    *   **Dataset:** [Kaggle - Boston Housing Dataset](https://www.kaggle.com/datasets/fedesoriano/boston-housing-dataset)
    *   **Deliverables:**
        1.  **GitHub Repo:** A dedicated repository for this capstone.
        2.  **Jupyter Notebook:** A clean, well-documented notebook covering:
            *   Problem Definition & EDA.
            *   Data Preprocessing (handling missing values if any, feature selection).
            *   Model Training (using `train_test_split`).
            *   Model Evaluation (reporting MSE, R², etc.).
            *   Conclusion & Interpretation.
        3.  **README.md:** A full project summary, including the problem, dataset link, setup instructions (`requirements.txt`), a summary of results, and ethical considerations.
        4.  **Ethical Considerations Section:** Discuss potential issues. Who could be harmed by an inaccurate prediction? Could this model be used to perpetuate housing inequality?
    *   **Rubric:** EDA (20%), Preprocessing & Modeling (30%), Evaluation & Interpretation (20%), Reproducibility/README (20%), Ethics (10%).
*   **BLOCK 1 SELF-ASSESSMENT & INTERVIEW PREP**
    *   **Self-Assessment Checklist:**
        *   \[ ] Can I write a Python function that takes a list and returns a dictionary?
        *   \[ ] Can I load a CSV into a Pandas DataFrame and select the first 100 rows?
        *   \[ ] Can I create a histogram and a scatter plot from a DataFrame?
        *   \[ ] Can I explain the purpose of a train-test split?
        *   \[ ] Can I train a `LinearRegression` model in scikit-learn and get its R² score?
    *   **Mock Interview Questions:**
        *   "Explain the difference between a list and a tuple in Python."
        *   "What is a Pandas DataFrame?"
        *   "How would you investigate a new dataset for the first time?"
        *   "What is the objective of Linear Regression? What does it try to minimize?"
        *   "Why do we split our data into training and testing sets?"

---

### **BLOCK 2: MASTERING CLASSICAL ML & INTRO TO DEEP LEARNING (WEEKS 5-8)**

*   **Block 2 Overview:** You now have the foundations. This block builds on them to cover the most widely used classical ML algorithms. By week 8, you'll be able to train and tune a variety of sophisticated models and even build a small neural network, which are core tasks for junior ML engineers and data scientists.

### **Week 5: Model Evaluation, Cross-Validation & Feature Engineering**
*   **Prerequisites:** Week 4 (completed first ML model).
*   **Topics to Cover:**
    *   **Classification Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, AUC.
    *   **Overfitting vs. Underfitting:** The bias-variance tradeoff.
    *   **Cross-Validation:** K-Fold CV for robust model evaluation.
    *   **Feature Engineering:**
        *   Handling categorical variables (One-Hot Encoding).
        *   Feature scaling (StandardScaler, MinMaxScaler).
        *   Creating new features from existing ones.
    *   **Scikit-learn Pipelines:** Combining preprocessing and modeling steps.
*   **Estimated time (14 hours):**
    *   **Tasks (9 hours):**
        *   Study classification metrics and the bias-variance tradeoff. (3 hrs)
        *   Learn about cross-validation and practice implementing it. (3 hrs)
        *   Learn and apply one-hot encoding and feature scaling. (3 hrs)
    *   **Project (5 hours):**
        *   Implement these techniques on a classification problem.
*   **Resources (Free Only):**
    *   **Metrics:** [StatQuest: Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o), [StatQuest: ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM)
    *   **Bias-Variance:** [Scott Fortmann-Roe - Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)
    *   **Cross-Validation:** [Scikit-learn Docs on Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
    *   **Feature Engineering:** [Kaggle - Feature Engineering Course](https://www.kaggle.com/learn/feature-engineering)
    *   **Pipelines:** [Scikit-learn Docs on Pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline)
*   **My resources:**
    *   *Hands-On Machine Learning* (Book): Read Chapter 2 again, focusing on data cleaning, feature scaling, and pipelines.
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Logistic Regression for Titanic Survival Prediction.
    *   **Description:** Revisit the Titanic dataset. This time, build a complete preprocessing pipeline (handling missing age, one-hot encoding 'Sex' and 'Embarked' columns, scaling features) and train a Logistic Regression classifier. Evaluate it using cross-validation and multiple metrics (Accuracy, Precision, Recall, AUC).
    *   **Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
    *   **Deliverables:**
        1.  **GitHub Repo:** New or updated repo.
        2.  **Jupyter Notebook:** A clean notebook demonstrating the pipeline, training, and a detailed evaluation section with a confusion matrix and discussion of metrics.
        3.  **README.md:** Update with the new model, its performance, and a discussion of why certain metrics (e.g., Recall) might be more important than others in this context.
    *   **Rubric:** Preprocessing Pipeline (40%), Model Evaluation (30%), Interpretation of Metrics (20%), Reproducibility (10%).
*   **Ethics & Best Practices:**
    *   **Metric Choice:** Is accuracy the right metric? If a model predicts everyone perishes, it's 62% accurate. Discuss why precision/recall might be better.
    *   **Feature Bias:** The 'Sex' feature is highly predictive. Acknowledge this and consider the ethical implications of using demographic data in models.
*   **Weekly Outcomes:**
    *   Evaluate classification models using a full suite of metrics.
    *   Explain and identify overfitting and underfitting.
    *   Implement K-fold cross-validation for robust model assessment.
    *   Build a scikit-learn pipeline for preprocessing and modeling.
*   **Stretch goals / Next steps:**
    *   Implement a `GridSearchCV` to find the best hyperparameters for `LogisticRegression`.

---

### **Week 6: Tree-Based Models & Ensembles**
*   **Prerequisites:** Week 5 (robust evaluation, pipelines).
*   **Topics to Cover:**
    *   **Decision Trees:** Intuition, splitting criteria (Gini, Entropy), pros and cons.
    *   **Ensemble Learning:** Bagging and Boosting.
    *   **Random Forests:** How they work and why they are effective.
    *   **Gradient Boosting Machines (GBM):** XGBoost, LightGBM. High-level intuition.
    *   **Hyperparameter Tuning:** Randomized Search, Grid Search.
*   **Estimated time (13 hours):**
    *   **Tasks (8 hours):**
        *   Learn Decision Tree theory. (2 hrs)
        *   Study Random Forests and Bagging. (2 hrs)
        *   Get an intuitive understanding of Gradient Boosting (XGBoost). (2 hrs)
        *   Practice using `GridSearchCV` and `RandomizedSearchCV`. (2 hrs)
    *   **Project (5 hours):**
        *   Apply tree-based models to a new problem.
*   **Resources (Free Only):**
    *   **Decision Trees:** [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
    *   **Random Forests:** [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4W_rPggl_c)
    *   **XGBoost:** [StatQuest: Gradient Boost (XGBoost)](https://www.youtube.com/watch?v=3CC4N4z3GJc)
    *   **Hyperparameter Tuning:** [Scikit-learn Docs on Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
*   **My resources:**
    *   100 Days of ML playlist — [https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) (Find videos on Decision Trees, Random Forest, XGBoost).
    *   *Hands-On Machine Learning* (Book): Read Chapters 6 and 7.
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Predicting Heart Disease with a Random Forest.
    *   **Description:** Use a dataset on heart disease to train a Random Forest classifier. Perform hyperparameter tuning using `RandomizedSearchCV` to find the best model. Compare its performance to a baseline Logistic Regression model.
    *   **Dataset:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
    *   **Deliverables:**
        1.  **GitHub Repo.**
        2.  **Jupyter Notebook:** Must include sections for baseline model, Random Forest with tuning, and a final comparison table of model performance.
        3.  **README.md:** Explain the project, the models used, and the results of the tuning process.
        4.  **Ethical Considerations:** This is a medical dataset. Discuss the high stakes of false negatives (predicting no disease when there is one) vs. false positives.
    *   **Rubric:** Model Implementation (30%), Hyperparameter Tuning (30%), Model Comparison/Analysis (20%), README/Ethics (20%).
*   **Ethics & Best Practices:**
    *   **Model Interpretability:** Decision trees are easy to visualize and explain. Random Forests are less so. Discuss this tradeoff. Tools like SHAP or LIME can help (a stretch goal).
    *   **Data Privacy:** This data is anonymized, but always be mindful of privacy when working with medical or personal information.
*   **Weekly Outcomes:**
    *   Explain the difference between bagging and boosting.
    *   Train and tune a Random Forest and a Gradient Boosting model.
    *   Use `GridSearchCV` or `RandomizedSearchCV` to optimize model hyperparameters.
*   **Stretch goals / Next steps:**
    *   Install XGBoost or LightGBM and see if you can get better performance.
    *   Try to visualize one of the decision trees from your trained Random Forest.

---

### **Week 7: Unsupervised Learning**
*   **Prerequisites:** Week 6 (familiarity with Scikit-learn API).
*   **Topics to Cover:**
    *   **Clustering:** Finding groups in data.
        *   **K-Means:** Algorithm and intuition, choosing K (Elbow Method).
    *   **Dimensionality Reduction:** Compressing data.
        *   **Principal Component Analysis (PCA):** Theory and application for visualization and feature reduction.
*   **Estimated time (12 hours):**
    *   **Tasks (7 hours):**
        *   Learn K-Means theory and practice implementing it. (3 hrs)
        *   Study the intuition behind PCA. (2 hrs)
        *   Practice using PCA for dimensionality reduction and visualization. (2 hrs)
    *   **Project (5 hours):**
        *   Apply clustering and PCA to a real dataset.
*   **Resources (Free Only):**
    *   **K-Means:** [StatQuest: K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
    *   **PCA:** [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ)
    *   **Scikit-learn Docs:** [Clustering](https://scikit-learn.org/stable/modules/clustering.html), [Decomposition (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca)
*   **My resources:**
    *   *Hands-On Machine Learning* (Book): Read Chapter 9 (Unsupervised Learning).
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Customer Segmentation with K-Means.
    *   **Description:** Use a mall customer dataset to segment customers into distinct groups based on their spending habits. Use the Elbow Method to determine the optimal number of clusters. Then, use PCA to reduce the features to 2 dimensions and visualize the clusters.
    *   **Dataset:** [Kaggle - Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
    *   **Deliverables:**
        1.  **GitHub Repo.**
        2.  **Jupyter Notebook:** Should include:
            *   EDA and feature scaling.
            *   Elbow Method plot to choose K.
            *   K-Means model training.
            *   PCA for visualization.
            *   A scatter plot of the 2 principal components, colored by cluster.
            *   A short analysis of each customer segment (e.g., "Cluster 0: high income, low spending").
        3.  **README.md:** Summarize the project, findings, and business implications of the customer segments.
    *   **Rubric:** K-Means Implementation (30%), PCA & Visualization (30%), Cluster Analysis/Interpretation (30%), README (10%).
*   **Ethics & Best Practices:**
    *   **Labeling Clusters:** Be careful with the labels you assign to clusters. Avoid stereotypical or pejorative names. Describe them based on data (e.g., "high-spending, frequent visitors").
    *   **Actionability:** How could these segments be used? Ethically (e.g., for targeted discounts) or unethically (e.g., for predatory pricing)? Discuss this.
*   **Weekly Outcomes:**
    *   Apply K-Means clustering to segment data.
    *   Use the Elbow Method to estimate the optimal number of clusters.
    *   Use PCA to reduce the dimensionality of data for visualization.
    *   Interpret the results of clustering and PCA.
*   **Stretch goals / Next steps:**
    *   Try another clustering algorithm like DBSCAN and compare the results.

---

### **Week 8: Intro to Deep Learning & Classical ML Capstone**
*   **Prerequisites:** Week 2 (Linear Algebra, Calculus concepts), Week 5 (Classification).
*   **Topics to Cover:**
    *   From Linear Regression to Neurons: The Perceptron.
    *   **Neural Networks:** Layers, weights, biases, activation functions (Sigmoid, ReLU).
    *   **How NNs Learn:** Forward Propagation, Backpropagation, and Gradient Descent (conceptual).
    *   Building a simple Neural Network from scratch with NumPy for a classification task.
*   **Estimated time (14 hours):**
    *   **Tasks (6 hours):**
        *   Watch videos explaining the components of a neural network. (3 hrs)
        *   Follow a tutorial to build a simple NN with NumPy to solidify understanding. (3 hrs)
    *   **Project (8 hours):**
        *   Complete the capstone project using classical ML models.
*   **Resources (Free Only):**
    *   **Neural Networks:** [3Blue1Brown - But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk), [Welch Labs - Neural Networks Demystified](https://www.youtube.com/playlist?list=PLiaHhY2iP9gY6eZ9w27U0V2gdEkYrc6A2)
    *   **NN from Scratch:** [Samson Zhang - Neural Network from Scratch in Python](https://www.youtube.com/watch?v=w8yWXqWQYmU)
*   **My resources:**
    *   Neural Networks / Deep Learning by StatQuest with Josh Starmer — [https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
    *   jay alammar blog — [https://jalammar.github.io/](https://jalammar.github.io/) (Read "A Visual and Interactive Guide to the Basics of Neural Networks").
    *   Neural Networks: Zero to Hero by Andrej Karpathy — [https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (Watch the first video for a great conceptual overview).
*   **Capstone Project 2: Customer Churn Prediction**
    *   **Description:** You are a data scientist at a telecom company tasked with predicting customer churn. Use the skills from the past 4 weeks. Perform EDA, build a robust preprocessing pipeline, and compare at least three different models (e.g., Logistic Regression, Random Forest, XGBoost). Select the best model based on appropriate metrics and hyperparameter tuning.
    *   **Dataset:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    *   **Deliverables:**
        1.  **GitHub Repo:** A polished, portfolio-ready repository.
        2.  **Jupyter Notebook:** A very clean, well-structured notebook that tells a story:
            *   Business Problem & EDA.
            *   Preprocessing Pipeline.
            *   Model Training & Tuning for multiple models.
            *   Final Model Selection & Evaluation (with confusion matrix, ROC curve, and metric comparison table).
            *   Feature Importance analysis (for tree-based models).
            *   Conclusion and Recommendations for the business.
        3.  **README.md:** A comprehensive project description, setup guide, summary of results, and link to a short write-up (e.g., a blog post or just a section in the README) explaining your approach and findings to a non-technical audience.
        4.  **Ethical Considerations:** Discuss how this model could be used. To offer discounts to at-risk customers (proactive retention)? Or to deprioritize support for them (cost-cutting)?
    *   **Rubric:** EDA & Preprocessing (20%), Modeling & Tuning (30%), Evaluation & Comparison (20%), Communication/README (20%), Ethics (10%).
*   **BLOCK 2 SELF-ASSESSMENT & INTERVIEW PREP**
    *   **Self-Assessment Checklist:**
        *   \[ ] Can I explain the bias-variance tradeoff?
        *   \[ ] Can I implement cross-validation and explain why it's better than a single train-test split?
        *   \[ ] Can I build a scikit-learn `Pipeline` with a scaler and an estimator?
        *   \[ ] Can I explain the difference between a Random Forest and a Gradient Boosting model at a high level?
        *   \[ ] Can I describe what a neuron does in a neural network (inputs, weights, activation)?
    *   **Mock Interview Questions:**
        *   "What are precision and recall? When would you prefer one over the other?"
        *   "Describe how a Random Forest model works to a non-technical person."
        *   "How would you handle categorical features in your data?"
        *   "What is overfitting, and how can you prevent it?"
        *   "Walk me through a typical machine learning project workflow."

---

### **BLOCK 3: DEEP LEARNING SPECIALIZATION (WEEKS 9-12)**

*   **Block 3 Overview:** With a strong foundation in classical ML, you're ready to tackle Deep Learning. This block focuses on practical implementation using modern frameworks like PyTorch and TensorFlow. You'll build models for computer vision and natural language processing, skills that unlock AI Engineer roles. For compute, use [Google Colab](https://colab.research.google.com/) or [Kaggle Kernels](https://www.kaggle.com/kernels) to get free GPU access.

### **Week 9: Deep Learning Frameworks & Convolutional Neural Networks (CNNs)**
*   **Prerequisites:** Week 8 (NN concepts).
*   **Topics to Cover:**
    *   **PyTorch (or TensorFlow/Keras):** Tensors, automatic differentiation, building sequential models, defining training loops. *We'll use PyTorch as an example, as it's very popular in research and increasingly in industry.*
    *   **CNNs:** The convolution operation, pooling layers, and architecture of a basic CNN.
    *   Applying CNNs to image classification.
    *   Understanding how to structure a DL project.
*   **Estimated time (14 hours):**
    *   **Tasks (9 hours):**
        *   Complete a "PyTorch in 60 Minutes" or equivalent tutorial. (3 hrs)
        *   Learn the theory behind convolutions and pooling. (3 hrs)
        *   Follow a tutorial to build your first CNN for a simple dataset like MNIST. (3 hrs)
    *   **Project (5 hours):**
        *   Build your own CNN for a new dataset.
*   **Resources (Free Only):**
    *   **PyTorch:** [Official PyTorch Tutorials](https://pytorch.org/tutorials/), [freeCodeCamp - PyTorch for Deep Learning](https://www.youtube.com/watch?v=G_R_4UNabEU)
    *   **TensorFlow/Keras:** [Official TensorFlow Keras Basics](https://www.tensorflow.org/tutorials/keras/classification)
    *   **CNN Theory:** [StatQuest: Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXUSlRUI), [Stanford CS231n - Lecture 5: CNNs](https://www.youtube.com/watch?v=bNb2fEVKeEo)
*   **My resources:**
    *   100 Days of DL playlist — [https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn](https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn) (Find videos on CNNs).
    *   jay alammar blog — [https://jalammar.github.io/](https://jalammar.github.io/) (Read "The Illustrated Transformer" later, but "A Visual and Interactive Guide to Building a Neural Network" is good now).
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Handwritten Digit Recognition with a CNN.
    *   **Description:** Build, train, and evaluate a simple CNN from scratch using PyTorch or TensorFlow to classify handwritten digits from the MNIST dataset.
    *   **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) (available directly in PyTorch/TensorFlow).
    *   **Deliverables:**
        1.  **GitHub Repo.**
        2.  **Jupyter/Colab Notebook:** A clean notebook showing data loading, model definition, the training loop, and evaluation (accuracy, confusion matrix).
        3.  **README.md:** Describe the CNN architecture (layers, filter sizes), training process, and final accuracy.
    *   **Rubric:** Correct Model Implementation (40%), Training Loop & Evaluation (40%), Documentation/README (20%).
*   **Ethics & Best Practices:**
    *   **Reproducibility in DL:** Set random seeds (`torch.manual_seed`) to make your training process repeatable.
    *   **Data Augmentation:** For real-world images, the original dataset is never enough. Mention that techniques like rotating, flipping, and cropping images are standard practice to prevent overfitting (a good stretch goal).
*   **Weekly Outcomes:**
    *   Build and train a simple deep learning model using PyTorch or TensorFlow.
    *   Explain the function of convolutional and pooling layers in a CNN.
    *   Implement a complete training and evaluation pipeline for an image classification task.
*   **Stretch goals / Next steps:**
    *   Implement data augmentation for your MNIST dataset.
    *   Try a slightly more complex dataset like Fashion-MNIST.

---

### **Week 10: Advanced CNNs & Transfer Learning**
*   **Prerequisites:** Week 9 (built a basic CNN).
*   **Topics to Cover:**
    *   **Modern CNN Architectures:** High-level overview of VGG, ResNet, Inception.
    *   **Transfer Learning:** The concept of using pre-trained models.
    *   **Fine-tuning:** How to adapt a pre-trained model for a new task.
    *   Techniques to improve training: data augmentation, dropout, batch normalization.
*   **Estimated time (13 hours):**
    *   **Tasks (8 hours):**
        *   Read about famous CNN architectures like ResNet. (2 hrs)
        *   Understand the concept and power of transfer learning. (2 hrs)
        *   Follow a tutorial on how to load a pre-trained model (e.g., ResNet50) and fine-tune it on a new dataset. (4 hrs)
    *   **Project (5 hours):**
        *   Apply transfer learning to a real-world image problem.
*   **Resources (Free Only):**
    *   **Transfer Learning:** [PyTorch Tutorial on Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [TensorFlow Tutorial on Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
    *   **Architectures:** [CS231n - Lecture 9: CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y), Blog post on [ResNets](https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/).
*   **My resources:**
    *   *Hands-On Machine Learning* (Book): Read Chapter 14 (Deep Computer Vision).
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Cat vs. Dog Image Classification with Transfer Learning.
    *   **Description:** Use a pre-trained model (like ResNet18 or MobileNetV2) and fine-tune it to classify images of cats and dogs. Aim for >95% accuracy.
    *   **Dataset:** [Kaggle - Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
    *   **Deliverables:**
        1.  **GitHub Repo.**
        2.  **Jupyter/Colab Notebook:** Document the process of loading the pre-trained model, freezing the early layers, adding a new classification head, and the fine-tuning training loop.
        3.  **README.md:** Explain what transfer learning is, why it's useful, which model you used, and your final test accuracy.
    *   **Rubric:** Correct Implementation of Transfer Learning (50%), Evaluation (20%), Explanation/README (30%).
*   **Ethics & Best Practices:**
    *   **Bias in Pre-trained Models:** Pre-trained models (like those trained on ImageNet) inherit biases from their training data. Discuss this. For example, they may perform worse on images from under-represented geographic regions or cultures.
    *   **Model Cards:** Introduce the concept of a [Model Card](https://modelcards.withgoogle.com/about), a document that provides context and transparency about a model's performance characteristics and limitations. Start drafting a simple one for your project.
*   **Weekly Outcomes:**
    *   Explain the concept and benefits of transfer learning.
    *   Implement fine-tuning on a pre-trained CNN using PyTorch or TensorFlow.
    *   Achieve high performance on an image classification task with relatively little data.
*   **Stretch goals / Next steps:**
    *   Experiment with fine-tuning more layers of the network vs. fewer. How does it affect performance and training time?

---

### **Week 11: Sequential Data & Recurrent Neural Networks (RNNs)**
*   **Prerequisites:** Week 9 (experience with a DL framework).
*   **Topics to Cover:**
    *   Working with text data: tokenization, embedding.
    *   **Word Embeddings:** Word2Vec, GloVe (conceptual).
    *   **RNNs:** The concept of hidden state and processing sequences.
    *   **Long Short-Term Memory (LSTM) & Gated Recurrent Unit (GRU):** Solving the vanishing gradient problem in RNNs.
    *   Building an RNN/LSTM for text classification.
*   **Estimated time (13 hours):**
    *   **Tasks (8 hours):**
        *   Learn about text preprocessing and word embeddings. (3 hrs)
        *   Understand the architecture and limitations of simple RNNs. (2 hrs)
        *   Learn how LSTMs and GRUs improve upon RNNs. (3 hrs)
    *   **Project (5 hours):**
        *   Build a text classifier.
*   **Resources (Free Only):**
    *   **RNN/LSTM Theory:** [StatQuest: Recurrent Neural Networks (RNNs) and LSTMs](https://www.youtube.com/playlist?list=PLblh5JKOoLUJjeXUvUEpo_602aSd3wG_9)
    *   **Implementation:** [PyTorch Tutorial on Text Classification](https://pytorch.org/tutorials/beginner/text_classification_tutorial.html), [TensorFlow Tutorial on Text Classification with an RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
*   **My resources:**
    *   jay alammar blog — [https://jalammar.github.io/](https://jalammar.github.io/) (Read "Visualizing a Neural Machine Translation Model" and "The Illustrated Word2vec").
    *   *Hands-On Machine Learning* (Book): Read Chapter 16 (Natural Language Processing with RNNs and Attention).
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** IMDB Movie Review Sentiment Analysis.
    *   **Description:** Build an LSTM-based model to classify movie reviews as positive or negative.
    *   **Dataset:** [IMDB Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (also available in TensorFlow Datasets/TorchText).
    *   **Deliverables:**
        1.  **GitHub Repo.**
        2.  **Jupyter/Colab Notebook:** Show the full pipeline: text preprocessing (tokenization, building a vocabulary), defining the LSTM model, training, and evaluating on test data.
        3.  **README.md:** Explain the model architecture and results. Include examples of a positive and a negative review that your model classified correctly and incorrectly.
    *   **Rubric:** Text Preprocessing (30%), Model Implementation (40%), Analysis of Results (20%), README (10%).
*   **Ethics & Best Practices:**
    *   **Toxicity & Bias in Language:** Language models trained on internet text can learn and amplify societal biases (racism, sexism). Acknowledge that your simple sentiment model could have such biases. How would you test for this?
    *   **Data Source:** The language used in movie reviews is specific. This model would likely not work well for classifying sentiment in legal documents or medical notes.
*   **Weekly Outcomes:**
    *   Preprocess text data for use in a deep learning model.
    *   Explain the function of a recurrent layer and a hidden state.
    *   Build, train, and evaluate an LSTM model for a text classification task.
*   **Stretch goals / Next steps:**
    *   Try using pre-trained word embeddings like GloVe instead of training your own. Does it improve performance?

---

### **Week 12: Transformers & Deep Learning Capstone**
*   **Prerequisites:** Week 11 (RNNs for text).
*   **Topics to Cover:**
    *   **Attention Mechanism:** The core idea behind Transformers.
    *   **Transformers:** High-level architecture (Encoder-Decoder, Self-Attention).
    *   **BERT and GPT:** Understanding pre-trained transformer models.
    *   Using the [Hugging Face](https://huggingface.co/) `transformers` library for fine-tuning a pre-trained model (e.g., DistilBERT) on a classification task.
*   **Estimated time (14 hours):**
    *   **Tasks (6 hours):**
        *   Get a high-level understanding of the Attention mechanism and Transformer architecture. (3 hrs)
        *   Complete a Hugging Face tutorial on fine-tuning a model for text classification. (3 hrs)
    *   **Project (8 hours):**
        *   Complete the two-part capstone.
*   **Resources (Free Only):**
    *   **Transformers Theory:** [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) (Essential Reading!), [StatQuest: Transformers, Clearly Explained!](https://www.youtube.com/watch?v=sznZotwJ0V8)
    *   **Hugging Face:** [Official Hugging Face Course (Chapter 1 & 2)](https://huggingface.co/course/chapter1)
*   **My resources:**
    *   jay alammar blog: Re-read The Illustrated Transformer. It is foundational.
*   **Capstone Project 3: Modern Deep Learning for Vision & NLP**
    *   **Part 1: Advanced Image Classification.**
        *   **Description:** Choose a more challenging image classification dataset and apply transfer learning with a modern architecture (e.g., EfficientNet). Implement data augmentation and other best practices to achieve the best possible performance.
        *   **Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) or [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
        *   **Deliverables:** A polished notebook and repo similar to prior projects, but with a focus on documenting the techniques used to maximize performance.
    *   **Part 2: Fine-tuning a Transformer for NLP.**
        *   **Description:** Use the Hugging Face library to fine-tune a pre-trained Transformer model (like DistilBERT) for a text classification task. Compare its performance and training time to the LSTM model from Week 11.
        *   **Dataset:** Use the same IMDB dataset from Week 11 for a direct comparison, or try a new one like [AG News Classification](https://huggingface.co/datasets/ag_news).
        *   **Deliverables:**
            1.  **GitHub Repo:** A single repo for both parts of the capstone.
            2.  **Notebooks:** Separate, clean notebooks for each part.
            3.  **README.md:** A comprehensive summary of both projects. Include a comparison table for the NLP task (LSTM vs. Transformer) showing metrics, training time, and model size.
            4.  **Model Card:** Write a more detailed model card for your Transformer model, noting its intended use, limitations, and potential biases.
    *   **Rubric:** Vision Project (30%), NLP Project (30%), Comparison & Analysis (20%), Documentation/Model Card (20%).
*   **BLOCK 3 SELF-ASSESSMENT & INTERVIEW PREP**
    *   **Self-Assessment Checklist:**
        *   \[ ] Can I build a simple CNN in PyTorch/TensorFlow?
        *   \[ ] Can I explain what transfer learning is and how to implement it?
        *   \[ ] Can I describe the difference between an RNN and a CNN and their primary use cases?
        *   \[ ] Can I load a pre-trained model from Hugging Face and fine-tune it on a new dataset?
        *   \[ ] Can I explain, at a high level, what the "attention" mechanism does?
    *   **Mock Interview Questions:**
        *   "Why is transfer learning so effective in computer vision?"
        *   "What is the vanishing gradient problem and how do LSTMs help solve it?"
        *   "Walk me through how you would approach an image classification problem."
        *   "What are the advantages of Transformer models over RNNs for NLP tasks?"
        *   "How would you handle a situation where your deep learning model is overfitting?"

---

### **BLOCK 4: MLOPS: PRODUCTIONIZING & MANAGING ML SYSTEMS (WEEKS 13-16)**

*   **Block 4 Overview:** You can build powerful models. Now it's time to learn how to deploy and manage them responsibly, the key skill that differentiates an ML Engineer. This block covers the MLOps lifecycle, from tracking experiments to deploying a model as a scalable service. We will use free tools and platforms like MLflow, DVC, Docker, GitHub Actions, and cloud free tiers.

### **Week 13: MLOps Principles: Versioning & Experiment Tracking**
*   **Prerequisites:** All prior blocks. You should have a trained model (e.g., your churn model from Week 8) ready to work with.
*   **Topics to Cover:**
    *   **The MLOps Lifecycle:** The loop of data, modeling, and deployment.
    *   **Code Versioning:** Git (review and best practices like branching).
    *   **Data & Model Versioning:** Introduction to **DVC (Data Version Control)**.
    *   **Experiment Tracking:** Introduction to **MLflow** for logging parameters, metrics, and models.
*   **Estimated time (14 hours):**
    *   **Tasks (9 hours):**
        *   Refactor a previous project (e.g., Week 8 churn model) from a notebook into a Python script (`train.py`). (3 hrs)
        *   Complete a DVC "Get Started" tutorial to version your dataset. (3 hrs)
        *   Integrate MLflow tracking into your `train.py` script to log metrics and parameters. (3 hrs)
    *   **Project (5 hours):**
        *   Apply DVC and MLflow to your project.
*   **Resources (Free Only):**
    *   **MLOps Intro:** [MLOps Guide by Chip Huyen](https://huyenchip.com/mlops-book/); [What is MLOps?](https://ml-ops.org/)
    *   **DVC:** [Official DVC Get Started](https://dvc.org/doc/start), [DVC Tutorial Video](https://www.youtube.com/watch?v=Y2-3v1C_cTs)
    *   **MLflow:** [Official MLflow Quickstart](https://mlflow.org/docs/latest/getting-started/index.html), [MLflow in 10 minutes](https://www.youtube.com/watch?v=x31Hy7s54sE)
*   **My resources:**
    *   *Designing Machine Learning Systems* (Book): Read Chapters 1 and 2 for a high-level overview of ML systems.
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Add Versioning and Tracking to the Churn Prediction Project.
    *   **Description:** Take your Week 8 capstone project.
        1.  Convert the core logic from the notebook into a `train.py` script.
        2.  Initialize DVC and use it to track your `churn.csv` dataset.
        3.  Integrate MLflow to log the hyperparameters and the final F1-score of your model each time you run `train.py`.
        4.  Run the script a few times with different model parameters and see the experiments logged in the MLflow UI.
    *   **Deliverables:**
        1.  **GitHub Repo:** Your updated churn project repo.
        2.  **Project Structure:** A clean structure with `train.py`, `requirements.txt`, a `data/` directory tracked by DVC, and a `dvc.yaml` file.
        3.  **README.md:** Update to explain how to set up the project, pull the data with DVC (`dvc pull`), and run the training script. Include a screenshot of your MLflow UI showing multiple experiment runs.
    *   **Rubric:** Script Refactoring (30%), DVC Implementation (30%), MLflow Integration (30%), README (10%).
*   **Ethics & Best Practices:**
    *   **Reproducibility:** You've now made your project *truly* reproducible. Anyone can check out your Git repo, pull the exact data version with DVC, and rerun the exact experiment. This is the foundation of responsible ML.
    *   **Model Lineage:** You can now trace a model artifact back to the exact code, data, and parameters that produced it. This is crucial for debugging and auditing.
*   **Weekly Outcomes:**
    *   Refactor a Jupyter notebook into a reusable Python script.
    *   Use DVC to version control a dataset alongside code.
    *   Use MLflow to log and compare machine learning experiments.
*   **Stretch goals / Next steps:**
    *   Use MLflow to also log your trained model as an artifact.

---

### **Week 14: Containerization & Model Serving**
*   **Prerequisites:** Week 13 (scripted model training).
*   **Topics to Cover:**
    *   **Containers:** Why use them? Introduction to **Docker**.
    *   **Dockerfile:** How to write a Dockerfile for a Python application.
    *   **Model Serving:** Turning a model into an API.
    *   **FastAPI:** A modern, high-performance web framework for Python.
    *   Creating a simple API endpoint that loads a trained model and makes predictions.
*   **Estimated time (14 hours):**
    *   **Tasks (9 hours):**
        *   Complete a Docker for beginners tutorial. (3 hrs)
        *   Learn the basics of FastAPI and build a simple "hello world" API. (3 hrs)
        *   Follow a tutorial on serving a scikit-learn model with FastAPI. (3 hrs)
    *   **Project (5 hours):**
        *   Containerize and create an API for your model.
*   **Resources (Free Only):**
    *   **Docker:** [Docker for Beginners](https://www.youtube.com/watch?v=3c-iBn73dDE), [Official Docker "Get Started"](https://docs.docker.com/get-started/)
    *   **FastAPI:** [Official FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/), [FastAPI in 60 minutes](https://www.youtube.com/watch?v=7t2alSnE2-I)
    *   **Serving ML Model:** [Deploying a Scikit-Learn Model with FastAPI](https://testdriven.io/blog/fastapi-machine-learning/)
*   **My resources:**
    *   FastAPI for Machine Learning — [https://www.youtube.com/playlist?list=PLKnIA16_RmvZ41tjbKB2ZnwchfniNsMuQ](https://www.youtube.com/playlist?list=PLKnIA16_RmvZ41tjbKB2ZnwchfniNsMuQ)
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Create a Dockerized Prediction API for the Churn Model.
    *   **Description:**
        1.  Save your best-trained churn model from last week (e.g., as a `joblib` or `pickle` file).
        2.  Create a simple FastAPI application with a `/predict` endpoint that takes customer data (as JSON) and returns a churn prediction.
        3.  Write a `Dockerfile` to package your FastAPI app, model file, and dependencies into a container image.
        4.  Build and run the Docker container locally and test the endpoint using `curl` or Python's `requests`.
    *   **Deliverables:**
        1.  **GitHub Repo:** All the new files (`main.py`, `Dockerfile`, saved model).
        2.  **Dockerfile:** A well-commented Dockerfile.
        3.  **README.md:** Update with clear instructions on how to build the Docker image (`docker build`) and run the container (`docker run`). Include an example `curl` command to test the API.
    *   **Rubric:** FastAPI App (40%), Dockerfile (40%), Documentation/Reproducibility (20%).
*   **Ethics & Best Practices:**
    *   **API Security:** Your current API is open to anyone. In a real system, you would need authentication. Acknowledge this limitation.
    *   **Input Validation:** Your API might crash if it receives bad data. Use Pydantic (built into FastAPI) to validate incoming data types and ranges.
*   **Weekly Outcomes:**
    *   Explain the benefits of containerization for ML deployment.
    *   Write a Dockerfile to package a Python application.
    *   Build a simple REST API using FastAPI to serve a machine learning model.
    *   Run and test the model serving application locally as a Docker container.
*   **Stretch goals / Next steps:**
    *   Write a simple Python script (`test_api.py`) that uses the `requests` library to send data to your running API and prints the response.

---

### **Week 15: Automation & CI/CD for ML**
*   **Prerequisites:** Week 14 (Dockerized model API).
*   **Topics to Cover:**
    *   **Continuous Integration (CI):** Automatically testing your code.
    *   **Continuous Delivery/Deployment (CD):** Automatically building and deploying your application.
    *   **GitHub Actions:** A tool for automating workflows directly in GitHub.
    *   Creating a simple CI pipeline: run linter (e.g., `flake8`) and unit tests (e.g., `pytest`) on every push.
    *   Creating a CD pipeline: automatically build and push a Docker image to a registry (e.g., Docker Hub, GitHub Container Registry) when you merge to the `main` branch.
*   **Estimated time (13 hours):**
    *   **Tasks (8 hours):**
        *   Learn the concepts of CI/CD. (2 hrs)
        *   Complete a GitHub Actions for beginners tutorial. (3 hrs)
        *   Write simple unit tests for your FastAPI app using `pytest`. (3 hrs)
    *   **Project (5 hours):**
        *   Set up a full CI/CD pipeline.
*   **Resources (Free Only):**
    *   **CI/CD Concepts:** [What is CI/CD? by Red Hat](https://www.redhat.com/en/topics/devops/what-is-ci-cd)
    *   **GitHub Actions:** [Official GitHub Actions Quickstart](https://docs.github.com/en/actions/quickstart), [GitHub Actions Tutorial by freeCodeCamp](https://www.youtube.com/watch?v=R8_veQiY-c8)
    *   **Pytest:** [Official Pytest "Get Started"](https://docs.pytest.org/en/7.1.x/getting-started.html)
*   **Mini Project / Hands-on Assignment:**
    *   **Title:** Automate the Churn Model API with GitHub Actions.
    *   **Description:**
        1.  **CI:** Create a GitHub Actions workflow (`ci.yml`) that triggers on every push. It should install dependencies, run a linter (`flake8`), and run at least one `pytest` test for your API (e.g., test that the `/` endpoint returns a 200 status code).
        2.  **CD:** Create a second workflow (`cd.yml`) that triggers only on pushes to the `main` branch. This workflow should build your Docker image and push it to [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
    *   **Deliverables:**
        1.  **GitHub Repo:** with a `.github/workflows` directory containing your `ci.yml` and `cd.yml` files.
        2.  **Tests:** A `tests/` directory with your `pytest` files.
        3.  **README.md:** Add a section explaining your CI/CD setup. Include the status badges for your workflows.
    *   **Rubric:** CI Workflow (40%), CD Workflow (40%), Tests (10%), Documentation (10%).
*   **Ethics & Best Practices:**
    *   **Automated Bias Checks:** A great CI pipeline for ML doesn't just test code, it tests the model. You could add a step that runs a data validation check or a model fairness report (e.g., using a tool like Fairlearn) and fails the build if a bias metric crosses a threshold.
    *   **Secrets Management:** Your CD pipeline needs to log in to the container registry. Use GitHub Secrets to store your credentials securely, never hardcode them in the workflow file.
*   **Weekly Outcomes:**
    *   Explain the purpose of CI and CD in an MLOps context.
    *   Write a simple unit test for a web API using `pytest`.
    *   Create a GitHub Actions workflow to automatically test code.
    *   Create a GitHub Actions workflow to automatically build and publish a Docker image.
*   **Stretch goals / Next steps:**
    *   Set up a staging environment. Modify your CD pipeline to deploy to a free service like [Fly.io](https://fly.io/) or [Render](https://render.com/) when you create a new GitHub release.

---

### **Week 16: Monitoring, Cloud Deployment & Final Capstone**
*   **Prerequisites:** Week 15 (CI/CD pipeline).
*   **Topics to Cover:**
    *   **Cloud Platforms:** Overview of AWS, GCP, Azure for ML.
    *   **Deployment:** Deploying your container to a cloud service (e.g., using a free tier like AWS App Runner, GCP Cloud Run, or a simpler alternative like Fly.io/Render).
    *   **Monitoring & Logging:** Why it's critical for ML models.
        *   **Logging:** Capturing predictions and errors from your live API.
        *   **Monitoring:** Tracking key metrics like latency, error rate, and **data drift/concept drift**.
    *   Putting it all together: The full end-to-end MLOps pipeline.
*   **Estimated time (14 hours):**
    *   **Tasks (6 hours):**
        *   Sign up for a cloud free tier and explore the dashboard. (2 hrs)
        *   Follow a tutorial to deploy a container to your chosen platform. (3 hrs)
        *   Read about monitoring for data drift and concept drift. (1 hr)
    *   **Project (8 hours):**
        *   Deploy your application and complete the final capstone summary.
*   **Resources (Free Only):**
    *   **Deployment:** [Deploying FastAPI to AWS with Docker (DigitalOcean guide)](https://www.digitalocean.com/community/tutorials/how-to-deploy-a-fastapi-app-on-aws-ec2-with-docker-and-nginx), [GCP Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/python), [Render Quickstart](https://render.com/docs/deploy-fastapi) (often the simplest to start).
    *   **Monitoring:** [What is Model Drift? by Evidently AI](https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift), [MLOps course by Andrew Ng - Week 3](https://www.coursera.org/learn/machine-learning-engineering-for-production-mlops/home/week/3) (audit for free).
*   **My resources:**
    *   *Designing Machine Learning Systems* (Book): Read Chapters 9, 10, and 11 on monitoring and operations.
*   **Final Capstone Project 4: Deploy and Document a Full MLOps Pipeline**
    *   **Description:** This is the culmination of your work.
        1.  **Deploy:** Take the container image you built in Week 15 and deploy it to a public cloud service (Render or Fly.io are great free-tier options). You should have a live, public API endpoint.
        2.  **Monitor:** Add basic logging to your FastAPI app to print incoming prediction requests and the model's output to the console (the cloud service will capture these logs).
        3.  **Portfolio Polish:** Create a new, top-level "AI/ML Portfolio" repository on your GitHub.
        4.  **Write-ups:** Inside this repo, create a high-quality summary page (e.g., `README.md` or a personal website/blog you link to). For at least 3-4 of your best projects (e.g., Churn Prediction, Image Classifier, NLP model, this MLOps pipeline), create a detailed project page. Each page should include:
            *   The business problem.
            *   A link to the dedicated project repo.
            *   A summary of your approach (EDA, model choice, etc.).
            *   Key results and visualizations.
            *   A link to the live deployed API (if applicable).
            *   A discussion of the challenges and ethical considerations.
    *   **Deliverables:**
        1.  A link to your live, running prediction API.
        2.  Your final, polished GitHub repository for the Churn model, with all MLOps components.
        3.  A link to your new portfolio repository that acts as a central hub for all your work.
    *   **Rubric:** This is about polish and communication. Successful Deployment (30%), Portfolio Quality (40%), Project Write-ups (30%).
*   **BLOCK 4 SELF-ASSESSMENT & INTERVIEW PREP**
    *   **Self-Assessment Checklist:**
        *   \[ ] Can I explain the difference between data versioning and code versioning?
        *   \[ ] Can I write a simple `Dockerfile` for a Python app?
        *   \[ ] Can I explain what a CI/CD pipeline does?
        *   \[ ] Can I describe the steps to deploy a model as a containerized API?
        *   \[ ] Can I explain the concept of "model drift" and why it's important to monitor?
    *   **Mock Interview Questions:**
        *   "How would you version control a 10GB dataset?"
        *   "Walk me through how you would deploy a scikit-learn model into production."
        *   "Your model's performance in production has dropped significantly over the last month. What are the possible causes and how would you investigate?"
        *   "What are the benefits of using Docker for deploying ML models?"
        *   "Design an MLOps system for a team of data scientists. What tools would you choose for experiment tracking, versioning, and deployment, and why?"

---

## Final Interview Readiness Checklist (30 Practical Tasks)

You've completed the roadmap! Now, consolidate your skills by practicing these common interview tasks.

**Theory & Concepts**
1.  Explain the bias-variance tradeoff with a diagram.
2.  Describe the difference between L1 and L2 regularization.
3.  Explain how a confusion matrix works.
4.  Describe how a decision tree makes a split.
5.  Explain gradient descent with an analogy.
6.  What is the purpose of an activation function in a neural network?
7.  Explain the difference between bagging and boosting.
8.  What is transfer learning and why is it powerful?
9.  Explain the core idea behind the "attention" mechanism.
10. Describe data drift vs. concept drift.

**Practical Coding (Whiteboard/Notebook)**
11. Write a Python function to compute the mean squared error.
12. Implement logistic regression from scratch using NumPy.
13. Write code to one-hot encode a categorical feature in Pandas.
14. Use scikit-learn to build a `Pipeline` that scales data and trains a model.
15. Write a simple training loop in PyTorch/TensorFlow.
16. Load a pre-trained model from Hugging Face and use it for inference.
17. Write a `pytest` unit test for a function.
18. Write a simple "hello world" FastAPI endpoint.
19. Write a basic `Dockerfile` for a Python script.
20. Use `curl` to send a JSON payload to a running API endpoint.

**System Design & MLOps**
21. Design a system to predict real-time housing prices for a website.
22. How would you set up an A/B test to compare two different recommendation models?
23. Design a CI/CD pipeline for an ML model. What are the key stages?
24. How would you monitor a deployed classification model for performance degradation?
25. Your team needs to retrain a model every night. How would you automate this?
26. You have a 1TB dataset. How do you manage and version it?
27. Describe the components of a feature store.
28. How would you serve a large language model (LLM) that requires a GPU?
29. A stakeholder wants to know why the model made a specific prediction. What tools could you use? (SHAP, LIME).
30. Walk me through one of your portfolio projects, explaining your design choices and the challenges you faced.

---

## JSON Summary

```json
[
  {
    "week": 1,
    "title": "Python Fundamentals & Environment Setup",
    "topics": ["Python Basics", "Control Flow", "Git & GitHub"],
    "project": "Build a simple number guessing game script."
  },
  {
    "week": 2,
    "title": "Essential Math & Data Libraries",
    "topics": ["Linear Algebra/Calculus/Stats", "NumPy", "Pandas"],
    "project": "Perform initial data analysis on the Titanic dataset."
  },
  {
    "week": 3,
    "title": "Data Visualization & Exploratory Data Analysis (EDA)",
    "topics": ["Matplotlib", "Seaborn", "EDA Principles"],
    "project": "Conduct a full EDA on the Iris flower dataset."
  },
  {
    "week": 4,
    "title": "First ML Model & Foundations Capstone",
    "topics": ["Scikit-learn", "Linear Regression", "Model Evaluation"],
    "project": "Predict Boston housing prices with an end-to-end linear regression model."
  },
  {
    "week": 5,
    "title": "Model Evaluation, Cross-Validation & Feature Engineering",
    "topics": ["Classification Metrics", "Cross-Validation", "Pipelines"],
    "project": "Build a robust logistic regression classifier for Titanic survival."
  },
  {
    "week": 6,
    "title": "Tree-Based Models & Ensembles",
    "topics": ["Decision Trees", "Random Forests", "Gradient Boosting"],
    "project": "Predict heart disease using a tuned Random Forest model."
  },
  {
    "week": 7,
    "title": "Unsupervised Learning",
    "topics": ["K-Means Clustering", "PCA", "Dimensionality Reduction"],
    "project": "Segment mall customers using K-Means and visualize with PCA."
  },
  {
    "week": 8,
    "title": "Intro to Deep Learning & Classical ML Capstone",
    "topics": ["Neural Networks", "Backpropagation", "Gradient Descent"],
    "project": "Predict customer churn by comparing several classical ML models."
  },
  {
    "week": 9,
    "title": "Deep Learning Frameworks & Convolutional Neural Networks (CNNs)",
    "topics": ["PyTorch/TensorFlow", "CNNs", "Image Classification"],
    "project": "Build a CNN from scratch to classify MNIST handwritten digits."
  },
  {
    "week": 10,
    "title": "Advanced CNNs & Transfer Learning",
    "topics": ["Transfer Learning", "Fine-tuning", "Modern CNN Architectures"],
    "project": "Classify cats vs. dogs using a pre-trained ResNet model."
  },
  {
    "week": 11,
    "title": "Sequential Data & Recurrent Neural Networks (RNNs)",
    "topics": ["RNNs", "LSTMs", "Text Preprocessing"],
    "project": "Perform sentiment analysis on IMDB movie reviews with an LSTM."
  },
  {
    "week": 12,
    "title": "Transformers & Deep Learning Capstone",
    "topics": ["Attention Mechanism", "Transformers", "Hugging Face"],
    "project": "Fine-tune a Transformer for NLP and build an advanced image classifier."
  },
  {
    "week": 13,
    "title": "MLOps Principles: Versioning & Experiment Tracking",
    "topics": ["MLOps Lifecycle", "DVC", "MLflow"],
    "project": "Add DVC and MLflow to the churn prediction project."
  },
  {
    "week": 14,
    "title": "Containerization & Model Serving",
    "topics": ["Docker", "FastAPI", "REST APIs"],
    "project": "Create a Dockerized FastAPI service to serve the churn model."
  },
  {
    "week": 15,
    "title": "Automation & CI/CD for ML",
    "topics": ["CI/CD", "GitHub Actions", "Pytest"],
    "project": "Build a CI/CD pipeline to automatically test and build the model API."
  },
  {
    "week": 16,
    "title": "Monitoring, Cloud Deployment & Final Capstone",
    "topics": ["Cloud Deployment", "Model Monitoring", "Data Drift"],
    "project": "Deploy the churn model API to the cloud and create a portfolio summary."
  }
]
```