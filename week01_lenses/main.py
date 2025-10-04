import argparse
import os
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def smoke_test():
    """
    This function runs a smoke test.
    - Creates a dummy dataset.
    - Trains a DummyClassifier.
    - Saves evaluation results.
    - Prints a summary and a success message.
    """
    print("Running smoke test...")

    # 1. Create a dummy dataset from UCI Lenses
    data = {
        'age': ['young', 'young', 'pre-presbyopic', 'pre-presbyopic', 'presbyopic', 'presbyopic', 'young', 'young', 'pre-presbyopic', 'pre-presbyopic', 'presbyopic', 'presbyopic', 'young', 'young', 'pre-presbyopic', 'pre-presbyopic', 'presbyopic', 'presbyopic', 'young', 'young', 'pre-presbyopic', 'pre-presbyopic', 'presbyopic', 'presbyopic'],
        'spectacle_prescript': ['myope', 'myope', 'myope', 'myope', 'hypermetrope', 'hypermetrope', 'myope', 'myope', 'myope', 'myope', 'hypermetrope', 'hypermetrope', 'hypermetrope', 'hypermetrope', 'hypermetrope', 'hypermetrope', 'myope', 'myope', 'myope', 'myope', 'hypermetrope', 'hypermetrope', 'hypermetrope', 'hypermetrope'],
        'astigmatism': ['no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
        'tear_prod_rate': ['reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal', 'reduced', 'normal'],
        'contact_lenses': ['none', 'soft', 'none', 'hard', 'none', 'soft', 'none', 'soft', 'none', 'hard', 'none', 'soft', 'none', 'soft', 'none', 'hard', 'none', 'soft', 'none', 'hard', 'none', 'soft', 'none', 'none']
    }
    df = pd.DataFrame(data)
    print("Loaded UCI Lenses dataset (24 rows):")
    print(df.info())
    print("\n")

    # Simple preprocessing
    X = pd.get_dummies(df.drop('contact_lenses', axis=1))
    y = df['contact_lenses']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    # 2. Train a DummyClassifier as a baseline
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    print("Baseline DummyClassifier trained.")
    print("\n")

    # 3. Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline Accuracy: {accuracy:.2f}")
    print("\n")

    # 4. Save evaluation results to artifacts/
    artifacts_dir = "artifacts"
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    eval_df = pd.DataFrame([{'metric': 'accuracy', 'value': accuracy}])
    eval_path = os.path.join(artifacts_dir, "eval.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Evaluation results saved to {eval_path}")
    print("\n")

    # 5. Print success message
    print("SMOKE OK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke test for Week 1 project.")
    parser.add_argument("--smoke", action="store_true", help="Run the smoke test.")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        print("No action specified. Use --smoke to run the smoke test.")