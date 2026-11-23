# Trained Models and Pipelines

This directory is intended to store trained Machine Learning models and pipelines.

Each team must include **at least one trained model or pipeline** in their final GitHub repository so that it can be loaded and reused without retraining.

## 1. Saving Models and Pipelines

We recommend saving complete scikit-learn pipelines whenever possible, including preprocessing steps and the trained estimator.

Use `joblib` to save your trained pipeline:

```python
import joblib

# Save the trained pipeline
joblib.dump(pipeline, "../models/best_pipeline.joblib")
```

To load it later:

```python
import joblib

# Load the trained pipeline
pipeline = joblib.load("../models/best_pipeline.joblib")

# Make predictions on new data
predictions = pipeline.predict(X_new)
```

These examples assume that your notebook's working directory is `notebooks/`.

## 2. File Naming

Use descriptive filenames that identify the saved model or pipeline.

For example:

- `logistic_regression.joblib`
- `random_forest.joblib`
- `best_pipeline.joblib`

Avoid generic names such as `model1.joblib` or `final_v2.joblib`.

## 3. File Size and GitHub

Trained models can become large, especially ensemble models such as Random Forests.

GitHub warns about files larger than 50 MiB and blocks regular Git uploads of files larger than 100 MiB.

**Recommendations:**

- Keep submitted model files below 50 MiB.
- Avoid committing unnecessarily large models.
- Do not use Git LFS unless explicitly instructed.
- Check file sizes before committing.

You can check the size of your saved models from the terminal:

```bash
ls -lh models/
```

Run this command from the repository's root directory.

## 4. Final Project Submission

Your team's GitHub repository must contain:

1. At least one trained model or pipeline saved as a `.joblib` file.
2. The code or notebook used to train the model.
3. Instructions in the main project `README.md` explaining how to load the saved model and make predictions.
4. A description of the model, its purpose, and its evaluation results.

The submitted model should be usable without retraining.

## 5. Important Considerations

- Only load `.joblib` files from trusted sources, as loading untrusted files can execute malicious code.
- Saved models may not be compatible with different versions of scikit-learn or other dependencies.
- Use the project's `uv.lock` file to reproduce the environment whenever possible.
- Make sure that the features supplied during prediction match those expected by the saved model or pipeline.

