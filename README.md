# Titanic Survival Classifier (TensorFlow.js, no server)

A lightweight, fully client-side web app that trains a shallow binary classifier on the **Kaggle Titanic** dataset using **TensorFlow.js** and **tfjs-vis**. Runs entirely in the browser (works on GitHub Pages). Upload `train.csv` and `test.csv`, train, evaluate with ROC/AUC + threshold slider, predict, and export `submission.csv`.

> **Files:**
>
> * `index.html` – HTML structure, minimal CSS, and UI controls
> * `app.js` – data loading/parsing, preprocessing, model, training, evaluation, prediction, export

---

## ✳ Rewritten Prompt / App Specification

**Goal:** Build a single-page, client-only web app that trains a binary classifier for the Titanic dataset and deploy it to GitHub Pages.

**Requirements:**

1. **Layout (in `index.html`)**
   Sections: **Data Load**, **Inspection**, **Preprocessing**, **Model**, **Training**, **Evaluation Metrics** (ROC/AUC + threshold slider), **Prediction**, **Export**, **Deployment Notes**.
   Include two file inputs for `train.csv` and `test.csv`. Use responsive, basic CSS. Load app logic from `app.js`.

2. **Tech Stack (CDNs)**

   * TensorFlow.js: `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest`
   * tfjs-vis: `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest`
     No bundlers or servers; must work on GitHub Pages out of the box.

3. **Data Schema (Titanic)**

   * **Target:** `Survived` (0/1)
   * **Features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
   * **Identifier (exclude from training):** `PassengerId`
     *Reuse note: you can swap schema later by changing the constants in `app.js`.*

4. **Load & Inspect**

   * Load CSVs from file inputs (training & test).
   * Show preview (first 10 rows), dataset shape, missing percentages.
   * Visualize survival by **Sex** and **Pclass** via tfjs-vis bar charts.

5. **Preprocessing**

   * Impute: `Age` (median), `Embarked` (mode).
   * Standardize: `Age`, `Fare` (z-score using training stats).
   * One-hot encode: `Sex`, `Pclass`, `Embarked`.
   * Optional engineered features (toggle): `FamilySize = SibSp + Parch + 1`, `IsAlone = (FamilySize == 1)`.
   * Print tensor shapes.

6. **Model**

   * `tf.sequential()` with: Dense(16, `relu`) → Dense(1, `sigmoid`)
   * Compile: optimizer `adam`, loss `binaryCrossentropy`, metric `accuracy`.
   * Render a human-readable summary (layers + param count).

7. **Training**

   * 80/20 split (on rows).
   * 50 epochs, batch size 32.
   * Live tfjs-vis charts for `loss/acc/val_loss/val_acc`.
   * Early stopping behavior not required by tfjs API directly, but the UI supports manual stop (close tab) and you can watch validation loss.

8. **Metrics (Evaluation)**

   * Compute ROC and AUC on **validation** predictions.
   * Interactive threshold slider (0–1) updates **Confusion Matrix**, **Precision**, **Recall**, **F1**, **Accuracy** dynamically.
   * Plot ROC curve with tfjs-vis.

9. **Inference & Export**

   * Predict probabilities for `test.csv`.
   * Export:

     * `submission.csv` → (`PassengerId`, `Survived` using current threshold=0.5)
     * `probabilities.csv` → (`PassengerId`, predicted probability)
   * Save model: `downloads://titanic-tfjs-model`.

10. **Deployment (GitHub Pages)**

    * Public repo → commit `index.html` & `app.js` → enable Pages (main branch, root) → open your Pages URL.

---

## 🚀 Quick Start

1. **Clone / download** this repo (or just copy `index.html` and `app.js` into a new repo).
2. Open `index.html` locally (double-click) **or** push to GitHub and enable **Pages**.
3. In the app:

   * Upload `train.csv` and `test.csv` (Kaggle Titanic).
   * Click **Load Data** → **Inspect Data** (optional) → **Preprocess Data** → **Create Model** → **Train Model**.
   * After training, open the visor (tfjs-vis floating button) to see charts.
   * Use the threshold slider in **Evaluation Metrics** to explore trade-offs.
   * Click **Predict on Test Data** → **Export Results** to download CSVs and the trained model.

> Tip: If you see nothing after training, check the bottom-right corner of the page for the tfjs-vis button to open the charts tab.

---

## 📦 Data Details

* **Train columns**: `PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`
* **Test columns**: same as train **minus** `Survived`.
* The app internally **ignores** `Name`, `Ticket`, and `Cabin` by default (they are not listed in the feature schema).

---

## 🧪 Evaluation & Metrics

* **Confusion Matrix** updates as you drag the **threshold slider** (defaults to 0.5).
* **Metrics**: Accuracy, Precision, Recall, F1, and **AUC** (computed via trapezoidal approximation on ROC).
* **ROC Curve** is shown in the **Evaluation** tab of tfjs-vis.

---

## 📤 Exports

* `submission.csv` — two columns: `PassengerId,Survived` (0/1 using threshold 0.5).
* `probabilities.csv` — `PassengerId,Probability` with 6-decimal probabilities.
* `titanic-tfjs-model` — saved to your browser downloads via `model.save('downloads://…')`.

---

## 🔧 What Was Fixed vs. Previous App

1. **CSV Comma/Escape Handling (Critical)**

   * Replaced naive `line.split(',')` with a **state-machine CSV parser** that supports:

     * Commas within quoted fields (e.g., `"Kelly, Mr. James"` in `Name`)
     * Escaped quotes as `""`
     * BOM stripping and CRLF/Unix newlines
   * Result: Clean, aligned rows; numeric casting preserved; no column shifts when names contain commas.

2. **Evaluation Table Not Showing Up**

   * **Flattened predictions** from `[[p], [p], …]` to `[p, …]` before thresholding → fixes confusion matrix & metric computations.
   * **tfjs-vis callbacks** were previously overridden; now merged into a **single callbacks object**, ensuring both status updates **and** visor charts render.
   * **ROC line chart** now uses the correct tfjs-vis shape: `{ values: [seriesPoints], series: ['ROC'] }`.

---

## 🔁 Reusing This App for Other Datasets

* In `app.js`, edit these constants to map to your dataset:

  ```js
  const TARGET_FEATURE = 'Survived';
  const ID_FEATURE = 'PassengerId';
  const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
  const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];
  ```
* Update imputation rules and one-hot categories if needed.
* The UI and training flow will work without further changes.

---

## ❓ Troubleshooting

* **Charts not visible?** Click the tfjs-vis floating button (bottom-right).
* **CSV load errors?** Ensure UTF-8 CSVs from Kaggle; the parser handles quotes/commas automatically.
* **NaN metrics?** This can occur if a split has only one class. Re-train or toggle engineered features.
* **Mobile performance:** Training is CPU-bound in the browser; consider fewer epochs on phones.

---
