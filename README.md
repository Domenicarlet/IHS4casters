# Forecasting hidradenitis suppurativa activity from fragmented records

**Almirall Data – Innovation Challenge 2026 · team `IHS4casters`**

We forecast each patient's average disease activity (IHS4) for January–June 2026 from synthetic,
real-world-like records that end on 31 December 2025. We also identify biologic super-responders and
measure what image and RNAseq embeddings add to a clinical model.

---

## 1. Motivation

Hidradenitis suppurativa (HS) is a chronic inflammatory skin disease that evolves through flares, with
very different responses to biologic treatment. Real-world records describe it poorly:

- IHS4 is scored at only some visits;
- patients disappear for months;
- flares treated outside the clinic leave no score;
- sites document differently;
- molecular and imaging data exist for few patients.

Almirall's question is whether we can still recover a reliable picture of each patient, and use what
one data source sees well to fill what others miss.

Our guiding idea: **treat the gaps and biases of the records as part of the problem, not as noise.** A
missing score, an antibiotic course without a visit, or the reason a visit happened all say something
about the patient, or about how they were documented.

---

## 2. The data challenges, and how we answered each

The challenges below come from the brief; the numbers come from our exploration (`01_exploration`).

| Challenge | What we found | What we did |
|---|---|---|
| **IHS4 is scored only sometimes** | 64 % of visits have a score; 1 in 5 train patients has no score in the last year | Features that say how much evidence each patient has; a prior for patients with little data |
| **Scores are biased by why the visit happened** | Flare and follow-up visits score about 25 % above the true 6-month average; scheduled visits are unbiased | Divide reactive-visit scores by a visit-type factor estimated without the target |
| **Old scores lose value** | Carry-forward error rises from 2.3 to 3.3 as the latest score ages | Weight recent scores more; pull sparse patients toward their Hurley × biologic group average ("shrinkage") |
| **Flares that never reach a clinic** | 30 % of antibiotic courses have no visit within ±3 days, and they predict the target even when no score exists | "Hidden flare" features: out-of-clinic antibiotics, unscored flare visits, incision & drainage surgeries |
| **Missing fields** | 8–28 % per demographic field, often missing together | "Missing" kept as a category plus flags; diagnostic delay recovered from dates (missing values drop from 25 % to 8.5 %) |
| **Information from the future** | Surgery recurrence is filled in even when its 6-month window ends in 2026 | Masked whenever it was not yet knowable on 31 Dec 2025 |
| **Practitioners document differently** | Scoring rates range from under 40 % to over 90 % | Practitioner-style features, computed from each practitioner's *other* patients |
| **A shifted test population (test_2)** | More women and Black patients, younger, and **much less documented**, especially Black patients and patients with missing ethnicity | A simulated "pseudo‑test_2" to validate against, and an analysis of where the gap comes from |
| **Embeddings on small, non-random subsets** | Images: 13 % of patients, all from flare visits. RNAseq: 3 %, all taken at biologic start | Used as corrections on top of the clinical model and compared on the same patients |

**A lesson learned along the way.** Our first practitioner features averaged over all of a practitioner's
patients, including the patient being described. A classifier could then tell train from test_1 (a random
holdout) with AUC 0.65, which should be impossible. Computing the averages from the *other* patients,
across all splits, brought it back to 0.50. Checking that train and a random holdout are
indistinguishable is a cheap, powerful leak detector.

---

## 3. How we validate without test targets

The test sets have no targets, so every number below is a **5-fold cross-validation estimate on train**:
each patient is predicted by a model that never saw them. We report two versions:

- **test_1 proxy:** train as it is. test_1 is a random holdout, so this should be close to the real test_1
  score.
- **pseudo‑test_2:** train made to look like test_2. Scores and visits are removed at test_2's rates per
  ethnicity and visit type, and patients are reweighted to test_2's demographics.

---

## 4. The models

### Exercise 1: forecast (`02_exercise1_forecast`)

- **Features (107 per patient):** recent IHS4 levels (raw and visit-type corrected), recency-weighted
  means, trend, the shrinkage level, hidden-flare traces, treatment, surgeries, demographics, and
  documentation and practitioner style. No feature uses the target.
- **Model:** LightGBM (gradient-boosted decision trees).
  - It builds a few hundred small trees, each correcting the errors of the previous ones.
  - It handles missing values natively.
  - The Huber loss keeps a few extreme patients from dominating and behaves close to MAE.
  - The final model averages 3 random seeds.

### Exercise 2: biologics and super-responders (`03_exercise2_biologics`)

- **Untreated estimate.** A super-responder is defined against what their IHS4 *would have been without
  treatment*, which is never observed. A **natural-history model** is trained on untreated Hurley 2–3
  patients to predict their 2026 IHS4 from their state at a given date. Applied at each patient's first
  biologic start, it estimates their untreated IHS4.
- **Label.** Super-responder = observed 2026 IHS4 below 25 % of that estimate (the brief's definition).
- **2a forecast.** The average of the Exercise 1 model and a dedicated model trained on biologic patients
  with treatment-specific features.
- **2b classifiers.** Three classifiers with growing information:
  - before treatment;
  - early in treatment (months 1–6);
  - at the end of 2025, which produces the submitted probabilities.

### Exercise 3: embeddings (`04_exercise3_embeddings`)

- **Image correction.** The embedding first estimates the IHS4 of the photographed flare, even when nobody
  scored it. A ridge regression then corrects the Exercise 1 forecast from the gap between that estimate
  and the forecast, weighted by how recent the photo is.
- **Images + RNAseq.** The same correction plus RNAseq components, trained only on the 600 patients with
  both embeddings, as the brief requires.

---

## 5. Results

### Exercise 1: forecast (MAE, lower is better)

| Model | test_1 proxy | pseudo‑test_2 |
|---|---|---|
| Population median | 4.33 | 4.44 |
| Hurley-stage median | 2.67 | 2.81 |
| Carry forward last 3 scores | 2.57 | 2.82 |
| Shrinkage level alone (no machine learning) | 1.91 | 2.03 |
| **LightGBM** | **1.43** (95 % CI 1.41–1.45) | **1.58** (1.55–1.61) |

- **44 % below the best baseline** in the brief; 83 % of patients land in the correct severity band.
- **What drives it:** 70 % of the attribution is recent disease level and 15 % is hidden-flare traces.
  Removing all documentation features leaves the error unchanged, so the model does not depend on
  recording artefacts.

### Why test_2 is harder

- test_1 is indistinguishable from train (AUC 0.50).
- test_2 differs in demographics (AUC 0.71) and documentation (AUC 0.68).
- Within the same ethnicity, the documentation loss hits Black patients (0.73) and patients with missing
  ethnicity (0.68), not White patients (0.54).
- Expected cost: the population change alone takes MAE to 1.50, the documentation change alone to 1.48,
  and **both together to 1.58**.
- **Equity implication:** the patients who are documented less get less accurate forecasts. In real data,
  that is a care-quality signal worth reporting.

### Exercise 2: biologic patients

**2a forecast:** MAE **1.15** (test_1 proxy) and **1.29** (pseudo‑test_2), against 1.17 / 1.31 for the
Exercise 1 model on the same patients.

**2b super-responders:** 11.1 % of biologic patients (the brief states roughly 12.5 %).

| Information available | AUC | Average precision (chance = 0.11) |
|---|---|---|
| Before treatment | 0.75 | 0.23 |
| Early in treatment (months 1–6) | 0.84 | 0.44 |
| At the end of 2025 (submitted) | 0.98 | 0.88 |

**Who are the super-responders?**

- **Hurley 2 rather than 3:** about 19 % super-responders among Hurley 2 patients on their first biologic,
  against under 3 % for Hurley 3.
- **First-line treatment.**
- **Milder disease before treatment.**
- **Earlier treatment:** shorter diagnostic delay, younger age.

This fits the clinical picture: biologics work best before irreversible tunnels form.

**Caveats:**

- Drug and line cannot be separated, because adalimumab is almost always first line.
- The rate does not differ by sex or by how often the practitioner prescribes biologics, so the signal is
  not a prescribing habit.
- The 0.98 model *identifies* super-responders using the recent on-treatment record. The first two rows
  show how well they can be *predicted* in advance.

### Exercise 3: embeddings (MAE on the same patients)

| Patients | Model | Trained on | test_1 proxy | Δ vs Exercise 1 (95 % CI) |
|---|---|---|---|---|
| 2,800 with an image | Exercise 1 | 21,000 | 1.48 | — |
| | + images (3a) | 2,800 | 1.49 | +0.005 (−0.009 to +0.019) |
| 600 with image + RNAseq | Exercise 1 | 21,000 | 2.07 | — |
| | + images | 600 | 1.91 | −0.16 (−0.26 to −0.08) |
| | **+ images + RNAseq (3b)** | 600 | **1.71** | **−0.36 (−0.48 to −0.24)** |

**Images.**
- On average, the photo shows what the records already contain: it predicts the IHS4 of its own visit
  (R² 0.55) but none of the clinical model's error.
- Images help where records are thin: recent biologic starters (−0.10), short histories (−0.08) and
  super-responders (−0.16).
- More training images do not turn the overall effect into a gain.

**RNAseq.**
- RNAseq is taken on the first day of a biologic and carries biology that the records can't show yet.
- It recognises super-responders with **AUC 0.93**, against 0.80 from clinical data.
- About 200 samples are enough to capture most of the gain.

**Confounds.**
- Device and batch track collection date but predict neither the outcome nor the model's errors.
- Removing them from the embeddings changes nothing.

---

## 6. Limitations

- All scores are cross-validation estimates. The pseudo‑test_2 assumes the lost scores and visits are
  random within ethnicity and visit type.
- The untreated estimate assumes treated and untreated patients in the same pre-treatment state would have
  evolved alike. Its error (MAE ≈ 2.5) carries into the super-responder labels.
- The RNAseq results rest on 600 recent starters, and their generalisation to long-treated patients is
  untested.

## 7. Ideas that carry over to real-world data

1. Correct each score for *why* the visit happened before averaging, with scheduled visits as the
   unbiased anchor.
2. Pull sparsely documented patients toward a clinical prior instead of trusting one old score.
3. Treat antibiotic courses with no nearby visit as evidence of flares that never reached the clinic.
4. Build site statistics without the patient's own records, and check that train and a random holdout are
   indistinguishable.
5. Report documentation gaps by patient group: they turn directly into unequal model quality.
6. Collect RNAseq at biologic start to flag likely super-responders before the first follow-up.

**Next steps we did not run:**

- carry the RNAseq signal to patients without RNAseq (cross-modal imputation, or using RNAseq as a
  teacher during training);
- a model weighted toward test_2's population;
- prediction intervals;
- an explicit model of the hidden flares between visits.

---

## 8. Repository and how to run

```
ALMIRALL DATA - INNOVATION CHALLENGE 2026/
├── public/                         # challenge data (train, test_1, test_2, evaluation_templates)
├── 01_exploration.ipynb            # 8 questions about the data → modelling decisions
├── 02_exercise1_forecast.ipynb     # features, validation, LightGBM, drivers, test_2 gap → exercise 1
├── 03_exercise2_biologics.ipynb    # untreated estimate, super-responders, 2a/2b → exercise 2
├── 04_exercise3_embeddings.ipynb   # confounds, image and RNAseq corrections → exercise 3a/3b
├── environment.yml
├── work/                           # created by the notebooks (features, out-of-fold predictions, models)
└── submissions/                    # created by the notebooks (the four answer files)
```

Each notebook is organised as **question → code → figure or table → comment on the results**.

```bash
conda env create -f environment.yml
conda activate hs-hackathon
python -m ipykernel install --user --name hs-hackathon --display-name "HS hackathon"
# open the notebooks in VS Code or Jupyter, select the "HS hackathon" kernel,
# and run 01 → 02 → 03 → 04 (about 5 minutes in total)
```

The team name is set in the variable `TEAM_NAME` at the top of every notebook and is used in the answer
file names. Notebooks 03 and 04 read what 02 saves in `work/`, so run them in order.
