# Naive Bayes Classifier

A from-scratch Bernoulli-style Naive Bayes classifier that labels emails from the TREC06 corpus as **spam** or **ham**. The notebook covers:

1. Building and evaluating a baseline classifier (no smoothing)
2. Lambda (additive) smoothing
3. Shrinking the vocabulary to 200 informative words with mutual information
4. Extra: cleaning the email bodies (HTML stripping, stopwords, token length filter)

The full write-up is in [`ocon_naive_bayes_pa1.pdf`](ocon_naive_bayes_pa1.pdf).

## Repository contents

| File | Description |
|---|---|
| `naive_bayes_improved.ipynb` | Notebook with all the code and outputs |
| `ocon_naive_bayes_pa1.pdf` | Report |
| `train_set.txt` | Training split (26,447 emails), one `label path` per line |
| `test_set.txt` | Test split (11,335 emails), same format |

The dataset itself (`trec06p-cs280`, about 90 MB zipped) is **not** in the repository. See the setup steps below.

---

## The model

### Document representation

Each email is turned into the **set** of distinct lowercase words it contains, so the model only cares whether a word appears, not how many times. A word is a run of letters with whitespace before it and whitespace, a comma, a period, or the end of the text after it:

```python
word_pattern = re.compile(r"(?<!\S)[a-zA-Z]+(?=[\s,.]|\Z)")
```

The vocabulary $V$ is every word that appears in at least one training email.

### Priors

With $N_{\text{spam}}$ spam and $N_{\text{ham}}$ ham training emails:

$$
P(\text{spam}) = \frac{N_{\text{spam}}}{N_{\text{spam}} + N_{\text{ham}}}, \qquad
P(\text{ham}) = \frac{N_{\text{ham}}}{N_{\text{spam}} + N_{\text{ham}}}
$$

For this split, $P(\text{spam}) \approx 0.6586$ and $P(\text{ham}) \approx 0.3414$.

### Likelihoods

Let $n_c(w)$ be the number of training emails of class $c$ that contain word $w$.

**Without smoothing** ($\lambda = 0$):

$$
P(w \mid c) = \frac{n_c(w)}{N_c}
$$

If a word never appears in a class, this probability is 0 and its log is $-\infty$.

**With lambda smoothing:**

$$
P(w \mid c) = \frac{n_c(w) + \lambda}{N_c + \lambda\,|V|}
$$

### Classification

By Bayes' rule and the naive independence assumption, an email with word set $W$ is scored per class as

$$
\text{score}(c) = \log P(c) + \sum_{w \in W \cap V} \log P(w \mid c)
$$

and the prediction is

$$
\hat{c} = \arg\max_{c \in \{\text{spam},\,\text{ham}\}} \text{score}(c)
$$

Logs are used so that multiplying many small probabilities does not underflow. Words not in $V$ are ignored. The posterior probability of spam is recovered with a stable softmax:

$$
P(\text{spam} \mid W) = \frac{e^{\text{score}(\text{spam}) - m}}{e^{\text{score}(\text{spam}) - m} + e^{\text{score}(\text{ham}) - m}}, \qquad m = \max\big(\text{score}(\text{spam}), \text{score}(\text{ham})\big)
$$

### Feature selection with mutual information

To pick the 200 most useful words, each word is scored by the mutual information between "word $w$ is present" ($X \in \{0,1\}$) and the class $C$:

$$
I(X; C) = \sum_{c} \sum_{x \in \{0,1\}} P(c)\,P(x \mid c)\,\log\frac{P(x \mid c)}{P(x)}
$$

where $P(X{=}1 \mid c)$ is the smoothed likelihood above, $P(X{=}0 \mid c) = 1 - P(X{=}1 \mid c)$, and

$$
P(X{=}1) = P(X{=}1 \mid \text{spam})\,P(\text{spam}) + P(X{=}1 \mid \text{ham})\,P(\text{ham})
$$

Words in fewer than 3 training emails are skipped. A second variant drops the 200 most frequent words before ranking.

### Evaluation metrics

Spam is the positive class.

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \qquad
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

## Results

70/30 stratified split (seed 42), evaluated on 11,335 test emails.

### Lambda smoothing (full vocabulary, 78,278 words)

| λ | TP | TN | FP | FN | Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (none) | 6851 | 3866 | 4 | 614 | 0.9994 | 0.9177 | 0.9568 | 0.9455 |
| 2.0 | 7375 | 3797 | 73 | 90 | 0.9902 | **0.9879** | **0.9891** | **0.9856** |
| 1.0 | 7351 | 3816 | 54 | 114 | 0.9927 | 0.9847 | 0.9887 | 0.9852 |
| 0.5 | 7292 | 3837 | 33 | 173 | 0.9955 | 0.9768 | 0.9861 | 0.9818 |
| 0.1 | 6973 | 3864 | 6 | 492 | 0.9991 | 0.9341 | 0.9655 | 0.9561 |
| 0.005 | 6652 | 3869 | 1 | 813 | **0.9999** | 0.8911 | 0.9423 | 0.9282 |

Smaller λ gives fewer false positives (higher precision) but misses more spam. λ = 2.0 has the best F1 and is used for the later experiments. Without smoothing, 632 test emails get a score of $-\infty$ for both classes.

### All approaches

| Approach | Vocab | λ | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Full vocabulary, no smoothing | 78278 | – | 0.9994 | 0.9177 | 0.9568 | 0.9455 |
| **Full vocabulary, best λ** | 78278 | 2.0 | 0.9902 | 0.9879 | **0.9891** | **0.9856** |
| MI 200 words | 200 | 2.0 | 0.9698 | 0.9456 | 0.9575 | 0.9448 |
| MI 200 words, 200 most frequent removed | 200 | 2.0 | 0.9713 | 0.9476 | 0.9593 | 0.9471 |
| Cleaned full vocabulary | 28084 | 2.0 | 0.9848 | 0.8955 | 0.9380 | 0.9221 |
| Cleaned document-frequency 200 words | 200 | 2.0 | 0.8762 | 0.9971 | 0.9327 | 0.9052 |
| Cleaned MI 200 words | 200 | 2.0 | 0.9696 | 0.7396 | 0.8391 | 0.8132 |
| Cleaned MI 200 words, 200 most frequent removed | 200 | 2.0 | 0.9511 | 0.9064 | 0.9282 | 0.9076 |

The best result overall is the full vocabulary with λ = 2.0. Reading only the cleaned email body hurt performance, which suggests the email headers (mail servers, client names like `outlook` or `foxmail`) carry a lot of the signal.

---

## How to run

### 1. Requirements

- Python 3.9 or newer (the notebook uses `str.removeprefix`)
- Jupyter

Only the Python standard library is used (`re`, `math`, `collections`, `email`, `csv`, `pathlib`, `random`), so there are no packages to install besides Jupyter:

```bash
pip install notebook
```

### 2. Get the dataset

Download the `trec06p-cs280` dataset (the class-provided version of TREC 2006 Public Spam Corpus) and unzip it. It should contain:

```
trec06p-cs280/
├── labels          # lines like "spam ../data/047/039"
└── data/
    ├── 000/
    │   ├── 000.eml
    │   └── ...
    └── ...
```

### 3. Put the files in place

The notebook looks for the dataset at `../trec06p-cs280` and writes outputs to `../results`, both relative to where the notebook runs. The simplest layout is:

```
project/
├── trec06p-cs280/            # unzipped dataset
├── results/                  # created automatically
└── Naive-Bayes-Classifier/    # this repository
    └── naive_bayes_improved.ipynb
```

If your layout is different, change these two lines in the first code cell:

```python
dataset_dir = Path("../trec06p-cs280")
results_dir = Path("../results")
```

### 4. Run the notebook

```bash
jupyter notebook naive_bayes_improved.ipynb
```

Then choose **Run All**. A full run reads all ~37,800 emails a few times, so it takes a few minutes.

### Output files

These are saved to `results/`:

| File | Contents |
|---|---|
| `train_set.txt`, `test_set.txt` | The train/test split |
| `lambda_metrics.csv` | Metrics for each λ |
| `top_200_informative_words.csv` | Words chosen by mutual information |
| `mi_attribute_selection_metrics.csv` | Results using only 200 words |
| `lambda_metrics_after_cleanup.csv` | λ results on cleaned email bodies |
| `overall_approach_comparison.csv` / `.md` | Comparison of all approaches |

### Classifying your own message

After running the cells up to the λ selection, you can test any text:

```python
classify_unknown("""
Congratulations, you won free money. Claim your prize now.
""", best_model)
```

```
Prediction: spam
P(spam | message): 0.99997...
Words used: ['claim', 'congratulations', 'free', 'money', 'now', 'prize', 'won', 'you', 'your']
```

---

## License

[MIT](LICENSE) © 2026 Guy Ivan Ocon
