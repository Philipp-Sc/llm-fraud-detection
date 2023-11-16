<div align="center">
<img src="https://img.shields.io/github/languages/top/Philipp-Sc/llm-fraud-detection">
<img src="https://img.shields.io/github/repo-size/Philipp-Sc/llm-fraud-detection">
<img src="https://img.shields.io/github/commit-activity/m/Philipp-Sc/llm-fraud-detection">
<img src="https://img.shields.io/github/license/Philipp-Sc/llm-fraud-detection">
</div>

# llm-fraud-detection
Robust semi-supervised fraud detection using Rust native NLP pipelines.
# About
**llm-fraud-detection** uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) and [llama.cpp](https://github.com/ggerganov/llama.cpp) to extract **topic predictions** and **text embeddings** from a given text to predict the fraud likelihood.

The training data is generated from a diverse collection of commonly used spam/ham datasets:
- Ling Spam,
- Enron Spam,
- Spam Assassin Dataset,
- SMS Spam Collection,
- Youtube Spam,
- Crypto Governance Proposals.

**llm-fraud-detection** archives state-of-the-art performance without fine tuning the LLMs directly, instead the outputs of the LLMs (embeddings, zero shot classification) are trained on and used for the spam/ham classification task.

# Use

Git clone and train the required models:

```bash
cargo run --release train_and_test_text_embedding_knn_regressor
cargo run --release train_and_test_zero_shot_classification_random_forest_regressor
cargo run --release train_and_test_mixture_model
```
(the training data is provided in this repository, the models are not due to size limitations)

Add to your `Cargo.toml` manifest:

```ini
[dependencies]
rust_fraud_detection_tools = { git="https://github.com/Philipp-Sc/llm-fraud-detection.git" }
```

Predict fraud/ham:
```rust

pub const SENTENCES: [&str;6] = [
    "Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];


fn main() -> anyhow::Result<()> {

    let fraud_probabilities: Vec<f32> = rust_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}

```
``` 
[0.16443461, 0.0062025306, 0.6938212, 0.0014256272, 0.9994333, 0.043457787]


[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
```

# Architecture

## Features

### KNN Regressor on Text Embeddings (llama.cpp) ⭐
Single fraud likelihood prediction based on text embeddings.
<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.948) </b> </summary>

```
Performance on the training data (80%)
```
```rust
Threshold >= 0.1: True Positive = 7180, False Positive = 890, Precision = 0.890, Recall = 1.000, F-Score = 0.942
Threshold >= 0.2: True Positive = 7180, False Positive = 222, Precision = 0.970, Recall = 1.000, F-Score = 0.985
Threshold >= 0.3: True Positive = 7175, False Positive = 143, Precision = 0.980, Recall = 0.999, F-Score = 0.990
Threshold >= 0.4: True Positive = 7166, False Positive = 65, Precision = 0.991, Recall = 0.998, F-Score = 0.995
Threshold >= 0.5: True Positive = 7166, False Positive = 0, Precision = 1.000, Recall = 0.998, F-Score = 0.999
Threshold >= 0.6: True Positive = 7114, False Positive = 0, Precision = 1.000, Recall = 0.991, F-Score = 0.995
Threshold >= 0.7: True Positive = 7072, False Positive = 0, Precision = 1.000, Recall = 0.985, F-Score = 0.992
Threshold >= 0.8: True Positive = 7063, False Positive = 0, Precision = 1.000, Recall = 0.984, F-Score = 0.992
Threshold >= 0.9: True Positive = 7056, False Positive = 0, Precision = 1.000, Recall = 0.983, F-Score = 0.991
```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1802, False Positive = 539, Precision = 0.770, Recall = 0.984, F-Score = 0.864
Threshold >= 0.2: True Positive = 1798, False Positive = 395, Precision = 0.820, Recall = 0.981, F-Score = 0.893
Threshold >= 0.3: True Positive = 1790, False Positive = 317, Precision = 0.850, Recall = 0.977, F-Score = 0.909
Threshold >= 0.4: True Positive = 1742, False Positive = 123, Precision = 0.934, Recall = 0.951, F-Score = 0.942
Threshold >= 0.5: True Positive = 1739, False Positive = 103, Precision = 0.944, Recall = 0.949, F-Score = 0.947
Threshold >= 0.6: True Positive = 1736, False Positive = 95, Precision = 0.948, Recall = 0.948, F-Score = 0.948
Threshold >= 0.7: True Positive = 1658, False Positive = 49, Precision = 0.971, Recall = 0.905, F-Score = 0.937
Threshold >= 0.8: True Positive = 1645, False Positive = 47, Precision = 0.972, Recall = 0.898, F-Score = 0.934
Threshold >= 0.9: True Positive = 1640, False Positive = 46, Precision = 0.973, Recall = 0.895, F-Score = 0.932
```
</details>



### Random Forest on BERT Topic Predictions (via zero shot classification) ⭐
Single fraud likelihood prediction based on the zero shot classification. 

<details>
<summary> <b>Feature Selection </b> </summary>

Topic predictions (fraud indicators) via zero shot classification using the BERT language model were generated, note that his is quite compute intensive.
The language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time. A GPU Setup is recommended.

## 
Since generating the topic predictions requires a lot of compute the permutation feature [importance](https://github.com/Philipp-Sc/importance) is used to enable us to focus only on the most relevant topics.     
```
Permutation feature importance:
```
```json 
   {
  "feature_importance": [
    [
      0.11508008431692174,
      "Ethical advertising practices"
    ],
    [
      0.09978720920624455,
      "Untrustworthy, not to be trusted, unreliable source, blacklisted"
    ],
    [
      0.08417019000170248,
      "Promotion of responsible digital citizenship"
    ],
    [
      0.07622704037823842,
      "Aggressive marketing, advertising, selling, promotion, authoritative, commanding"
    ],
    [
      0.06750496943657096,
      "Political bias or agenda"
    ],
    [
      0.058185966973215826,
      "Suspicious, questionable, dubious"
    ],
...
```
See more [feature_importance.json](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/feature_importance_random_forest_topics_only.json)


</details>

<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.841) </b> </summary>

```
Performance on the training data (80%)
```
```rust
Threshold >= 0.1: True Positive = 7223, False Positive = 5440, Precision = 0.570, Recall = 1.000, F-Score = 0.726
Threshold >= 0.2: True Positive = 7220, False Positive = 2053, Precision = 0.779, Recall = 0.999, F-Score = 0.875
Threshold >= 0.3: True Positive = 7200, False Positive = 725, Precision = 0.909, Recall = 0.997, F-Score = 0.950
Threshold >= 0.4: True Positive = 7123, False Positive = 227, Precision = 0.969, Recall = 0.986, F-Score = 0.977
Threshold >= 0.5: True Positive = 6911, False Positive = 52, Precision = 0.993, Recall = 0.957, F-Score = 0.974
Threshold >= 0.6: True Positive = 6466, False Positive = 8, Precision = 0.999, Recall = 0.895, F-Score = 0.944
Threshold >= 0.7: True Positive = 5688, False Positive = 0, Precision = 1.000, Recall = 0.787, F-Score = 0.881
Threshold >= 0.8: True Positive = 4461, False Positive = 0, Precision = 1.000, Recall = 0.617, F-Score = 0.763
Threshold >= 0.9: True Positive = 2762, False Positive = 0, Precision = 1.000, Recall = 0.382, F-Score = 0.553
```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1762, False Positive = 2044, Precision = 0.463, Recall = 0.986, F-Score = 0.630
Threshold >= 0.2: True Positive = 1703, False Positive = 1063, Precision = 0.616, Recall = 0.953, F-Score = 0.748
Threshold >= 0.3: True Positive = 1621, False Positive = 558, Precision = 0.744, Recall = 0.907, F-Score = 0.817
Threshold >= 0.4: True Positive = 1506, False Positive = 289, Precision = 0.839, Recall = 0.843, F-Score = 0.841
Threshold >= 0.5: True Positive = 1360, False Positive = 139, Precision = 0.907, Recall = 0.761, F-Score = 0.828
Threshold >= 0.6: True Positive = 1212, False Positive = 66, Precision = 0.948, Recall = 0.678, F-Score = 0.791
Threshold >= 0.7: True Positive = 1033, False Positive = 25, Precision = 0.976, Recall = 0.578, F-Score = 0.726
Threshold >= 0.8: True Positive = 763, False Positive = 8, Precision = 0.990, Recall = 0.427, F-Score = 0.597
Threshold >= 0.9: True Positive = 454, False Positive = 1, Precision = 0.998, Recall = 0.254, F-Score = 0.405
```
</details>


### ~~Neural Net on BERT Topic Predictions (via zero shot classification)~~
Performs slightly worse compared to the Random Forest.  
<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.847) </b> </summary>

```
Performance on the training data (80%)
```
```rust
Threshold >= 0.1: True Positive = 6899, False Positive = 2397, Precision = 0.742, Recall = 0.960, F-Score = 0.837
Threshold >= 0.2: True Positive = 6717, False Positive = 1198, Precision = 0.849, Recall = 0.935, F-Score = 0.890
Threshold >= 0.3: True Positive = 6582, False Positive = 824, Precision = 0.889, Recall = 0.916, F-Score = 0.902
Threshold >= 0.4: True Positive = 6468, False Positive = 649, Precision = 0.909, Recall = 0.900, F-Score = 0.905
Threshold >= 0.5: True Positive = 6353, False Positive = 509, Precision = 0.926, Recall = 0.884, F-Score = 0.905
Threshold >= 0.6: True Positive = 6230, False Positive = 392, Precision = 0.941, Recall = 0.867, F-Score = 0.903
Threshold >= 0.7: True Positive = 6110, False Positive = 307, Precision = 0.952, Recall = 0.851, F-Score = 0.898
Threshold >= 0.8: True Positive = 5937, False Positive = 238, Precision = 0.961, Recall = 0.826, F-Score = 0.889
Threshold >= 0.9: True Positive = 5655, False Positive = 156, Precision = 0.973, Recall = 0.787, F-Score = 0.870
```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1631, False Positive = 500, Precision = 0.765, Recall = 0.892, F-Score = 0.824
Threshold >= 0.2: True Positive = 1572, False Positive = 324, Precision = 0.829, Recall = 0.860, F-Score = 0.844
Threshold >= 0.3: True Positive = 1547, False Positive = 281, Precision = 0.846, Recall = 0.846, F-Score = 0.846
Threshold >= 0.4: True Positive = 1524, False Positive = 247, Precision = 0.861, Recall = 0.834, F-Score = 0.847
Threshold >= 0.5: True Positive = 1505, False Positive = 226, Precision = 0.869, Recall = 0.823, F-Score = 0.846
Threshold >= 0.6: True Positive = 1485, False Positive = 201, Precision = 0.881, Recall = 0.812, F-Score = 0.845
Threshold >= 0.7: True Positive = 1466, False Positive = 182, Precision = 0.890, Recall = 0.802, F-Score = 0.843
Threshold >= 0.8: True Positive = 1434, False Positive = 160, Precision = 0.900, Recall = 0.784, F-Score = 0.838
Threshold >= 0.9: True Positive = 1398, False Positive = 141, Precision = 0.908, Recall = 0.765, F-Score = 0.830
```
</details>

### ~~Random Forest on Custom features~~
- word count information 
- punctuation, url, emoji and upper-case counts

<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.692) </b> </summary>

```
Performance on the training data (80%)
```
```rustThreshold >= 0.1: True Positive = 7192, False Positive = 7676, Precision = 0.484, Recall = 0.999, F-Score = 0.652
Threshold >= 0.2: True Positive = 7148, False Positive = 4179, Precision = 0.631, Recall = 0.993, F-Score = 0.772
Threshold >= 0.3: True Positive = 7009, False Positive = 2030, Precision = 0.775, Recall = 0.974, F-Score = 0.863
Threshold >= 0.4: True Positive = 6681, False Positive = 828, Precision = 0.890, Recall = 0.928, F-Score = 0.908
Threshold >= 0.5: True Positive = 6089, False Positive = 255, Precision = 0.960, Recall = 0.846, F-Score = 0.899
Threshold >= 0.6: True Positive = 5309, False Positive = 47, Precision = 0.991, Recall = 0.737, F-Score = 0.846
Threshold >= 0.7: True Positive = 4134, False Positive = 10, Precision = 0.998, Recall = 0.574, F-Score = 0.729
Threshold >= 0.8: True Positive = 2865, False Positive = 0, Precision = 1.000, Recall = 0.398, F-Score = 0.569
Threshold >= 0.9: True Positive = 1525, False Positive = 0, Precision = 1.000, Recall = 0.212, F-Score = 0.350

```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1735, False Positive = 2296, Precision = 0.430, Recall = 0.957, F-Score = 0.594
Threshold >= 0.2: True Positive = 1639, False Positive = 1526, Precision = 0.518, Recall = 0.904, F-Score = 0.658
Threshold >= 0.3: True Positive = 1480, False Positive = 992, Precision = 0.599, Recall = 0.816, F-Score = 0.691
Threshold >= 0.4: True Positive = 1285, False Positive = 616, Precision = 0.676, Recall = 0.709, F-Score = 0.692
Threshold >= 0.5: True Positive = 1100, False Positive = 354, Precision = 0.757, Recall = 0.607, F-Score = 0.673
Threshold >= 0.6: True Positive = 895, False Positive = 178, Precision = 0.834, Recall = 0.494, F-Score = 0.620
Threshold >= 0.7: True Positive = 699, False Positive = 77, Precision = 0.901, Recall = 0.386, F-Score = 0.540
Threshold >= 0.8: True Positive = 505, False Positive = 23, Precision = 0.956, Recall = 0.279, F-Score = 0.431
Threshold >= 0.9: True Positive = 267, False Positive = 6, Precision = 0.978, Recall = 0.147, F-Score = 0.256
```
</details>


### ~~Sentiment Prediction~~
The BERT language model also was used to predict the sentiment of the text.

### ~~Naive Bayes~~
Naive Bayes classifier are well known for their effectiveness in text related tasks especially spam detection.  

Two Naive Bayes classifiers have been trained on a Bag of Words representation of the spam/ham datasets.
- [Categorical Naive Bayes classifier](https://docs.rs/smartcore/latest/smartcore/naive_bayes/categorical/struct.CategoricalNB.html) **(F-Score = 0.870)** 
- [Gaussian Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest) **(F-Score = 0.859)**



  
 
## Final mixture model
Takes in
- KNN Regressor on Text Embeddings (llama.cpp)
- Random Forest on BERT Topic Predictions (via zero shot classification)
and predicts spam/ham.

Other features like Sentiment, Custom Features or Naive Bayes did not meaningfully improve the final mixture model's performance and therefore were not included.
<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.999) </b> </summary>

```
Performance on the training data (80%)
```
```rust
Threshold >= 0.1: True Positive = 7208, False Positive = 817, Precision = 0.898, Recall = 1.000, F-Score = 0.946
Threshold >= 0.2: True Positive = 7208, False Positive = 165, Precision = 0.978, Recall = 1.000, F-Score = 0.989
Threshold >= 0.3: True Positive = 7205, False Positive = 111, Precision = 0.985, Recall = 0.999, F-Score = 0.992
Threshold >= 0.4: True Positive = 7202, False Positive = 65, Precision = 0.991, Recall = 0.999, F-Score = 0.995
Threshold >= 0.5: True Positive = 7199, False Positive = 3, Precision = 1.000, Recall = 0.999, F-Score = 0.999
Threshold >= 0.6: True Positive = 7152, False Positive = 0, Precision = 1.000, Recall = 0.992, F-Score = 0.996
Threshold >= 0.7: True Positive = 7104, False Positive = 0, Precision = 1.000, Recall = 0.985, F-Score = 0.993
Threshold >= 0.8: True Positive = 7098, False Positive = 0, Precision = 1.000, Recall = 0.985, F-Score = 0.992
Threshold >= 0.9: True Positive = 7092, False Positive = 0, Precision = 1.000, Recall = 0.984, F-Score = 0.992
```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1802, False Positive = 203, Precision = 0.899, Recall = 0.999, F-Score = 0.946
Threshold >= 0.2: True Positive = 1802, False Positive = 41, Precision = 0.978, Recall = 0.999, F-Score = 0.988
Threshold >= 0.3: True Positive = 1802, False Positive = 33, Precision = 0.982, Recall = 0.999, F-Score = 0.991
Threshold >= 0.4: True Positive = 1802, False Positive = 20, Precision = 0.989, Recall = 0.999, F-Score = 0.994
Threshold >= 0.5: True Positive = 1802, False Positive = 3, Precision = 0.998, Recall = 0.999, F-Score = 0.999
Threshold >= 0.6: True Positive = 1791, False Positive = 2, Precision = 0.999, Recall = 0.993, F-Score = 0.996
Threshold >= 0.7: True Positive = 1783, False Positive = 2, Precision = 0.999, Recall = 0.989, F-Score = 0.994
Threshold >= 0.8: True Positive = 1780, False Positive = 2, Precision = 0.999, Recall = 0.987, F-Score = 0.993
Threshold >= 0.9: True Positive = 1776, False Positive = 2, Precision = 0.999, Recall = 0.985, F-Score = 0.992
```
</details>


# Evaluation

## Training Data
Trained and tested with the following datasets:
-  enronSpamSubset.csv
-  lingSpam.csv
-  completeSpamAssassin.csv
-  youtubeSpamCollection.csv
-  smsspamcollection.csv
-  governance_proposal_spam_likelihood.csv

```
total: 27.982
---------------
count spam: 9.012
count ham: 18.970
---------------
```
<details>
<summary> <b>Expand to display the full dataset breakdown </b> </summary>

```
enronSpamSubset.csv
---------------
count spam: 5000
count ham: 5000

lingSpam.csv
---------------
count spam: 433
count ham: 2172

completeSpamAssassin.csv
---------------
count spam: 1560
count ham: 3952

youtubeSpamCollection.csv
---------------
count spam: 1005
count ham: 951
 
smsspamcollection.csv
---------------
count spam: 747
count ham: 4825

governance_proposal_spam_likelihood.csv
--------------- 
count spam: ?
count ham: ?
``` 
</details>




Feel free to evaluate your own features 'hard-coded' or topic classes and find out if you can improve the model.     
Or just use the provided model to get rid of spam!
 

# Outlook
- **llm-fraud-detection** may be improved by finding a better set of [topics/fraud indicators](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) to be extracted and used for the classification.
- Feature selection: using more topic predictions (limited by available compute)
- Replace BERT with a better language model (zero shot classification, ~~sentiment prediction~~, ~~text embedding~~ (llama.cpp))
- More training data. (All models were trained on **English** text)
- ~~Improving the performance of the [Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest), including adjustments to the used count vectorizer.~~
- ~~Replacing the Random Forest Regressor with a Neural Network.~~
- ~~Improving the KNN Regressor (find a better ```k``` value)~~
- ~~Find better custom features~~ (Not in the spirit of ML)

# 
This project is part of [CosmosRustBot](https://github.com/Philipp-Sc/cosmos-rust-bot), which provides Governance Proposal Notifications for Cosmos Blockchains. The goal is automatically detect fraudulent and deceitful proposals to prevent users falling for crypto scams. The current model is very effective in detecting fake governance proposals.

