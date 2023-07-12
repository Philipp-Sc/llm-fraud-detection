<div align="center">
<img src="https://img.shields.io/github/languages/top/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/repo-size/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/commit-activity/m/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/license/Philipp-Sc/rust-bert-fraud-detection">
</div>

# rust-bert-fraud-detection
Robust semi-supervised fraud detection using Rust native NLP pipelines.
# About
**rust-bert-fraud-detection** uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract [topic predictions](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) (fraud indicators) via zero shot classification and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is then trained to predict spam/ham.     

The training data is generated from a diverse collection of commonly used spam/ham datasets:
- Ling Spam,
- Enron Spam,
- Spam Assassin Dataset,
- SMS Spam Collection,
- Youtube Spam,
- Crypto Governance Proposals.

Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features  and NOT on a text encoding (such as Bag of Words) much less datapoints are needed to build a general and robust model which should work better accross different domains (e.g emails, websites and governance proposals).


# Use

Add to your `Cargo.toml` manifest:

```ini
[dependencies]
rust_fraud_detection_tools = { git="https://github.com/Philipp-Sc/rust-bert-fraud-detection.git" }
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

    let fraud_probabilities: Vec<f64> = rust_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}

```
``` 
[0.7556460210069226, 0.10628275662417444, 0.5515625, 0.0, 0.6143134447463007, 0.15900739517397644]

[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
```

# Architecture

## Models
The fraud detection consists of a Random Forest Regressor and a set of predictors (Naive Bayes, Random Forest, Neural Net) whose predictions are provided to the model.

### 1. Naive Bayes
Naive Bayes classifier are well known for their effectiveness in text related tasks especially spam detection.  

Two Naive Bayes classifiers have been trained on a Bag of Words representation of the spam/ham datasets.
- [Categorical Naive Bayes classifier](https://docs.rs/smartcore/latest/smartcore/naive_bayes/categorical/struct.CategoricalNB.html) **(F-Score = 0.870)** 
- [Gaussian Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest) **(F-Score = 0.859)**

### 2. Random Forest
Performs exceptionally well when trained on the topic predictions, sentiment and custom features.
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

### 3. Neural Net 
Performs equally as well as the Random Forest, both have different strenghts and weaknesses, adding both leads to the best fraud detection.
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

## Final aggregator model
Takes in the predictions from 1-3 and predicts spam/ham.

<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.982) </b> </summary>

```
Performance on the training data (80%)
```
```rust
Threshold >= 0.1: True Positive = 7158, False Positive = 466, Precision = 0.939, Recall = 0.998, F-Score = 0.967
Threshold >= 0.2: True Positive = 7117, False Positive = 174, Precision = 0.976, Recall = 0.992, F-Score = 0.984
Threshold >= 0.3: True Positive = 7099, False Positive = 110, Precision = 0.985, Recall = 0.989, F-Score = 0.987
Threshold >= 0.4: True Positive = 7074, False Positive = 82, Precision = 0.989, Recall = 0.986, F-Score = 0.987
Threshold >= 0.5: True Positive = 7058, False Positive = 53, Precision = 0.993, Recall = 0.984, F-Score = 0.988
Threshold >= 0.6: True Positive = 7037, False Positive = 36, Precision = 0.995, Recall = 0.981, F-Score = 0.988
Threshold >= 0.7: True Positive = 7000, False Positive = 25, Precision = 0.996, Recall = 0.976, F-Score = 0.986
Threshold >= 0.8: True Positive = 6964, False Positive = 9, Precision = 0.999, Recall = 0.971, F-Score = 0.984
Threshold >= 0.9: True Positive = 6606, False Positive = 2, Precision = 1.000, Recall = 0.921, F-Score = 0.959
```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1827, False Positive = 143, Precision = 0.927, Recall = 0.995, F-Score = 0.960
Threshold >= 0.2: True Positive = 1811, False Positive = 58, Precision = 0.969, Recall = 0.986, F-Score = 0.977
Threshold >= 0.3: True Positive = 1809, False Positive = 43, Precision = 0.977, Recall = 0.985, F-Score = 0.981
Threshold >= 0.4: True Positive = 1803, False Positive = 36, Precision = 0.980, Recall = 0.981, F-Score = 0.981
Threshold >= 0.5: True Positive = 1802, False Positive = 32, Precision = 0.983, Recall = 0.981, F-Score = 0.982
Threshold >= 0.6: True Positive = 1792, False Positive = 23, Precision = 0.987, Recall = 0.976, F-Score = 0.981
Threshold >= 0.7: True Positive = 1785, False Positive = 16, Precision = 0.991, Recall = 0.972, F-Score = 0.981
Threshold >= 0.8: True Positive = 1776, False Positive = 11, Precision = 0.994, Recall = 0.967, F-Score = 0.980
Threshold >= 0.9: True Positive = 1676, False Positive = 3, Precision = 0.998, Recall = 0.912, F-Score = 0.953
```
</details>




## Features

### Topic Predictions
Topic predictions (fraud indicators) via zero shot classification using the BERT language model were generated, note that his is quite compute intensive.
The language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time. A GPU Setup is recommended.


### Sentiment Prediction
The BERT language model also was used to predict the sentiment of the text.

### Custom features
[Custom features](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/feature_engineering/mod.rs) to improve the performance further have been added.

- word count information 
- punctuation, url, emoji and upper-case counts


## Feature Selection
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
- **rust-bert-fraud-detection** may be improved by finding a better set of [topics/fraud indicators](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) to be extracted and used for the classification.
- Feature selection: using more topic predictions (limited by available compute)
- Replace BERT with a better language model (zero shot classification, sentiment prediction)
- Improving the performance of the [Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest), including adjustments to the used count vectorizer.
- Adding new custom features.
- More (domain specific) training data. (All models were only trained on English text)

# Docker
```sudo docker build -t rust-bert-fraud-detection .``` (build image)

## Test

```sudo docker run -it --rm -v "$(pwd)/rust_bert_cache":/usr/rust_bert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release``` (runs the above example)

## Run as a service via UNIX sockets

- Start service container:   
```sudo docker run -d --rm -v "$(pwd)/rust_bert_cache":/usr/rust_bert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release start_service```

- Run service test:     
```sudo docker run -it --rm -v "$(pwd)/rust_bert_cache":/usr/rust_bert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release --no-default-features test_service```
- Check out the ```Cargo.toml``` and ```main.rs``` to see how to send a request from your own Rust Application via UNIX sockets.

- To later stop the service container:     
```sudo docker container ls```    
```sudo docker stop CONTAINER_ID```


# 
This project is part of [CosmosRustBot](https://github.com/Philipp-Sc/cosmos-rust-bot), which provides Governance Proposal Notifications for Cosmos Blockchains. The goal is automatically detect fraudulent and deceitful proposals to prevent users falling for crypto scams. The current model is very effective in detecting fake governance proposals.

