<div align="center">
<img src="https://img.shields.io/github/languages/top/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/repo-size/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/commit-activity/m/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/license/Philipp-Sc/rust-bert-fraud-detection">
</div>

# rust-bert-fraud-detection
Robust semi-supervised fraud detection using Rust native NLP pipelines.
# About
**rust-bert-fraud-detection** uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract [topics](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) (zero shot classification) and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is then trained to predict spam/ham.     

The training data is generated from a diverse collection of commonly used spam/ham datasets:
- Ling Spam,
- Enron Spam,
- Spam Assassin Dataset,
- SMS Spam Collection,
- Youtube Spam,
- Crypto Governance Proposals.

Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features  and NOT on a text encoding (such as Bag of Words) much less datapoints are needed to build a general and robust model which should work better accross different domains (e.g emails, websites and governance proposals).

## 'hard-coded' features
Additional ['hard-coded' features](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/feature_engineering/mod.rs) to improve the performance further have been evaluated and added:

### Various Counts

- word count information 
- punctuation, number, url, emoji and upper-case counts
  
Together these counts increase the models F-Score <ins>by approx. 4 percentage points</ins>.

### Naive Bayes
Naive Bayes classifier are well known for their effectiveness in text related tasks especially spam detection.  

Two Naive Bayes classifiers have been trained on a Bag of Words representation of the spam/ham datasets.
- [Categorical Naive Bayes classifier](https://docs.rs/smartcore/latest/smartcore/naive_bayes/categorical/struct.CategoricalNB.html) 
- [Gaussian Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest)
  
Adding the predictions of both of the classifiers to the 'hard-coded' features increases the F-Score <ins>by approx. 8 percentage points</ins>.

#
Note that the language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time. A GPU Setup is recommended.


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
# Training Data
Trained and tested with the following training data.

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

Total spam/ham: 27982

```
# Model Eval 
 
```
Test results on train/test dataset with split ratio of 80%/20%
```
```
original, only including topics and sentiements:
Threshold >= 0.4: True Positive = 1484, False Positive = 348, Precision = 0.810, Recall = 0.800, F-Score = 0.805

with 'hard-coded' features (counts):
Threshold >= 0.4: True Positive = 1523, False Positive = 298, Precision = 0.836, Recall = 0.847, F-Score = 0.841

with 'hard-coded' features (counts + naive bayes):
Threshold >= 0.5: True Positive = 1603, False Positive = 64, Precision = 0.962, Recall = 0.896, F-Score = 0.928 
```
```
Test results on train=test dataset.
```
```
with 'hard-coded' features (counts + naive bayes):
Threshold >= 0.4: True Positive = 8860, False Positive = 168, Precision = 0.981, Recall = 0.983, F-Score = 0.982
```

```Note: If you are okay with few emails incorrectly not classified as fraud and do not want any ham email classified as fraud, select a higher threshold.```



# Outlook
- **rust-bert-fraud-detection** can be further improved by finding a better set of [topics/fraud indicators](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) to be extracted and used for the classification. 
- Using a better model for the topic extraction and sentiment prediction should also improve the fraud detection.
- Replacing the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) with a better model (Neural Network) might also improve the performance. 
- Improving the performance of the [Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest), including adjustments to the used count vectorizer.
- More (domain specific) training data is always good. (All models were only trained on English text.)

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

