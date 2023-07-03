<div align="center">
<img src="https://img.shields.io/github/languages/top/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/repo-size/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/commit-activity/m/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/license/Philipp-Sc/rust-bert-fraud-detection">
</div>

# rust-bert-fraud-detection
Robust semi-supervised fraud detection using Rust native NLP pipelines.
# About
**rust-bert-fraud-detection** uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract topics and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is then trained to predict spam/ham.     
The training data is generated using a diverse collection of commonly used spam/ham datasets:
- LingSpam,
- EnronSpam,
- Spam Assassin Dataset,
- SMSSpamCollection,
- YoutubeSpam,
- Crypto Governance Proposals.

Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features [Topics (fraud indicators)](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) and NOT on a text encoding (such as Bag of Words) much less datapoints are needed to build a general model which should work better accross different domains (e.g emails, websites and governance proposals).
#
Nonetheless **rust-bert-fraud-detection** uses an additional ['hard-coded' features](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/feature_engineering/mod.rs) to improve the performance further, they include: 

- word count information,
- punctuation, number, url, emoji and upper-case counts.

Additionally the prediction of a [Categorical Naive Bayes classifier](https://docs.rs/smartcore/latest/smartcore/naive_bayes/categorical/struct.CategoricalNB.html) which was trained on a Bag of Words representation of the used spam/ham dataset. Naive Bayes classifier are well known for their effectiveness in text related tasks especially spam detection.   
In the tests the Categorical variant performed better than the [Gaussian Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest) (F1-score of `0.90` vs `0.82`), but both predictions were added to the feature vector.
This addition of the Naive Bayes predictions improves the accuracy of the final Random Forest Regressor from 97% towards 99%.

# 
This project is part of [CosmosRustBot](https://github.com/Philipp-Sc/cosmos-rust-bot), which provides Governance Proposal Notifications for Cosmos Blockchains. The goal is automatically detect fraudulent and deceitful proposals to prevent users falling for crypto scams. The current model is very effective in detecting fake governance proposals.

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
[0.7316134185712937, 0.17592926565828138, 0.7340029761904763, 0.010054703866156384, 0.7400900873289896, 0.11926537059602724]

[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
```
# Training Data
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


Total SPAM/HAM: 27982

```
# Model Eval 
 
Trained and tested with the training data above.

```
Test results on train/test dataset with split ratio 0.7/0.3 
```
``` 
Threshold >= 0.1: True Positive = 2690, False Positive = 3328, Precision = 0.447, Recall = 0.990, F-Score = 0.616
Threshold >= 0.2: True Positive = 2612, False Positive = 1811, Precision = 0.591, Recall = 0.961, F-Score = 0.732
Threshold >= 0.3: True Positive = 2454, False Positive = 934, Precision = 0.724, Recall = 0.903, F-Score = 0.804
Threshold >= 0.4: True Positive = 2280, False Positive = 465, Precision = 0.831, Recall = 0.839, F-Score = 0.835
Threshold >= 0.5: True Positive = 2067, False Positive = 210, Precision = 0.908, Recall = 0.760, F-Score = 0.828
Threshold >= 0.6: True Positive = 1795, False Positive = 89, Precision = 0.953, Recall = 0.660, F-Score = 0.780
Threshold >= 0.7: True Positive = 1458, False Positive = 36, Precision = 0.976, Recall = 0.536, F-Score = 0.692
Threshold >= 0.8: True Positive = 999, False Positive = 7, Precision = 0.993, Recall = 0.368, F-Score = 0.537
Threshold >= 0.9: True Positive = 525, False Positive = 0, Precision = 1.000, Recall = 0.193, F-Score = 0.324
```
```
(Naive Bayes prediction added to the feature vector)
```
```
Threshold >= 0.1: True Positive = 2583, False Positive = 1887, Precision = 0.578, Recall = 0.981, F-Score = 0.727
Threshold >= 0.2: True Positive = 2526, False Positive = 705, Precision = 0.782, Recall = 0.960, F-Score = 0.862
Threshold >= 0.3: True Positive = 2441, False Positive = 267, Precision = 0.901, Recall = 0.927, F-Score = 0.914
Threshold >= 0.4: True Positive = 2346, False Positive = 118, Precision = 0.952, Recall = 0.891, F-Score = 0.921
Threshold >= 0.5: True Positive = 2272, False Positive = 73, Precision = 0.969, Recall = 0.863, F-Score = 0.913
Threshold >= 0.6: True Positive = 2160, False Positive = 53, Precision = 0.976, Recall = 0.821, F-Score = 0.892
Threshold >= 0.7: True Positive = 1951, False Positive = 34, Precision = 0.983, Recall = 0.741, F-Score = 0.845
Threshold >= 0.8: True Positive = 1685, False Positive = 15, Precision = 0.991, Recall = 0.640, F-Score = 0.778
Threshold >= 0.9: True Positive = 1328, False Positive = 2, Precision = 0.998, Recall = 0.505, F-Score = 0.670

```
Increased risk of overfitting, it's important to perform an evaluation with a train/test dataset split, to check if the performance increases for unseen data.

- [x] here adding the Naive Bayes predictions increases the performance for unseen data.

```
Test results on train/test dataset being equal
```
``` 
Threshold >= 0.1: True Positive = 9010, False Positive = 7030, Precision = 0.562, Recall = 1.000, F-Score = 0.719
Threshold >= 0.2: True Positive = 9006, False Positive = 2527, Precision = 0.781, Recall = 0.999, F-Score = 0.877
Threshold >= 0.3: True Positive = 8986, False Positive = 784, Precision = 0.920, Recall = 0.997, F-Score = 0.957
Threshold >= 0.4: True Positive = 8907, False Positive = 182, Precision = 0.980, Recall = 0.988, F-Score = 0.984
Threshold >= 0.5: True Positive = 8718, False Positive = 29, Precision = 0.997, Recall = 0.967, F-Score = 0.982
Threshold >= 0.6: True Positive = 8184, False Positive = 2, Precision = 1.000, Recall = 0.908, F-Score = 0.952
Threshold >= 0.7: True Positive = 7197, False Positive = 1, Precision = 1.000, Recall = 0.799, F-Score = 0.888
Threshold >= 0.8: True Positive = 5671, False Positive = 0, Precision = 1.000, Recall = 0.629, F-Score = 0.772
Threshold >= 0.9: True Positive = 3324, False Positive = 0, Precision = 1.000, Recall = 0.369, F-Score = 0.539
```
```
(Naive Bayes prediction added to the feature vector)
```
```
Threshold >= 0.1: True Positive = 9010, False Positive = 3092, Precision = 0.745, Recall = 1.000, F-Score = 0.853
Threshold >= 0.2: True Positive = 9007, False Positive = 1011, Precision = 0.899, Recall = 0.999, F-Score = 0.947
Threshold >= 0.3: True Positive = 8948, False Positive = 358, Precision = 0.962, Recall = 0.993, F-Score = 0.977
Threshold >= 0.4: True Positive = 8848, False Positive = 169, Precision = 0.981, Recall = 0.982, F-Score = 0.982
Threshold >= 0.5: True Positive = 8707, False Positive = 81, Precision = 0.991, Recall = 0.966, F-Score = 0.978
Threshold >= 0.6: True Positive = 8525, False Positive = 27, Precision = 0.997, Recall = 0.946, F-Score = 0.971
Threshold >= 0.7: True Positive = 8251, False Positive = 3, Precision = 1.000, Recall = 0.916, F-Score = 0.956
Threshold >= 0.8: True Positive = 7770, False Positive = 0, Precision = 1.000, Recall = 0.862, F-Score = 0.926
Threshold >= 0.9: True Positive = 6686, False Positive = 0, Precision = 1.000, Recall = 0.742, F-Score = 0.852
```
 
- p(>=0.4) has the best performance (~98%).

```Note: This makes sense because the training data contains more ham than spam entries.``` 

```If you are okay with few emails incorrectly not classified as fraud and do not want any ham email classified as fraud, select a higher threshold.```
 

# 
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
