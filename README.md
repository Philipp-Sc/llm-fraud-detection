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

# Evaluation

## Training Data
Trained and tested with the following datasets.

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
For the datasets above the topic predictions were generated, note that his is quite compute intensive.
The language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time. A GPU Setup is recommended.

## 'hard-coded' features
Additional ['hard-coded' features](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/feature_engineering/mod.rs) to improve the performance further have been evaluated. (See [Feature Selection](#feature-selection))

### Counts

- word count information 
- punctuation, number, url, emoji and upper-case counts
  
### Naive Bayes
Naive Bayes classifier are well known for their effectiveness in text related tasks especially spam detection.  

Two Naive Bayes classifiers have been trained on a Bag of Words representation of the spam/ham datasets.
- [Categorical Naive Bayes classifier](https://docs.rs/smartcore/latest/smartcore/naive_bayes/categorical/struct.CategoricalNB.html) 
- [Gaussian Naive Bayes classifier](https://docs.rs/crate/linfa-bayes/latest)
 

## Model Eval 

First the random forest regressor has been trained on the whole dataset. This way the permutation feature [importance](https://github.com/Philipp-Sc/importance) of each feature can be calculated using the best model.
```
Test results on train=test dataset.
```
``` 
Threshold >= 0.4: True Positive = 8876, False Positive = 128, Precision = 0.986, Recall = 0.985, F-Score = 0.985
```

## Feature Selection
The permutation feature importance gives insights into if a feature were to be 'removed' how big of an impact that would have on the prediction error.
```
Permutation feature importance means:
```
```rust
    (0.3985863423201774, "gaussian_nb_model_predict"),
    (0.14208074954988376, "categorical_nb_model_predict"),
    (0.05072013818401902, "word_count.characters"),
    (0.04691123690753006, "Ethical advertising practices"),
    (0.045243339951067274, "RE_PUNCTUATION"),
    (0.03684411074134846, "Promotion of responsible digital citizenship"),
    (0.036669126452287315, "Untrustworthy, not to be trusted, unreliable source, blacklisted"),
    (0.032868717073914214, "word_count.whitespaces"),
    (0.032794123803081186, "Political bias or agenda"),
    (0.03093559093435697, "RE_EMOJI"),
    (0.02994326629473548, "RE_NON_STANDARD"),
    (0.029406976115361624, "word_count.words"),
    (0.029013546987538587, "Reputable source"),
    ...
```
[feature_importance.json](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/feature_importance.txt)

```
To evaluate the selected features we evaluate them on the train/test dataset with split ratio of 80%/20%.
```

```
Selected features with importance >= 0.01:
Threshold >= 0.4: True Positive = 1700, False Positive = 94, Precision = 0.948, Recall = 0.939, F-Score = 0.943

Selected features with importance >= 0.015:
Threshold >= 0.4: True Positive = 1692, False Positive = 100, Precision = 0.944, Recall = 0.922, F-Score = 0.933

Selected features with importance >= 0.02:
Threshold >= 0.4: True Positive = 1680, False Positive = 119, Precision = 0.934, Recall = 0.911, F-Score = 0.922
```
First of all we can see that the model works also well on data it was not trained on.
Secondly the selection with importance >= 0.01 still gets a performance close the the model trained on all 168 features, with only 35 of the total features!
```
Test results on train=test dataset.
```
```
Selected features with importance >= 0.01:
Threshold >= 0.4: True Positive = 8812, False Positive = 204, Precision = 0.977, Recall = 0.978, F-Score = 0.978
```

Feel free to evaluate your own features 'hard-coded' or topic classes and find out if you can improve the model.     
Or just use the provided model to get rid of spam!
 

# Outlook
- **rust-bert-fraud-detection** may be improved by finding a better set of [topics/fraud indicators](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) to be extracted and used for the classification. 
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

