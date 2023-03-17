<div align="center">
<img src="https://img.shields.io/github/languages/top/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/repo-size/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/commit-activity/m/Philipp-Sc/rust-bert-fraud-detection">
<img src="https://img.shields.io/github/license/Philipp-Sc/rust-bert-fraud-detection">
</div>

# rust-bert-fraud-detection
Robust semi-supervised fraud detection using Rust native NLP pipelines.
# About
**rust-bert-fraud-detection** uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract topics and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained to predict fraud/ham. The training data is generated using a diverse collection of commonly used spam/ham datasets (LingSpam, EnronSpam, Spam Assassin Dataset, SMSSpamCollection, YoutubeSpam, ...). Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features ([topics/fraud indicators](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs)) and NOT on a text encoding (such as Bag of Words) much less datapoints are needed to build a general model.
# 
This project is part of [CosmosRustBot](https://github.com/Philipp-Sc/cosmos-rust-bot), which provides Governance Proposal Notifications for Cosmos Blockchains. To detect fake & fraud proposals **rust-bert-fraud-detection** was created. Since **rust-bert-fraud-detection** is semi-supervised it is works accross different domains, even though the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) was trained only on a fraud/ham email dataset.
Recently trainig data from Governance Proposals of Cosmos Blockchains were added, to further improve the accuracy.
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
[0.8911458333333334, 0.1259672619047619, 0.5364583333333331, 0.0984375, 0.8971354166666666, 0.27120535714285715]
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

governance_proposal_spam_ham.csv 
---------------
count spam: 26
count ham: 780

TOTAL SPAM: 8771
TOTAL HAM: 17680

```
# Model Performance 

```
Trained and tested with
---------------
count spam: 7766
count ham: 16729
``` 
```
true p(>=0.1)==label 19847
false 4646
false positive 4646

true p(>=0.2)==label 22703
false 1790
false positive 1788

true p(>=0.3)==label 23725
false 768
false positive 747

true p(>=0.4)==label 24067
false 426
false positive 288

true p(>=0.5)==label 23927
false 566
false positive 104

true p(>=0.6)==label 23504
false 989
false positive 27

true p(>=0.7)==label 22748
false 1745
false positive 0

true p(>=0.8)==label 21648
false 2845
false positive 0

true p(>=0.9)==label 20049
false 4444
false positive 0

```
- p(>=0.4) has the best performance (98,2%).

```Note: This makes sense because the training data contains more ham than spam entries.```
- p(>=0.5) has the second best performance (97,6%), with a lot less **false positives** (ham incorrectly classified as spam).
- p(>=0.7) has the fewest **false positives** and a performance of 88,3%.

```If you are okay with few emails incorrectly not classified as fraud and do not want any ham email classified as fraud, select the later.```

# 
- **rust-bert-fraud-detection** can be further improved by finding a better set of [topics/fraud indicators](https://github.com/Philipp-Sc/rust-bert-fraud-detection/blob/main/package/src/build/mod.rs) to be extracted and used for the classification. 
- Using a better model for the topic extraction and sentiment prediction should also improve the fraud detection.
- Replacing the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) with a better model (Neural Network) might also improve the performance. 
- More training data is always good.

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
