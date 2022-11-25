<div align="center">
<img src="https://img.shields.io/github/languages/top/Philipp-Sc/rustbert-fraud-detection">
<img src="https://img.shields.io/github/repo-size/Philipp-Sc/rustbert-fraud-detection">
<img src="https://img.shields.io/github/commit-activity/m/Philipp-Sc/rustbert-fraud-detection">
<img src="https://img.shields.io/github/license/Philipp-Sc/rustbert-fraud-detection">
</div>

# rustbert-fraud-detection
Robust semi-supervised fraud detection using Rust native NLP pipelines.
# About
**rustbert-fraud-detection** uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract topics and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained to predict fraud/ham. The training data is generated from the [LingSpam, EnronSpam and Spam Assassin Dataset](https://www.kaggle.com/datasets/nitishabharathi/email-fraud-dataset) containing ham and fraud email. Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features ([topics/fraud indicators](https://github.com/Philipp-Sc/rustbert-fraud-detection/blob/main/package/src/build/mod.rs)) and NOT on a text encoding (such as Bag of Words) much less datapoints are needed to generate an accurate model.
# 
This project is part of [CosmosRustBot](https://github.com/Philipp-Sc/cosmos-rust-bot), which provides Governance Proposal Notifications for Cosmos Blockchains. To detect fake & fraud proposals **rustbert-fraud-detection** was created. Since **rustbert-fraud-detection** is semi-supervised it is works accross different domains, even though the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) was trained only on a fraud/ham email dataset.
#
Note that the language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time.
# Use

Add to your `Cargo.toml` manifest:

```ini
[dependencies]
rust_fraud_detection_tools = { git="https://github.com/Philipp-Sc/rustbert-fraud-detection.git" }
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
[0.9183035714285714, 0.6243303571428571, 0.9877232142857143, 0.5344494047619046, 0.9184523809523809, 0.6588541666666666]
[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
```
# Model Performance
```len of x_dataset / y_dataset: 13986``` (LingSpam, EnronSpam and Spam Assassin Dataset)
``` 
true p(>=0.4)==label 13319
false 667
false positive 575

true p(>=0.5)==label 13428
false 558
false positive 218

true p(>=0.6)==label 13112
false 874
false positive 66

true p(>=0.7)==label 12372
false 1614
false positive 5

true p(>=0.8)==label 11311
false 2675
false positive 0

true p(>=0.9)==label 9898
false 4088
false positive 0
```
- p(>=0.5) has the best performance (95%).
- p(>=0.7) has the fewest **false positives** and a performance of 87%.

If you are okay with few fraud emails not classified as fraud, but don't want any ham email classified as fraud, select the later.

# 
- **rustbert-fraud-detection** can be further improved by finding a better set of [topics/fraud indicators](https://github.com/Philipp-Sc/rustbert-fraud-detection/blob/main/package/src/build/mod.rs) to be extracted and used for the classification. 
- Using a better model for the topic extraction and sentiment prediction should also improve the fraud detection.
- Replacing the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) with a better model might also improve the performance. 

# Test with Docker
```sudo docker build -t rust-devel-image .``` (build image)

```sudo docker run -it --rm -v "$(pwd)/rustbert_cache":/usr/rustbert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace rust-devel-image cargo run --release``` (runs the above example)
