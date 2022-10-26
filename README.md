# rustbert-spam-detection
Robust semi-supervised spam detection using Rust native NLP pipelines.
# About
**rustbert-spam-detection** uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract topics and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained to predict spam/ham. The training data is generated from the [LingSpam, EnronSpam and Spam Assassin Dataset](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset) containing ham and spam email. Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features ([topics/scam indicators](https://github.com/Philipp-Sc/rustbert-spam-detection/blob/main/package/src/build/mod.rs)) and NOT on a text encoding (such as Bag of Words) much less datapoints are needed to generate an accurate model.
# 
This project is part of [CosmosRustBot](https://github.com/Philipp-Sc/cosmos-rust-bot), which provides Governance Proposal Notifications for Cosmos Blockchains. To detect fake & scam proposals **rustbert-spam-detection** was created. Since **rustbert-spam-detection** is semi-supervised it is works accross different domains, even though the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) was trained only on a spam/ham email dataset.
#
Note that the language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time.
# Use

Add to your `Cargo.toml` manifest:

```ini
[dependencies]
rust_scam_detection_tools = { git="https://github.com/Philipp-Sc/rustbert-spam-detection.git" }
``` 
Predict spam/ham:
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

    let scam_probabilities: Vec<f64> = rust_scam_detection_tools::scam_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",scam_probabilities);
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

If you are okay with few spam emails not classified as spam, but don't want any ham email classified as spam, select the later.

# 
- **rustbert-spam-detection** can be further improved by finding a better set of topics to be extracted and used for the classification. 
- Using a better model for the topic extraction and sentiment prediction should also improve the spam detection.
- Replacing the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) with a better model might also improve the performance. 

# Test with Docker
```docker-compose build rust-devel``` (build image)   
```docker-compose run --rm rust-devel cargo run --release``` (runs the above example)
