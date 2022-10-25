# rustbert-spam-detection
Semi-supervised spam detection using Rust native NLP pipelines.
#
rustbert-spam-detection uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract topics and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained to predict spam/ham. The training data is generated from the [LingSpam, EnronSpam and Spam Assassin Dataset](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset) containing ham and spam email.
#
Note that the language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time.
#
This model can be further improved by finding a better set of topics to be extracted and used for the classification. Replacing the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) with a better model can also improve the performance.
