# rustbert-spam-detection
Robust semi-supervised spam detection using Rust native NLP pipelines.
#
rustbert-spam-detection uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract topics and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained to predict spam/ham. The training data is generated from the [LingSpam, EnronSpam and Spam Assassin Dataset](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset) containing ham and spam email. Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features (the topic predictions) and NOT on a text encoding (such as Bag of Words, etc) much less datapoints are needed to generate an accurate model.
#
Note that the language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time.
# 
- This model can be further improved by finding a better set of topics to be extracted and used for the classification. 
- Using a better model for the topic extraction and sentiment prediction should also improve the spam detection.
- Replacing the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) with a better model might also improve the performance. 
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
