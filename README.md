# rustbert-spam-detection
Robust semi-supervised spam detection using Rust native NLP pipelines.
#
rustbert-spam-detection uses the NLP pipelines from [rust-bert](https://github.com/guillaume-be/rust-bert) to extract topics and sentiment from the given text. A simple [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained to predict spam/ham. The training data is generated from the [LingSpam, EnronSpam and Spam Assassin Dataset](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset) containing ham and spam email. Since the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) is trained on latent features (the topic predictions) and NOT on a text encoding (such as Bag of Words, etc) much less datapoints are needed to generate an accurate model.
#
Note that the language models used by [rust-bert](https://github.com/guillaume-be/rust-bert) are in the order of the 100s of MBs to GBs. This impacts the hardware requirements and model inference time.
#
This model can be further improved by finding a better set of topics to be extracted and used for the classification. Replacing the [Random Forest Regressor](https://docs.rs/smartcore/latest/smartcore/ensemble/random_forest_regressor/index.html) with a better model might also improve the performance.
# Model Performance
```len of x_dataset / y_dataset: 13986
true p(>=0.1)==label 9198
false 4788
false positive 4788

true p(>=0.2)==label 11406
false 2580
false positive 2580

true p(>=0.3)==label 12721
false 1265
false positive 1261

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
- Select p(>=0.5) for the best performance, 95%.
- Select p(>=0.7) to reduce false positives, has a performance of 87%.
