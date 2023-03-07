pub mod build;
pub mod cache;

pub mod service;

use crate::build::feature_engineering::get_features;

pub fn fraud_probabilities(texts: &[&str]/*, topics: &[&str]*/) ->  anyhow::Result<Vec<f64>> {
    // gets topic predictions
    let mut topic_predictions: Vec<Vec<f64>> = build::language_model::get_topic_predictions(texts,&build::FRAUD_INDICATORS)?;

    let sentiment_predictions = build::sentiment::get_sentiments(texts);
    // add custom features
    // add sentiment predictions
    for i in 0..texts.len() {
        topic_predictions[i].append(&mut get_features(texts[i].to_owned()));
        topic_predictions[i].push(sentiment_predictions[i]);
    }

    build::classification::predict(&topic_predictions)
}


