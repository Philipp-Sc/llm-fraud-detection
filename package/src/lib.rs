pub mod build;
pub mod cache;

pub mod service;

use importance::score::Model;
use crate::build::classification::deep_learning::MockModel;
use crate::build::feature_engineering::get_features;

pub fn fraud_probabilities(texts: &[&str]/*, topics: &[&str]*/) ->  anyhow::Result<Vec<f64>> {



    let mut topic_predictions: Vec<Vec<f64>> = build::language_model::get_topic_predictions(texts,&build::FRAUD_INDICATORS)?;
    let sentiment_predictions = build::sentiment::get_sentiments(texts);


    let mut input: Vec<Vec<f64>> = Vec::with_capacity(texts.len());

    for i in 0..texts.len() {
        let mut tmp = Vec::new();

        let text = texts[i].to_owned();
        tmp.append(&mut get_features(&text));

        let model = MockModel{ label: "./NeuralNet.bin".to_string()};
        let prediction: f64 = model.predict(&vec![topic_predictions[i].clone()])[0];

        tmp.append(&mut topic_predictions[i]);
        tmp.push(prediction);

        tmp.push(sentiment_predictions[i]);

        input.push(tmp);

    }



    build::classification::predict(&topic_predictions)
}


