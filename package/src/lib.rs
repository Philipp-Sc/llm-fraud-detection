pub mod build;
pub mod cache;

pub mod service;

use importance::score::Model;
use crate::build::classification::deep_learning::NNMockModel;
use crate::build::feature_engineering::get_features;
use crate::build::language_model::get_n_best_fraud_indicators;

pub fn fraud_probabilities(texts: &[&str]/*, topics: &[&str]*/) ->  anyhow::Result<Vec<f64>> {

    let (custom_features, topics, latent_variables) = (false, false, true);

    let topic_selection = get_n_best_fraud_indicators(30usize,&"feature_importance_random_forest_topics_only.json".to_string());
    let topics_str = topic_selection.iter().map(|x| x.as_str()).collect::<Vec<&str>>();


    let topics_dataset: Vec<Vec<f64>> = build::language_model::get_topic_predictions(texts,&topics_str[..])?;
    assert_eq!(texts.len(), topics_dataset.len());
    let sentiment_dataset = build::sentiment::get_sentiments(texts);
    assert_eq!(topics_dataset.len(), sentiment_dataset.len());


    let mut input: Vec<Vec<f64>> = Vec::with_capacity(texts.len());

    for i in 0..texts.len() {

        let text = texts[i].to_owned();

        let mut tmp = Vec::new();
        tmp.append(&mut get_features(&text, topics_dataset[i].clone(), sentiment_dataset[i].clone(), custom_features, topics, latent_variables));
        input.push(tmp);

    }



    build::classification::predict(&input)
}


