use std::fs;
use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
use rust_bert::pipelines::sequence_classification::Label;

use crate::build::feature_engineering::get_features;

use std::sync::Arc;
use std::sync::Mutex;
use rand::seq::SliceRandom;
use rand::thread_rng;

lazy_static::lazy_static! {
        //    Set-up model
        static ref SEQUENCE_CLASSIFICATION_MODEL: Arc<Mutex<ZeroShotClassificationModel>> = Arc::new(Mutex::new(ZeroShotClassificationModel::new(Default::default()).unwrap()));
    }

pub fn get_topic_predictions(input: &[&str], topics: &[&str])  -> anyhow::Result<Vec<Vec<f64>>> {

    let output: Vec<Vec<Label>>= SEQUENCE_CLASSIFICATION_MODEL.try_lock().unwrap().predict_multilabel(
        input,
        topics,
        Some(Box::new(|label: &str| {
            format!("This example is about {}.", label)
        })),
        128,
    );
    Ok(output.iter().map(|x| x.iter().map(|y| y.score).collect::<Vec<f64>>()).collect())
}


pub fn extract_topics(dataset: &Vec<(String,bool)>, topics: &[&str], path: Option<String>) -> anyhow::Result<Vec<Vec<Label>>> {

    let sequence_classification_model = SEQUENCE_CLASSIFICATION_MODEL.try_lock().unwrap();

    let mut list_outputs: Vec<Vec<Vec<Label>>> = Vec::new();

    let mut count: usize = 0;
    let chunks = 1;
    for batch in dataset.chunks(chunks) {

        let output: Vec<Vec<Label>>= sequence_classification_model.predict_multilabel(
            &batch.iter().map(|x| x.0.as_str()).collect::<Vec<&str>>()[..],
            topics,
            Some(Box::new(|label: &str| {
                format!("This example is about {}.", label)
            })),
            128,
        );
        println!("{}",count);
        count += output.len();
        list_outputs.push(output);

        if let Some(ref path) = path {
            let outputs: Vec<&Vec<Label>> = list_outputs.iter().flatten().collect();
            let features: Vec<Vec<f64>> = outputs.iter().map(|x| x.iter().map(|y| y.score).collect::<Vec<f64>>()).collect();
            let json_string = serde_json::json!({"predictions":&features,"dataset":dataset, "topics":topics}).to_string();

            fs::write(&path, &json_string).ok();
        }
    }
    let outputs: Vec<Vec<Label>> = list_outputs.into_iter().flatten().collect();

    Ok(outputs)
}

pub fn load_topics_from_file(paths: &[&str]) -> anyhow::Result<(Vec<Vec<f64>>,Vec<f64>)> {

    let mut list_sequence_classification_multi_label_prediction: Vec<serde_json::Value> = Vec::new();
    for path in paths { 
    let sequence_classification_multi_label_prediction: serde_json::Value = match fs::read_to_string(format!("language_model_extract_topics_{}",path)) {
        Ok(file) => {
            match serde_json::from_str(&file) {
                Ok(res) => {
                    res
                }
                Err(err) => {
                    println!("{:?}", err);
                    Default::default()
                }
            }
        }
        Err(err) => {
            println!("{:?}", err);
            Default::default()
        }
    };
    list_sequence_classification_multi_label_prediction.push(sequence_classification_multi_label_prediction);
    }

    let mut predictions: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<(String,bool)> = Vec::new();

    for sequence_classification_multi_label_prediction in list_sequence_classification_multi_label_prediction {
    for list_p in sequence_classification_multi_label_prediction["predictions"].as_array().unwrap(){
        predictions.push(list_p.as_array().unwrap().into_iter().map(|x| x.as_f64().unwrap()).collect());
    }
    for each in sequence_classification_multi_label_prediction["dataset"].as_array().unwrap() {
        let entry = each.as_array().unwrap();
        labels.push((entry[0].as_str().unwrap().to_string(),entry[1].as_bool().unwrap()));
    }

    }
    println!("len of predictions: {}",predictions.len());
    println!("len of labels: {}",labels.len());
    assert_eq!(predictions.len(),labels.len());

    let mut dataset: Vec<(&Vec<f64>,&(String,bool))> = Vec::new();
    for i in 0..predictions.len() {
        if labels[i].0 != "empty" {
            dataset.push((&predictions[i],&labels[i]));
        }
    }

    dataset.shuffle(&mut thread_rng());

    println!("len of dataset: {}",dataset.len());


    let mut x_dataset: Vec<Vec<f64>> = Vec::new();
    let mut y_dataset: Vec<f64> = Vec::new();
    for each in &dataset {
        // adding custom features
        let mut new_list = each.0.clone();
        new_list.append(&mut get_features(each.1.0.to_owned()));

        x_dataset.push(new_list);
        y_dataset.push(if each.1.1 { 1.0 } else { 0.0 });
    }

    assert_eq!(x_dataset.len(),y_dataset.len());
    println!("len of x_dataset / y_dataset: {}",y_dataset.len());

    Ok((x_dataset,y_dataset))
}
