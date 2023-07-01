use std::fs;
use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
use rust_bert::pipelines::sequence_classification::Label;

use crate::build::feature_engineering::get_features;

use std::sync::Arc;
use std::sync::Mutex;

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
    )?;
    Ok(output.iter().map(|x| x.iter().map(|y| y.score).collect::<Vec<f64>>()).collect())
}


pub fn extract_topics(dataset: &Vec<(&str,&bool)>, topics: &[&str], path: Option<String>) -> anyhow::Result<Vec<Vec<Label>>> {

    let sequence_classification_model = SEQUENCE_CLASSIFICATION_MODEL.try_lock().unwrap();

    let mut list_outputs: Vec<Vec<Vec<Label>>> = Vec::new();

    let chunks = 16;
    let total_batches = dataset.len() / chunks;
    let mut completed_batches = 0;

    for batch in dataset.chunks(chunks) {

        println!("\nProcessing batch {}/{}", completed_batches + 1, total_batches);

        let output: Vec<Vec<Label>>= sequence_classification_model.predict_multilabel(
            &batch.iter().map(|x| x.0).collect::<Vec<&str>>()[..],
            topics,
            Some(Box::new(|label: &str| {
                format!("This example is about {}.", label)
            })),
            128,
        )?;
        list_outputs.push(output);
        completed_batches += 1;

        if let Some(ref path) = path {
            let features: Vec<Vec<f64>> = list_outputs
                .iter()
                .flatten()
                .map(|x| x.iter().map(|y| y.score).collect::<Vec<f64>>()).collect();
            let json_string = serde_json::json!({"predictions":&features,"dataset":dataset, "topics":topics}).to_string();

            fs::write(&path, &json_string).ok();
        }
    }

    println!("Total batches processed: {}", total_batches);

    let outputs: Vec<Vec<Label>> = list_outputs.into_iter().flatten().collect();

    Ok(outputs)
}

pub fn extract_topic_pairs(dataset: &Vec<(&str,&bool)>, topic_pairs: &[[&str;2];9], path: Option<String>) -> anyhow::Result<Vec<Vec<Label>>> {

    let sequence_classification_model = SEQUENCE_CLASSIFICATION_MODEL.try_lock().unwrap();

    let mut list_outputs: Vec<Vec<Vec<Label>>> = Vec::new();

    let chunks = 256;

    let total_batches = dataset.len() / chunks;
    let mut completed_batches = 0;


    for batch in dataset.iter().map(|x| x.0).collect::<Vec<&str>>().chunks(chunks) {

        println!("\nProcessing batch {}/{}", completed_batches + 1, total_batches);

        let mut output: Vec<Vec<Label>> = Vec::new();

        for topics in topic_pairs {

            print!(".");
            let mut output_for_pair: Vec<Vec<Label>>= sequence_classification_model.predict_multilabel(
                &batch,
                &topics,
                None, /*Some(Box::new(|label: &str| {
                    format!("This example is about {}.", label)
                }))*/
                128,
            )?;
            for i in 0..output.len(){
                output[i].append(&mut output_for_pair[i]);
            }
        }

        list_outputs.push(output);
        completed_batches += 1;

    }

    println!("Total batches processed: {}", total_batches);


    if let Some(ref path) = path {
        let features: Vec<Vec<f64>> = list_outputs
            .iter()
            .flatten()
            .map(|x| x.iter().map(|y| y.score).collect::<Vec<f64>>()).collect();
        let json_string = serde_json::json!({"predictions":&features,"dataset":dataset, "topics":topic_pairs}).to_string();

        fs::write(&path, &json_string).ok();
    }


    let outputs: Vec<Vec<Label>> = list_outputs.into_iter().flatten().collect();

    Ok(outputs)
}

// This function takes a list of file paths as input, loads the predicted values and labels
// from each file, creates a dataset by combining the predicted values and labels,
// and returns a tuple containing the dataset and the corresponding labels.

pub fn load_topics_from_file(paths: &[&str]) -> anyhow::Result<(Vec<Vec<f64>>, Vec<f64>)> {

    // Initialize an empty vector to store the predicted values for each path.
    let mut list_sequence_classification_multi_label_prediction: Vec<serde_json::Value> = Vec::new();

    // Iterate through each path provided.
    for path in paths {
        // Read the contents of a file that is expected to be present in the directory
        // named "language_model_extract_topics_<path>".
        let sequence_classification_multi_label_prediction: serde_json::Value = match fs::read_to_string(format!("language_model_extract_topics_{}", path)) {
            // If the file exists and its contents can be parsed as JSON, parse the JSON value
            // and append it to the list_sequence_classification_multi_label_prediction vector.
            Ok(file) => {
                match serde_json::from_str(&file) {
                    Ok(res) => {
                        res
                    }
                    // If parsing the JSON value fails, print the error and append a default value
                    // to the list_sequence_classification_multi_label_prediction vector.
                    Err(err) => {
                        println!("{:?}", err);
                        Default::default()
                    }
                }
            }
            // If the file cannot be read, print the error and append a default value
            // to the list_sequence_classification_multi_label_prediction vector.
            Err(err) => {
                println!("{:?}", err);
                Default::default()
            }
        };

        // Append the sequence_classification_multi_label_prediction value to the vector.
        list_sequence_classification_multi_label_prediction.push(sequence_classification_multi_label_prediction);
    }

    // Initialize two empty vectors to store the predicted values and labels.
    let mut predictions: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<(String, bool)> = Vec::new();

    // Iterate through each element of the list_sequence_classification_multi_label_prediction vector.
    for sequence_classification_multi_label_prediction in list_sequence_classification_multi_label_prediction {
        // Extract the predicted values and append them to the predictions vector.
        for list_p in sequence_classification_multi_label_prediction["predictions"].as_array().unwrap() {
            predictions.push(list_p.as_array().unwrap().into_iter().map(|x| x.as_f64().unwrap()).collect());
        }
        // Extract the labels and append them to the labels vector as tuples.
        for each in sequence_classification_multi_label_prediction["dataset"].as_array().unwrap() {
            let entry = each.as_array().unwrap();
            labels.push((entry[0].as_str().unwrap().to_string(), entry[1].as_bool().unwrap()));
        }
    }

    // Print the lengths of the predictions and labels vectors and assert that they are equal.
    println!("len of predictions: {}", predictions.len());
    println!("len of labels: {}", labels.len());
    assert_eq!(predictions.len(), labels.len());

    // Create a dataset by combining the predicted values and labels and filtering out
    // any labels with a string value of "empty".
    let mut dataset: Vec<(&Vec<f64>, &(String, bool))> = Vec::new();
    for i in 0..predictions.len() {
        if labels[i].0 != "empty" {
            dataset.push((&predictions[i], &labels[i]));
        }
    }

    // Print the length of the dataset.
    println!("len of dataset: {}",dataset.len());

    // Initialize empty vectors to hold the new features and labels.
    let mut x_dataset: Vec<Vec<f64>> = Vec::new();
    let mut y_dataset: Vec<f64> = Vec::new();

    // Loop through each item in the dataset.
    for each in &dataset {
        // Create a new list by cloning the existing features and appending custom features.
        let mut new_list = each.0.clone();
        new_list.append(&mut get_features(each.1.0.to_owned()));

        // Add the new features to the x_dataset vector and add the corresponding label to the y_dataset vector.
        x_dataset.push(new_list);
        y_dataset.push(if each.1.1 { 1.0 } else { 0.0 });
    }

    // Check that the x_dataset and y_dataset vectors have the same length.
    assert_eq!(x_dataset.len(),y_dataset.len());

    // Print the length of the x_dataset and y_dataset vectors.
    println!("len of x_dataset / y_dataset: {}",y_dataset.len());

    // Return the x_dataset and y_dataset vectors as a tuple wrapped in an Ok Result.
    Ok((x_dataset,y_dataset))
}
