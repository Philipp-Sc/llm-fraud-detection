use std::fs;
use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
use rust_bert::pipelines::sequence_classification::Label;

use crate::build::feature_engineering::get_features;

use std::sync::Arc;
use std::sync::Mutex;
use crate::build::FRAUD_INDICATORS;

lazy_static::lazy_static! {
        //    Set-up model
        static ref SEQUENCE_CLASSIFICATION_MODEL: Arc<Mutex<ZeroShotClassificationModel>> = Arc::new(Mutex::new(ZeroShotClassificationModel::new(Default::default()).unwrap()));
    }

pub fn get_labels() -> Vec<String> {
    FRAUD_INDICATORS.into_iter().map(|x| x.to_string()).collect()
}

pub fn get_topic_predictions(batch: &[&str], topics: &[&str])  -> anyhow::Result<Vec<Vec<f64>>> {

        let sequence_classification_model = SEQUENCE_CLASSIFICATION_MODEL.try_lock().unwrap();

        let mut output: Vec<Vec<Label>> = Vec::new();
        for _ in 0..batch.len() {
            output.push(Vec::new());
        }

        for topic in topics {
            let mut output_for_pair: Vec<Vec<Label>> = sequence_classification_model.predict_multilabel(
                &batch,
                &[*topic],
                None, /*Some(Box::new(|label: &str| {
                    format!("This example is about {}.", label)
                }))*/
                128,
            )?;
            for i in 0..output.len() {
                output[i].append(&mut output_for_pair[i]);
            }
        }

    Ok(output.iter().map(|x| x.iter().map(|y| y.score).collect::<Vec<f64>>()).collect())
}


pub fn extract_topics(dataset: &Vec<(&str,&f64)>, topics: &[&str], path: Option<String>) -> anyhow::Result<Vec<Vec<Label>>> {

    let sequence_classification_model = SEQUENCE_CLASSIFICATION_MODEL.try_lock().unwrap();

    let mut list_outputs: Vec<Vec<Vec<Label>>> = Vec::new();

    let chunks = 512;

    let total_batches = dataset.len() / chunks;
    let mut completed_batches = 0;


    for batch in dataset.iter().map(|x| x.0).collect::<Vec<&str>>().chunks(chunks) {

        println!("\nProcessing batch {}/{}", completed_batches + 1, total_batches);

        let mut output: Vec<Vec<Label>> = Vec::new();
        for _ in 0..batch.len() {
            output.push(Vec::new());
        }

        for topic in topics {
            let mut output_for_pair: Vec<Vec<Label>> = sequence_classification_model.predict_multilabel(
                &batch,
                &[*topic],
                None, /*Some(Box::new(|label: &str| {
                format!("This example is about {}.", label)
            }))*/
                128,
            )?;
            print!(".");
            for i in 0..output.len() {
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
        let json_string = serde_json::json!({"predictions":&features,"dataset":dataset, "topics":topics}).to_string();

        fs::write(&path, &json_string).ok();
    }


    let outputs: Vec<Vec<Label>> = list_outputs.into_iter().flatten().collect();

    Ok(outputs)
}

// This function takes a list of file paths as input, loads the predicted values and labels
// from each file, creates a dataset by combining the predicted values and labels,
// and returns a tuple containing the dataset and the corresponding labels.

pub fn load_topics_from_file_and_add_hard_coded_features(paths: &[&str]) -> anyhow::Result<(Vec<Vec<f64>>, Vec<f64>)> {

    // Initialize an empty vector to store the predicted values for each path.
    let mut list_sequence_classification_multi_label_prediction: Vec<serde_json::Value> = Vec::new();

    // Iterate through each path provided.
    for path in paths {
        println!("Processing file: {}", path);

        // Read the contents of a file that is expected to be present in the directory
        // named "language_model_extract_topics_<path>".
        let sequence_classification_multi_label_prediction: serde_json::Value = match fs::read_to_string(format!("language_model_extract_topics_{}", path)) {
            // If the file exists and its contents can be parsed as JSON, parse the JSON value
            // and append it to the list_sequence_classification_multi_label_prediction vector.
            Ok(file) => {
                match serde_json::from_str(&file) {
                    Ok(res) => {
                        println!("Successfully read and parsed JSON for file: {}", path);
                        res
                    }
                    // If parsing the JSON value fails, print the error and append a default value
                    // to the list_sequence_classification_multi_label_prediction vector.
                    Err(err) => {
                        println!("Error parsing JSON for file: {}: {:?}", path, err);
                        Default::default()
                    }
                }
            }
            // If the file cannot be read, print the error and append a default value
            // to the list_sequence_classification_multi_label_prediction vector.
            Err(err) => {
                println!("Error parsing JSON for file: {}: {:?}", path, err);
                Default::default()
            }
        };

        // Append the sequence_classification_multi_label_prediction value to the vector.
        list_sequence_classification_multi_label_prediction.push(sequence_classification_multi_label_prediction);
    }

    let (x_dataset, y_dataset): (Vec<Vec<f64>>, Vec<f64>) = list_sequence_classification_multi_label_prediction
        .iter()
        .flat_map(|sequence_classification_multi_label_prediction| {
            sequence_classification_multi_label_prediction["predictions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|topics| topics.as_array().unwrap().into_iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>())
                .zip(
                    sequence_classification_multi_label_prediction["dataset"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|entry| {
                            let entry = entry.as_array().unwrap();
                            (
                                entry[0].as_str().unwrap().to_string(),
                                entry[1].as_f64().unwrap(),
                            )
                        })
                        .filter(|(text, _)| text != "empty"),
                ).map(|(mut topics,(text,label))|{
                    topics.append(&mut get_features(text.to_owned()));
                    (topics,label)
                })
        })
        .unzip();

    println!("Final x_dataset and y_dataset both contain {} entries.", y_dataset.len());

    Ok((x_dataset,y_dataset))
}
