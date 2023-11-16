use std::fs;
use std::fs::File;
use serde_json::Value;
use std::io::BufReader;


use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
use rust_bert::pipelines::sequence_classification::Label;


use std::sync::Arc;
use std::sync::Mutex; 
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsModel, SentenceEmbeddingsModelType};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder; 

use std::process::Command;
use std::io::{self};


lazy_static::lazy_static! {
        //    Set-up model
        static ref SEQUENCE_CLASSIFICATION_MODEL: Arc<Mutex<ZeroShotClassificationModel>> = Arc::new(Mutex::new(ZeroShotClassificationModel::new(Default::default()).unwrap()));
        static ref SENTENCE_EMBEDDINGS_MODEL: Arc<Mutex<SentenceEmbeddingsModel>> = Arc::new(Mutex::new(
            SentenceEmbeddingsBuilder::remote(
                SentenceEmbeddingsModelType::BertBaseNliMeanTokens
            )
            .create_model().unwrap()
        ));

    }


pub fn get_embeddings(batch: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {

    let sentence_embeddings_model = SENTENCE_EMBEDDINGS_MODEL.try_lock().unwrap();

    let output: Vec<Vec<f32>> = sentence_embeddings_model.encode(&batch)?;

    Ok(output)

}

pub fn get_topic_predictions(batch: &[&str], topics: &[&str])  -> anyhow::Result<Vec<Vec<f32>>> {

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

    Ok(output.iter().map(|x| x.iter().map(|y| y.score as f32).collect::<Vec<f32>>()).collect())
}


pub fn extract_topics(dataset: &Vec<(&str,&f32)>, topics: &[&str], path: Option<String>) -> anyhow::Result<Vec<Vec<Label>>> {

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
        let features: Vec<Vec<f32>> = list_outputs
            .iter()
            .flatten()
            .map(|x| x.iter().map(|y| y.score as f32).collect::<Vec<f32>>()).collect();
        let json_string = serde_json::json!({"predictions":&features,"dataset":dataset, "topics":topics}).to_string();

        fs::write(&path, &json_string).ok();
    }


    let outputs: Vec<Vec<Label>> = list_outputs.into_iter().flatten().collect();

    Ok(outputs)
}


pub fn extract_embeddings(dataset: &Vec<(&str,&f64)>, path: Option<String>) -> anyhow::Result<Vec<Vec<f32>>> {

    let sentence_embeddings_model = SENTENCE_EMBEDDINGS_MODEL.try_lock().unwrap();

    let mut list_outputs: Vec<Vec<f32>> = Vec::new();

    let chunks = 512;

    let total_batches = dataset.len() / chunks;
    let mut completed_batches = 0;


    for batch in dataset.iter().map(|x| x.0).collect::<Vec<&str>>().chunks(chunks) {

        println!("\nProcessing batch {}/{}", completed_batches + 1, total_batches);


        let mut output: Vec<Vec<f32>> = sentence_embeddings_model.encode(&batch)?;


        list_outputs.append(&mut output);
        completed_batches += 1;

    }

    println!("Total batches processed: {}", total_batches);


    if let Some(ref path) = path {
        let json_string = serde_json::json!({"embeddings":&list_outputs,"dataset":dataset}).to_string();
        fs::write(&path, &json_string).ok();
    }

    Ok(list_outputs)
}


pub fn huggingface_transformers_extract_topics(dataset: &Vec<(&str,&f32)>, topics: &[&str], path: Option<String>) -> anyhow::Result<Vec<Vec<f32>>> {

    let mut outputs: Vec<Vec<f32>> = Vec::new();


    for (i,each) in dataset.iter().map(|x| x.0).enumerate().collect::<Vec<(usize,&str)>>(){

        println!("\nProcessing index {}", i);
        outputs.push(huggingface_transformers_predict_multilabel(&topics[..], each)?);
    }

    if let Some(ref path) = path { 
        let json_string = serde_json::json!({"predictions":&outputs,"dataset":dataset, "topics":topics}).to_string();
        fs::write(&path, &json_string).ok();
    }
    
    Ok(outputs)
}

pub fn huggingface_transformers_predict_multilabel(topics: &[&str], text: &str) -> Result<Vec<f32>, io::Error> {
    let output = Command::new("/usr/workspace/extract_topics.sh")
        .args(&[&topics.join(","), text])
        .stderr(std::process::Stdio::null())
        .spawn()?
        .wait_with_output()?;

    if output.status.success() {
        let output_str = match String::from_utf8(output.stdout) {
            Ok(s) => s,
            Err(e) => return Err(io::Error::new(io::ErrorKind::Other, format!("Failed to parse embeddings: {}", e))),
        };
        let values: Result<Vec<f32>, _> = output_str
            .split_whitespace()
            .map(|s| s.parse())
            .collect();

        match values {
            Ok(embeddings) => Ok(embeddings),
            Err(e) => Err(io::Error::new(io::ErrorKind::Other, format!("Failed to parse embeddings: {}", e))),
        }
    } else {
        Err(io::Error::new(io::ErrorKind::Other, format!("Command execution failed {:?}", output)))
    }
}


// This function takes a list of file paths as input, loads the predicted values and labels
// from each file, creates a dataset by combining the predicted values and labels,
// and returns a tuple containing the dataset and the corresponding labels.

pub fn load_topics_from_file(paths: &[&str], topics: &Vec<String>) -> anyhow::Result<(Vec<Vec<f32>>, Vec<f32>)> {

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


    let (x_dataset, y_dataset): (Vec<Vec<f32>>, Vec<f32>) = list_sequence_classification_multi_label_prediction
        .iter()
        .flat_map(|sequence_classification_multi_label_prediction| {
            let available_topics =  sequence_classification_multi_label_prediction["topics"]
                .as_array()
                .unwrap()
                .iter()
                .map(|topic| topic.as_str().unwrap().to_string())
                .map(|topic| topics.iter().any(|s| s == &topic))
                .collect::<Vec<bool>>();

            sequence_classification_multi_label_prediction["predictions"]
                .as_array()
                .unwrap()
                .iter()
                .map(move |topic_val| topic_val.as_array().unwrap().into_iter().map(|x| x.as_f64().unwrap() as f32).enumerate().filter_map(|(i,t)| if available_topics[i] {Some(t)}else{None}).collect::<Vec<f32>>())
                .zip(
                    sequence_classification_multi_label_prediction["dataset"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|entry| {
                            let entry = entry.as_array().unwrap();
                            (
                                entry[0].as_str().unwrap().to_string(),
                                entry[1].as_f64().unwrap() as f32,
                            )
                        })
                        .filter(|(text, _)| text != "empty"),
                ).map(|(topics,(_text,label))|{
                    (topics,label)
                })
        })
        .unzip();

    println!("Final x_dataset and y_dataset both contain {} entries.", y_dataset.len());

    Ok((x_dataset,y_dataset))
}


pub fn get_n_best_fraud_indicators(n: usize, path: &String) -> Vec<String> {
    let file = File::open(&path).expect("Failed to open file");
    let reader = BufReader::new(file);

    // Parse the JSON from the file
    let json: Value = serde_json::from_reader(reader).expect("Failed to parse JSON");

    // Extract the feature importance array
    let feature_importance = json["feature_importance"].as_array().unwrap_or_else(|| {
        panic!("Failed to extract feature importance array");
    });

    // Load the strings into a Vec<String>
    let strings: Vec<String> = feature_importance
        .iter()
        .map(|entry| entry[1].as_str().unwrap_or_else(|| {
            panic!("Failed to extract string from entry");
        }))
        .map(|string| string.to_string())
        .take(n)
        .collect();
    strings
}
