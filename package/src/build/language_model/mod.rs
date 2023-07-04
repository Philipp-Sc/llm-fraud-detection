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

pub fn get_labels() -> Vec<String> {
    vec![
        ["Clickbait, suspected spam, fake news, sensationalism, hype", "Authentic, verified news/information"],
        ["Aggressive marketing, advertising, selling, promotion, authoritative, commanding", "Informative content, unbiased information"],
        ["Call to immediate action", "No urgency or pressure to take action, passive suggestion"],
        ["Suspicious, questionable, dubious", "Trustworthy, credible, reliable"],
        ["Untrustworthy, not to be trusted, unreliable source, blacklisted", "Reputable source"],
        ["Misleading or deceptive information: The product advertisement made false claims about the benefits of the product.", "Accurate, transparent information"],
        ["Of importance, significant, crucial", "Insignificant, inconsequential"],
        ["Giveaway, tokens, airdrops, rewards, gratis, claim now", "No incentives or rewards provided"],
        ["To hide illegal activity", " Legal, lawful activity"],
        ["Exaggeration or hyperbole", "Factual, restrained language"],
        ["Sensationalism in headlines", "Balanced, informative headlines"],
        ["Bias or slant", "Objective, unbiased reporting"],
        ["Editorial or opinion pieces", "Fact-based reporting"],
        ["Unverified or unverified content", "Fact-checking or verification"],
        ["Sponsored content or native advertising", "Independent, non-sponsored content"],
        ["User-generated content", "Professional journalism or organization-created content"],
        ["Comparing reputation, bias, credibility", "News sources or media outlets"],
        ["Irresponsible consumption and ecological degradation", "Sustainable practices and environmental impact"],
        ["Harassment/threatening", "Constructive communication"],
        ["Violence", "Peaceful behavior"],
        ["Sexual", "Non-sexual in nature"],
        ["Hate", "Expressing kindness and acceptance"],
        ["Self-harm", "Promoting well-being and self-care"],
        ["Sexual/minors", "Content appropriate for all ages"],
        ["Hate/threatening", "Positive and supportive communication"],
        ["Violence/graphic", "Non-violent and non-graphic"],
        ["Self-harm/intent", "Encouraging positive intentions"],
        ["Self-harm/instructions", "Promoting safety and well-being"],
    ].into_iter().flatten().map(|x| x.to_string()).collect()
}

pub fn get_topic_predictions(batch: &[&str], topics: &[&str])  -> anyhow::Result<Vec<Vec<f64>>> {

        let sequence_classification_model = SEQUENCE_CLASSIFICATION_MODEL.try_lock().unwrap();

        let mut output: Vec<Vec<Label>> = Vec::new();
        for _ in 0..batch.len() {
            output.push(Vec::new());
        }

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

        let mut output_for_pair: Vec<Vec<Label>>= sequence_classification_model.predict_multilabel(
            &batch,
            &topics,
            None, /*Some(Box::new(|label: &str| {
                format!("This example is about {}.", label)
            }))*/
            128,
        )?;
        print!(".");
        for i in 0..output.len(){
            output[i].append(&mut output_for_pair[i]);
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
