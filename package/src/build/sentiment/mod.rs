use rust_bert::pipelines::sentiment::SentimentModel;
use std::sync::Arc;
use std::sync::Mutex;
use std::fs;

lazy_static::lazy_static! {
        static ref SENTIMENT_CLASSIFIER: Arc<Mutex<SentimentModel>> = Arc::new(Mutex::new(SentimentModel::new(Default::default()).unwrap()));
    }

pub fn extract_sentiments(dataset: &Vec<(&str,&f64)>, path: Option<String>) -> anyhow::Result<Vec<f64>> {

    let mut list_sentiments: Vec<Vec<f64>> = Vec::new();
    let chunks = 256;

    let total_batches = dataset.len() / chunks;
    let mut completed_batches = 0;

    for batch in dataset.iter().map(|x| x.0).collect::<Vec<&str>>().chunks(chunks) {
        println!("\nProcessing batch {}/{}", completed_batches + 1, total_batches);
        let sentiments = get_sentiments(&batch);
        list_sentiments.push(sentiments);
        completed_batches += 1;

    }

    println!("Total batches processed: {}", total_batches);

    if let Some(ref path) = path {
        let json_string = serde_json::json!({"sentiments":&(list_sentiments.iter().flatten().collect::<Vec<&f64>>()),"dataset":&dataset}).to_string();
        fs::write(&path, &json_string).ok();
    }


    Ok(list_sentiments.into_iter().flatten().collect::<Vec<f64>>())
}


pub fn get_sentiments(texts: &[&str]) -> Vec<f64> {

    let sentiment_classifier = &SENTIMENT_CLASSIFIER.try_lock().unwrap();

    let sentiments = sentiment_classifier.predict(texts);
    sentiments.into_iter().map(|output| {
        if output.polarity == rust_bert::pipelines::sentiment::SentimentPolarity::Negative {
            output.score * -1.0
        } else {
            output.score
        }
    }).collect::<Vec<f64>>()

}

pub fn get_sentiment(text: &str) -> f64 {

    let output = &SENTIMENT_CLASSIFIER.try_lock().unwrap().predict(&[text])[0];
    if output.polarity ==  rust_bert::pipelines::sentiment::SentimentPolarity::Negative {
        output.score * -1.0
    }else{
        output.score
    }
}

pub fn load_texts_from_file(paths: &[&str]) -> anyhow::Result<(Vec<String>,Vec<f64>)> {
    let mut list_sentiment_classifier_prediction: Vec<serde_json::Value> = Vec::new();

    // Iterate through each path provided and load predictions.
    for path in paths {
        println!("Processing file: {}", path);

        // Try to read and parse the JSON file.
        let sentiment_classifier_prediction: serde_json::Value = match fs::read_to_string(format!("sentiment_extract_sentiments_{}",path)) {
            Ok(file) => {
                match serde_json::from_str(&file) {
                    Ok(res) => {
                        println!("Successfully read and parsed JSON for file: {}", path);
                        res
                    }
                    Err(err) => {
                        println!("Error parsing JSON for file: {}: {:?}", path, err);
                        Default::default()
                    }
                }
            }
            Err(err) => {
                println!("Error reading file: {}: {:?}", path, err);
                Default::default()
            }
        };

        list_sentiment_classifier_prediction.push(sentiment_classifier_prediction);
    }
    let (x_dataset, y_dataset): (Vec<String>, Vec<f64>) = list_sentiment_classifier_prediction
        .iter()
        .flat_map(|sentiment_classifier_prediction| { sentiment_classifier_prediction["dataset"]
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
        .filter(|(text, _)| text != "empty")
        })
        .unzip();

    println!("Final x_dataset and y_dataset both contain {} entries.", y_dataset.len());
    Ok((x_dataset,y_dataset))

}

pub fn load_sentiments_from_file(paths: &[&str]) -> anyhow::Result<(Vec<f64>,Vec<f64>)> {
    let mut list_sentiment_classifier_prediction: Vec<serde_json::Value> = Vec::new();

    // Iterate through each path provided and load predictions.
    for path in paths {
        println!("Processing file: {}", path);

        // Try to read and parse the JSON file.
        let sentiment_classifier_prediction: serde_json::Value = match fs::read_to_string(format!("sentiment_extract_sentiments_{}",path)) {
            Ok(file) => {
                match serde_json::from_str(&file) {
                    Ok(res) => {
                        println!("Successfully read and parsed JSON for file: {}", path);
                        res
                    }
                    Err(err) => {
                        println!("Error parsing JSON for file: {}: {:?}", path, err);
                        Default::default()
                    }
                }
            }
            Err(err) => {
                println!("Error reading file: {}: {:?}", path, err);
                Default::default()
            }
        };

        list_sentiment_classifier_prediction.push(sentiment_classifier_prediction);
    }

    let (x_dataset, y_dataset): (Vec<f64>, Vec<f64>) = list_sentiment_classifier_prediction
        .iter()
        .flat_map(|sentiment_classifier_prediction| {
            sentiment_classifier_prediction["sentiments"]
                .as_array()
                .unwrap()
                .iter()
                .map(|sentiment| sentiment.as_f64().unwrap())
                .zip(
                    sentiment_classifier_prediction["dataset"]
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
                        .filter(|(text, _)| text != "empty").map(|(text, label)| label),
                )
        })
        .unzip();

    println!("Final x_dataset and y_dataset both contain {} entries.", y_dataset.len());
    Ok((x_dataset,y_dataset))

}
