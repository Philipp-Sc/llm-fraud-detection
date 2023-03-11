use rust_bert::pipelines::sentiment::SentimentModel;
use std::sync::Arc;
use std::sync::Mutex;
use std::fs;

lazy_static::lazy_static! {
        static ref SENTIMENT_CLASSIFIER: Arc<Mutex<SentimentModel>> = Arc::new(Mutex::new(SentimentModel::new(Default::default()).unwrap()));
    }

pub fn extract_sentiments(dataset: &Vec<(String,bool)>, path: Option<String>) -> anyhow::Result<Vec<f64>> {

    let mut list_sentiments: Vec<Vec<f64>> = Vec::new();
    let chunks = 200;
    let mut count: usize = 0;
    for batch in dataset.chunks(chunks) {

        let data = &batch.iter().map(|x| x.0.as_str().as_ref()).collect::<Vec<&str>>();
        let sentiments = get_sentiments(data);
        count += sentiments.len();
        println!("sentiments generated: {}",count);

        list_sentiments.push(sentiments);

        if let Some(ref path) = path {
            let json_string = serde_json::json!({"sentiments":&(list_sentiments.iter().flatten().collect::<Vec<&f64>>()),"dataset":&dataset}).to_string();
            fs::write(&path, &json_string).ok();
        }

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

// This function loads sentiment data from files and returns two vectors, one for predictions and one for labels.
pub fn load_sentiments_from_file(paths: &[&str]) -> anyhow::Result<(Vec<f64>,Vec<f64>)> {
    // Initialize an empty vector to store the predicted values for each path.
    let mut list_sentiment_classifier_prediction: Vec<serde_json::Value> = Vec::new();
    // Iterate through each path provided.

    for path in paths {
        // Read the contents of a file that is expected to be present in the directory
        // named "sentiment_extract_sentiments_<path>".
    let sentiment_classifier_prediction: serde_json::Value = match fs::read_to_string(format!("sentiment_extract_sentiments_{}",path)) {
        // If the file exists and its contents can be parsed as JSON, parse the JSON value
        // and append it to the list_sentiment_classifier_prediction vector.
        Ok(file) => {
            match serde_json::from_str(&file) {
                Ok(res) => {
                    res
                }
                // If parsing the JSON value fails, print the error and append a default value
                // to the list_sentiment_classifier_prediction vector.
                Err(err) => {
                    println!("{:?}", err);
                    Default::default()
                }
            }
        }
        // If the file cannot be read, print the error and append a default value
        // to the list_sentiment_classifier_prediction vector.
        Err(err) => {
            println!("{:?}", err);
            Default::default()
        }
    };
        // Append the sentiment_classifier_prediction value to the vector.
    list_sentiment_classifier_prediction.push(sentiment_classifier_prediction);
    }
    // Initialize two empty vectors to store the predicted values and labels.
    let mut predictions: Vec<f64> = Vec::new();
    let mut labels: Vec<(String,bool)> = Vec::new();

    for sentiment_classifier_prediction in list_sentiment_classifier_prediction {

    for sentiment in sentiment_classifier_prediction["sentiments"].as_array().unwrap() {
        predictions.push(sentiment.as_f64().unwrap());
    }
    for each in sentiment_classifier_prediction["dataset"].as_array().unwrap() {
        let entry = each.as_array().unwrap();
        labels.push((entry[0].as_str().unwrap().to_string(),entry[1].as_bool().unwrap()));
    }

    }
    println!("len of predictions: {}",predictions.len());
    println!("len of labels: {}",labels.len());
    assert_eq!(predictions.len(),labels.len());

    let mut dataset: Vec<(&f64,&(String,bool))> = Vec::new();
    for i in 0..predictions.len() {
        if labels[i].0 != "empty" {
            dataset.push((&predictions[i],&labels[i]));
        }
    }

    println!("len of dataset: {}",dataset.len());

    let mut x_dataset: Vec<f64> = Vec::new();
    let mut y_dataset: Vec<f64> = Vec::new();
    for each in &dataset {
            x_dataset.push(*each.0);
            y_dataset.push(if each.1.1 { 1.0 } else { 0.0 });
    }

    assert_eq!(x_dataset.len(),y_dataset.len());
    println!("len of x_dataset / y_dataset: {}",y_dataset.len());

    Ok((x_dataset,y_dataset))
}
