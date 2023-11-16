use std::fs;



pub fn load_texts_from_file(paths: &[&str]) -> anyhow::Result<(Vec<String>,Vec<f32>)> {
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
    let (x_dataset, y_dataset): (Vec<String>, Vec<f32>) = list_sentiment_classifier_prediction
        .iter()
        .flat_map(|sentiment_classifier_prediction| { sentiment_classifier_prediction["dataset"]
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
        .filter(|(text, _)| text != "empty")
        })
        .unzip();

    println!("Final x_dataset and y_dataset both contain {} entries.", y_dataset.len());
    Ok((x_dataset,y_dataset))

}

pub fn load_sentiments_from_file(paths: &[&str]) -> anyhow::Result<(Vec<Vec<f32>>,Vec<f32>)> {
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

    let (x_dataset, y_dataset): (Vec<Vec<f32>>, Vec<f32>) = list_sentiment_classifier_prediction
        .iter()
        .flat_map(|sentiment_classifier_prediction| {
            sentiment_classifier_prediction["sentiments"]
                .as_array()
                .unwrap()
                .iter()
                .map(|sentiment| vec![sentiment.as_f64().unwrap() as f32])
                .zip(
                    sentiment_classifier_prediction["dataset"]
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
                        .filter(|(text, _)| text != "empty").map(|(_text, label)| label),
                )
        })
        .unzip();

    println!("Final x_dataset and y_dataset both contain {} entries.", y_dataset.len());
    Ok((x_dataset,y_dataset))

}
