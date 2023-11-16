use std::fs;





use std::process::Command;
use std::io::{self};


pub fn llama_cpp_embedding(text: &str) -> Result<Vec<f32>, io::Error> {
    let input = text.to_string().chars().take(8192*4).collect::<String>();
    let output = Command::new("/usr/llama.cpp/embedding")
        .args(&["-m", "./zephyr-7b-alpha.Q8_0.gguf", "--log-disable", "--ctx-size","8192","--mlock","--threads","14", "-p", &input])
        .stderr(std::process::Stdio::null())
        .output()?;

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
        Err(io::Error::new(io::ErrorKind::Other, "Command execution failed"))
    }
}


pub fn extract_embeddings(dataset: &Vec<(&str,&f32)>, path: Option<String>) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut list_outputs: Vec<Vec<f32>> = Vec::new();


    for text in dataset.iter().map(|x| x.0) {
        match llama_cpp_embedding(text).map_err(|original_error| {
            anyhow::anyhow!(format!("An error occurred during embeddings generation: {:?}",original_error))
        }) {
            Ok(output) => list_outputs.push(output),
            Err(err) => {
               println!("Error: {:?}", &err); 
               list_outputs.push(Vec::new());
            }
        };
        println!("count: {}",list_outputs.len());
    }
    if let Some(ref path) = path {
        let json_string = serde_json::json!({"embeddings":&list_outputs,"dataset":dataset}).to_string();
        fs::write(&path, &json_string).ok();
    }

    Ok(list_outputs)

}

pub fn load_embeddings_from_file(paths: &[&str]) -> anyhow::Result<(Vec<Vec<f32>>, Vec<f32>)> {

    // Initialize an empty vector to store the predicted values for each path.
    let mut list_sentence_embeddings: Vec<serde_json::Value> = Vec::new();

    // Iterate through each path provided.
    for path in paths {
        println!("Processing file: {}", path);

        // Read the contents of a file that is expected to be present in the directory
        // named "language_model_extract_topics_<path>".
        let sentence_embeddings: serde_json::Value = match fs::read_to_string(format!("language_model_extract_embeddings_{}", path)) {
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
        list_sentence_embeddings.push(sentence_embeddings);
    }

    let (x_dataset, y_dataset): (Vec<Vec<f32>>, Vec<f32>) = list_sentence_embeddings
        .iter()
        .flat_map(|sentence_embeddings| {
            sentence_embeddings["embeddings"]
                .as_array()
                .unwrap()
                .iter()
                .map(|topics| topics.as_array().unwrap().into_iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>())
                .zip(
                    sentence_embeddings["dataset"]
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
