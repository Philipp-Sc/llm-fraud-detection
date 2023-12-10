use std::fs;
use std::fs::File;
use std::process::Command;
use std::io::{self, Read};


pub fn llama_cpp_embedding(text: &str) -> Result<Vec<f32>, io::Error> {
    let input = text.to_string().chars().take(8192*4).collect::<String>();
    let output = Command::new("/usr/llama.cpp/embedding")
        .args(&["-m", "./una-cybertron-7b-v2-bf16.Q8_0.gguf", "--log-disable", "--ctx-size","8192", "-p", &input])
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


pub fn extract_embeddings(dataset: &Vec<(&str,&f32)>) -> anyhow::Result<Vec<Vec<f32>>> {
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

    Ok(list_outputs)

}

pub fn load_llama_cpp_embeddings_from_file(path: &str) -> anyhow::Result<(Vec<Vec<f32>>, Vec<f32>)> {
    // Read the contents of the file
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    // Parse the JSON content
    let json_data: Vec<serde_json::Value> = serde_json::from_str(&contents)?;

    // Initialize vectors to store embeddings and labels
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    // Iterate over each entry in the JSON data
    for entry in json_data {
        // Extract the "embedding" field
        if let Some(embedding) = entry["embedding"].as_array() {
            // Convert the embedding to a vector of f32
            let embedding_vec: Vec<f32> = embedding
                .iter()
                .filter_map(|value| value.as_f64().map(|v| v as f32))
                .collect();

            // Push the embedding to the embeddings vector
            embeddings.push(embedding_vec);
        }

        // Extract the label from the "entry" field (assuming it's an array)
        if let Some(entry_array) = entry["entry"].as_array() {
            // Assuming the label is the second item in the array
            if let Some(label_value) = entry_array.get(1) {
                // Convert the label to f32 and push it to the labels vector
                if let Some(label) = label_value.as_f64() {
                    labels.push(label as f32);
                }
            }
        }
    }

    // Return the result as a tuple
    Ok((embeddings, labels))
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
