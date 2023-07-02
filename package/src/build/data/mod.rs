use std::fs::File;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::build::{language_model, sentiment};

const MIN_TEXT_LENGTH: usize = 20;

fn read_dataset(path: &str) -> anyhow::Result<Vec<(String,f64)>> {

    let (text_index,label_index) = if path.contains("enronSpamSubset") {
        (2,3)
    }else if path.contains("smsspamcollection"){
        (1,0)
    }else if path.contains("lingSpam") || path.contains("completeSpamAssassin") || path.contains("governance_proposal_spam_ham"){
        (1,2)
    }else{ // youtubeSpamCollection
        (0,1)
    };

    let mut training_data: Vec<(String,f64)> = Vec::new();

    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result?;
        let record_text = record.get(text_index);
        let record_label = record.get(label_index);
        if let (Some(text),Some(label)) = (record_text,record_label) {
                if let Ok(parsed_value) = label.parse::<f64>() {
                    if text.len() >= MIN_TEXT_LENGTH {
                        training_data.push((text.to_string(),parsed_value));
                    }
                }
        }
    }
    Ok(training_data)
}


pub fn read_datasets(dataset_paths: &[&str]) -> anyhow::Result<Vec<(String,f64)>> {
    let mut dataset: Vec<(String,f64)> = read_dataset(dataset_paths.get(0).ok_or(anyhow::anyhow!("Error: dataset_paths is empty!'"))?)?;
    if dataset_paths.len() > 1 {
        for i in 1..dataset_paths.len() {
            dataset.append(&mut read_dataset(dataset_paths[i])?);
        }
    }
    Ok(dataset)
}



pub fn create_dataset(paths: &[&str]) -> anyhow::Result<(Vec<Vec<f64>>,Vec<f64>)> {

    let (mut x_dataset, y_dataset): (Vec<Vec<f64>>, Vec<f64>) = language_model::load_topics_from_file(paths)?;
    let (x_dataset_sentiment, _) = sentiment::load_sentiments_from_file(paths)?;

    assert_eq!(x_dataset.len(), x_dataset_sentiment.len());
    for i in 0..x_dataset.len() {
        x_dataset[i].push(x_dataset_sentiment[i]);
    }

    // create an index array
    let mut idx: Vec<usize> = (0..x_dataset.len()).collect();

    // shuffle the index array using the thread_rng() random number generator
    idx.shuffle(&mut thread_rng());

    // use the shuffled index array to reorder both datasets at once
    let mut x_dataset_shuffled = Vec::with_capacity(x_dataset.len());
    let mut y_dataset_shuffled = Vec::with_capacity(y_dataset.len());

    for i in idx {
        x_dataset_shuffled.push(x_dataset[i].clone());
        y_dataset_shuffled.push(y_dataset[i].clone());
    }

    Ok((x_dataset_shuffled, y_dataset_shuffled))
}

pub fn split_vector<T>(vector: &[T], ratio: f64) -> (&[T], &[T]) {
    let split_index = (vector.len() as f64 * ratio) as usize;
    vector.split_at(split_index)
}