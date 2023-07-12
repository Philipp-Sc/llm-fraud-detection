use std::fs::File;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::build::{language_model, sentiment};
use crate::build::feature_engineering::get_features;
use crate::build::classification::deep_learning::NNMockModel;
use importance::score::Model;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

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




pub fn generate_shuffled_idx(paths: &[&str]) -> anyhow::Result<Vec<usize>> {
    // let (x_dataset, _): (Vec<Vec<f64>>, Vec<f64>) = language_model::load_topics_from_file(paths)?;
    let (x_dataset, _) = sentiment::load_sentiments_from_file(paths)?;

    // create an index array
    let mut idx: Vec<usize> = (0..x_dataset.len()).collect();

    // shuffle the index array using the thread_rng() random number generator
    idx.shuffle(&mut thread_rng());

    Ok(idx)
}

pub fn read_datasets_and_shuffle(paths: &[&str], shuffled_idx: &Vec<usize>) -> anyhow::Result<Vec<(String,f64)>> {
    let dataset: Vec<(String, f64)> = read_datasets(paths)?;

    let mut dataset_shuffled = Vec::with_capacity(dataset.len());

    for i in shuffled_idx {
        dataset_shuffled.push(dataset[*i].clone());
    }

    Ok(dataset_shuffled)
}


pub fn create_dataset(paths: &[&str], shuffled_idx: &Vec<usize>, topic_selection: &Vec<String>, embeddings: bool, custom_features: bool, topics: bool, latent_variables: bool) -> anyhow::Result<(Vec<Vec<f64>>,Vec<f64>)> {

    let (mut text_dataset, y_dataset) = sentiment::load_texts_from_file(paths)?;
    assert_eq!(shuffled_idx.len(),text_dataset.len());

    let (mut topics_dataset, y_data): (Vec<Vec<f64>>, Vec<f64>) = language_model::load_topics_from_file(paths, topic_selection)?;
    assert_eq!(text_dataset.len(), topics_dataset.len());
    assert_eq!(y_dataset, y_data);
    let (mut sentiment_dataset, y_data): (Vec<f64>, Vec<f64>) = sentiment::load_sentiments_from_file(paths)?;
    assert_eq!(topics_dataset.len(), sentiment_dataset.len());
    assert_eq!(y_dataset, y_data);

    let (mut embedding_dataset, y_data): (Vec<Vec<f64>>, Vec<f64>) = language_model::load_embeddings_from_file(paths)?;
    assert_eq!(embedding_dataset.len(), sentiment_dataset.len());
    assert_eq!(y_dataset, y_data);

    let x_dataset: Vec<_> = (0..text_dataset.len())
        .into_iter()
        .step_by(500)
        .flat_map(|batch_start| {
            let batch_end = std::cmp::min(batch_start + 500, text_dataset.len());
            (batch_start..batch_end)
                .into_par_iter()
                .map(|i| {
                    get_features(
                        &text_dataset[i],
                        embedding_dataset[i].clone(),
                        topics_dataset[i].clone(),
                        sentiment_dataset[i].clone(),
                        embeddings,
                        custom_features,
                        topics,
                        latent_variables,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();



    // create an index array
    //let mut idx: Vec<usize> = (0..x_dataset.len()).collect();

    // shuffle the index array using the thread_rng() random number generator
    // idx.shuffle(&mut thread_rng());

    // use the shuffled index array to reorder both datasets at once
    let mut x_dataset_shuffled = Vec::with_capacity(x_dataset.len());
    let mut y_dataset_shuffled = Vec::with_capacity(y_dataset.len());

    for &i in shuffled_idx {
        x_dataset_shuffled.push(x_dataset[i].clone());
        y_dataset_shuffled.push(y_dataset[i].clone());
    }

    Ok((x_dataset_shuffled, y_dataset_shuffled))
}

pub fn split_vector<T>(vector: &[T], ratio: f64) -> (&[T], &[T]) {
    let split_index = (vector.len() as f64 * ratio) as usize;
    vector.split_at(split_index)
}
