use std::fs::File;
use rand::seq::SliceRandom;
use rand::thread_rng;
use super::language_model::embeddings::*;
use super::language_model::zero_shot_classification::*;
use super::language_model::sentiment::*;


use linkify::LinkFinder;
use std::collections::HashSet;


use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

lazy_static::lazy_static! {

        static ref RE_UPPER_CASE_WORD: regex::Regex = regex::Regex::new(r"\b[A-Z]+\b").unwrap();
        
        static ref RE_NON_STANDARD: regex::Regex = regex::Regex::new(r"[^\w\s]").unwrap();
        
        static ref RE_PUNCTUATION: regex::Regex = regex::Regex::new(r"[[:punct:]]+").unwrap();

        static ref RE_EMOJI: regex::Regex = regex::Regex::new(r"\p{Emoji}").unwrap();

        static ref LINK_FINDER: LinkFinder = get_link_finder();

     }


const MIN_TEXT_LENGTH: usize = 20;

fn read_dataset(path: &str) -> anyhow::Result<Vec<(String,f32)>> {

    let (text_index,label_index) = if path.contains("enronSpamSubset") {
        (2,3)
    }else if path.contains("smsspamcollection"){
        (1,0)
    }else if path.contains("lingSpam") || path.contains("completeSpamAssassin") || path.contains("governance_proposal_spam_ham"){
        (1,2)
    }else{ // youtubeSpamCollection
        (0,1)
    };

    let mut training_data: Vec<(String,f32)> = Vec::new();

    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result?;
        let record_text = record.get(text_index);
        let record_label = record.get(label_index);
        if let (Some(text),Some(label)) = (record_text,record_label) {
                if let Ok(parsed_value) = label.parse::<f64>() {
                    if text.len() >= MIN_TEXT_LENGTH {
                        training_data.push((text.to_string(),parsed_value as f32));
                    }
                }
        }
    }
    Ok(training_data)
}


pub fn read_datasets(dataset_paths: &[&str]) -> anyhow::Result<Vec<(String,f32)>> {
    let mut dataset: Vec<(String,f32)> = read_dataset(dataset_paths.get(0).ok_or(anyhow::anyhow!("Error: dataset_paths is empty!'"))?)?;
    if dataset_paths.len() > 1 {
        for i in 1..dataset_paths.len() {
            dataset.append(&mut read_dataset(dataset_paths[i])?);
        }
    }
    Ok(dataset)
}




pub fn generate_shuffled_idx(paths: &[&str]) -> anyhow::Result<Vec<usize>> {
    let (x_dataset, _) = load_topics_from_file(paths,&vec![])?;

    // create an index array
    let mut idx: Vec<usize> = (0..x_dataset.len()).collect();

    // shuffle the index array using the thread_rng() random number generator
    idx.shuffle(&mut thread_rng());

    Ok(idx)
}

pub fn read_datasets_and_shuffle(paths: &[&str], shuffled_idx: &Vec<usize>) -> anyhow::Result<Vec<(String,f32)>> {
    let dataset: Vec<(String, f32)> = read_datasets(paths)?;

    let mut dataset_shuffled = Vec::with_capacity(dataset.len());

    for i in shuffled_idx {
        dataset_shuffled.push(dataset[*i].clone());
    }

    Ok(dataset_shuffled)
}


pub enum DatasetKind{
    Embedding,
    ZeroShotClassification(Vec<String>),
    Sentiment,
    OtherFeatures,
}

pub fn create_dataset(kind: DatasetKind, paths: &[&str], shuffled_idx: &Vec<usize>) -> anyhow::Result<(Vec<Vec<f32>>,Vec<f32>)> 
{

    let (x_dataset, y_dataset): (Vec<Vec<f32>>, Vec<f32>) = match kind {
        DatasetKind::Embedding => {
            load_embeddings_from_file(paths)?
        }
        DatasetKind::ZeroShotClassification(topic_selection) => {
            load_topics_from_file(paths, &topic_selection)?
        }
        DatasetKind::Sentiment => {
            load_sentiments_from_file(paths)?
        }
        DatasetKind::OtherFeatures => {
            let (text_dataset, y) = load_texts_from_file(paths)?;
            (text_dataset.into_par_iter().map(|x| get_custom_features(&x)).collect(), y)
        }
    };

    // use the shuffled index array to reorder both datasets at once
    let mut x_dataset_shuffled = Vec::with_capacity(x_dataset.len());
    let mut y_dataset_shuffled = Vec::with_capacity(y_dataset.len());

    for &i in shuffled_idx {
        x_dataset_shuffled.push(x_dataset[i].clone());
        y_dataset_shuffled.push(y_dataset[i].clone());
    }

    Ok((x_dataset_shuffled, y_dataset_shuffled))
}

pub fn merge_datasets(
    x_dataset1: Vec<Vec<f32>>,
    x_dataset2: Vec<Vec<f32>>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    if x_dataset1.len() != x_dataset2.len() {
        anyhow::bail!("Datasets have different lengths and cannot be merged.");
    }
    let x_dataset_combined: Vec<Vec<f32>> = x_dataset1
        .into_iter()
        .zip(x_dataset2.into_iter())
        .map(|(v1, v2)| v1.into_iter().chain(v2.into_iter()).collect())
        .collect();
    Ok(x_dataset_combined)
}


pub fn split_vector<T>(vector: &[T], ratio: f64) -> (&[T], &[T]) {
    let split_index = (vector.len() as f64 * ratio) as usize;
    vector.split_at(split_index)
}


fn extract_links(text: &String) -> Vec<String> {

    let links = LINK_FINDER.links(&text);
    let mut output: Vec<String> = Vec::new();
    for link in links  {
        let s = link.as_str().to_string();
        if s.parse::<f64>().is_err() && s.chars().count() >= 8 && s.chars().any(|c| c.is_alphabetic()) {
           output.push(s);
        }       
    }
    // Convert the vector to a HashSet
    let set: HashSet<String> = output.into_iter().collect();
    // Convert the HashSet back to a vector
    set.into_iter().collect()
}

fn get_link_finder() -> LinkFinder {
    let mut finder = LinkFinder::new();
    finder.url_must_have_scheme(false);
    finder
}

fn get_custom_features(text: &String) -> Vec<f32> {
    let mut features = Vec::new();
    let word_count = words_count::count(text);
    features.push(word_count.words as f32); // 8
    features.push(word_count.characters as f32); // 3
    features.push(word_count.whitespaces as f32); // 5
    features.push(word_count.cjk as f32);

    features.push(extract_links(text).into_iter().count() as f32); // 9
    features.push(RE_UPPER_CASE_WORD.captures_iter(text).count() as f32);
    features.push(RE_NON_STANDARD.captures_iter(text).count() as f32); // 7
    features.push(RE_PUNCTUATION.captures_iter(text).count() as f32); // 4
    features.push(RE_EMOJI.captures_iter(text).count() as f32); // 6

    features
}
