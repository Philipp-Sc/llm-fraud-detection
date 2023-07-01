use std::fs::File;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub mod data;
pub mod classification;
pub mod language_model;
pub mod feature_engineering;
pub mod sentiment;
pub mod naive_bayes;


use data::read_datasets;

pub const FRAUD_INDICATORS: [&str;10] = [
"(clickbait, suspected spam, fake news)", 
"(aggressive marketing, advertising, selling, promotion, authoritative, commanding)", 
"(call to immediate action)", 
"(suspicious, questionable, dubious)", 
"(untrustworthy, not to be trusted, unreliable source, blacklisted)", 
"Misleading or deceptive information: The product advertisement made false claims about the benefits of the product.", 
"(of importance, significant, crucial)",  
"(clickbait, sensationalism, hype)", 
"(giveaway, tokens, airdrops, rewards, gratis, claim now)",  
"(to hide illegal activity)" ];

pub const FRAUD_INDICATOR_PAIRS: [[&str;2];9] = [
     ["Clickbait, suspected spam, fake news, sensationalism, hype", "Authentic, verified news/information"],
     ["Aggressive marketing, advertising, selling, promotion, authoritative, commanding", "Informative content, unbiased information"],
     ["Call to immediate action", "No urgency or pressure to take action, passive suggestion"],
     ["Suspicious, questionable, dubious", "Trustworthy, credible, reliable"],
     ["Untrustworthy, not to be trusted, unreliable source, blacklisted", "Reputable source"],
     ["Misleading or deceptive information: The product advertisement made false claims about the benefits of the product.", "Accurate, transparent information"],
     ["Of importance, significant, crucial", "Insignificant, inconsequential"],
     ["Giveaway, tokens, airdrops, rewards, gratis, claim now", "No incentives or rewards provided"],
     ["To hide illegal activity", " Legal, lawful activity"]];

// Note: Any additional topic increases the model inference time!

// https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset
// "./dataset/completeSpamAssassin.csv" 1896/4150
// "./dataset/lingSpam.csv" 433/2172
// "./dataset/enronSpamSubset.csv" 5000/5000
pub fn create_training_data(dataset_paths: Vec<&str>,topics_output_path: &str) -> anyhow::Result<()> {

    let mut dataset: Vec<(String,bool)> = read_datasets(&dataset_paths)?;


    let spam_count = dataset.iter().filter(|x| x.1).count();
    let ham_count = dataset.iter().filter(|x| !x.1).count();
    let total_count = dataset.len();

    let spam_percentage = (spam_count as f64 / total_count as f64) * 100.0;
    let ham_percentage = (ham_count as f64 / total_count as f64) * 100.0;

    println!("Spam count: {}", spam_count);
    println!("Ham count: {}", ham_count);
    println!("===================");
    println!("Total count: {}", total_count);
    println!("===================");
    println!("Spam percentage: {:.2}%", spam_percentage);
    println!("Ham percentage: {:.2}%\n", ham_percentage);

    /*
    let indices_spam: Vec<usize> = dataset.iter().enumerate().filter(|(i,x)| x.1).map(|(i,_)| i).collect();
    let indices_spam_ham: Vec<usize> = dataset.iter().enumerate().filter(|(i,x)| !x.1)/*.take(indices_spam.len())*/.map(|(i,_)| i).collect();
    let dataset: Vec<(&str,bool)> = vec![
        indices_spam.iter().map(|&i| (dataset[i].0.as_str(),dataset[i].1)).collect::<Vec<(&str,bool)>>(),
        indices_spam_ham.iter().map(|&i| (dataset[i].0.as_str(),dataset[i].1)).collect::<Vec<(&str,bool)>>()
    ].into_iter().flatten().map(|x| x.clone()).collect();
    */


    let dataset_view: Vec<(&str,&bool)> = dataset.iter().map(|(text,label)| (text.as_str(),label)).collect();


    //sentiment::extract_sentiments(&dataset,Some(format!("sentiment_extract_sentiments_{}",topics_output_path)))?;
    //language_model::extract_topics(&dataset,&FRAUD_INDICATORS,Some(format!("language_model_extract_topics_{}",topics_output_path)))?;
    language_model::extract_topic_pairs(&dataset_view,&FRAUD_INDICATOR_PAIRS,Some(format!("language_model_extract_topics_{}",topics_output_path)))?;

    Ok(())
}

pub fn create_classification_model(paths: &[&str]) -> anyhow::Result<()> {

    let (mut x_dataset, y_dataset): (Vec<Vec<f64>>,Vec<f64>) = language_model::load_topics_from_file(paths)?;
    let (x_dataset_sentiment,_) = sentiment::load_sentiments_from_file(paths)?;
    assert_eq!(x_dataset.len(),x_dataset_sentiment.len());
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

    classification::update_linear_regression_model(&x_dataset_shuffled,&y_dataset_shuffled)?;
    //classification::test_linear_regression_model(&x_dataset,&y_dataset)?;

    Ok(())
}
pub fn test_classification_model(paths: &[&str]) -> anyhow::Result<()> {

    let (mut x_dataset, y_dataset): (Vec<Vec<f64>>,Vec<f64>) = language_model::load_topics_from_file(paths)?;
    let (x_dataset_sentiment,_) = sentiment::load_sentiments_from_file(paths)?;
    assert_eq!(x_dataset.len(),x_dataset_sentiment.len());
    for i in 0..x_dataset.len() {
        x_dataset[i].push(x_dataset_sentiment[i]);
    }

    classification::test_linear_regression_model(&x_dataset,&y_dataset)?;

    Ok(())
}

pub fn create_naive_bayes_model(paths: &[&str], test_paths: &[&str]) -> anyhow::Result<()> {

    let dataset: Vec<(String,bool)> = read_datasets(paths)?;

    let x_dataset= dataset.iter().map(|x| x.0.to_string()).collect::<Vec<String>>();
    let y_dataset= dataset.into_iter().map(|x| if x.1 {1} else {0}).collect::<Vec<i32>>();

    let dataset: Vec<(String,bool)> = read_datasets(test_paths)?;

    let test_x_dataset= dataset.iter().map(|x| x.0.to_string()).collect::<Vec<String>>();
    let test_y_dataset= dataset.into_iter().map(|x| if x.1 {1} else {0}).collect::<Vec<i32>>();

    naive_bayes::update_naive_bayes_model(x_dataset.clone(),y_dataset.clone(),test_x_dataset.clone(),test_y_dataset.clone())?;
    naive_bayes::update_categorical_naive_bayes_model(x_dataset.clone(),y_dataset.clone(),test_x_dataset.clone(),test_y_dataset.clone())?;
    //classification::test_linear_regression_model(&x_dataset,&y_dataset)?;

    Ok(())
}