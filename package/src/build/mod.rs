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

pub const FRAUD_INDICATOR_PAIRS: [[&str;2];28] = [
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
];


// Note: Any additional topic increases the model inference time!

// https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset
// "./dataset/completeSpamAssassin.csv" 1896/4150
// "./dataset/lingSpam.csv" 433/2172
// "./dataset/enronSpamSubset.csv" 5000/5000
pub fn create_training_data(dataset_paths: Vec<&str>,topics_output_path: &str) -> anyhow::Result<()> {

    let mut dataset: Vec<(String,f64)> = read_datasets(&dataset_paths)?;


    let spam_count = dataset.iter().filter(|x| x.1 >= 0.5).count();
    let ham_count = dataset.iter().filter(|x| x.1 < 0.5).count();
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


    let dataset_view: Vec<(&str,&f64)> = dataset.iter().map(|(text,label)| (text.as_str(),label)).collect();


    //sentiment::extract_sentiments(&dataset_view,Some(format!("sentiment_extract_sentiments_{}",topics_output_path)))?;
    //language_model::extract_topics(&dataset_view,&FRAUD_INDICATORS,Some(format!("language_model_extract_topics_{}",topics_output_path)))?;
    language_model::extract_topic_pairs(&dataset_view,&FRAUD_INDICATOR_PAIRS,Some(format!("language_model_extract_topics_{}",topics_output_path)))?;

    Ok(())
}

pub fn create_classification_model(x_dataset_shuffled: &Vec<Vec<f64>>, y_dataset_shuffled: &Vec<f64>) -> anyhow::Result<()> {

    classification::update_regression_model(&x_dataset_shuffled, &y_dataset_shuffled)?;

    Ok(())
}
pub fn test_classification_model(x_dataset_shuffled: &Vec<Vec<f64>>, y_dataset_shuffled: &Vec<f64>) -> anyhow::Result<()> {

    classification::test_regression_model(&x_dataset_shuffled, &y_dataset_shuffled)?;

    Ok(())
}

pub fn create_naive_bayes_model(paths: &[&str], test_paths: &[&str]) -> anyhow::Result<()> {

    let dataset: Vec<(String,f64)> = read_datasets(paths)?;

    let x_dataset= dataset.iter().map(|x| x.0.to_string()).collect::<Vec<String>>();
    let y_dataset= dataset.into_iter().map(|x| if x.1 < 0.5 {0} else {1}).collect::<Vec<i32>>();

    let dataset: Vec<(String,f64)> = read_datasets(test_paths)?;

    let test_x_dataset= dataset.iter().map(|x| x.0.to_string()).collect::<Vec<String>>();
    let test_y_dataset= dataset.into_iter().map(|x| if x.1 < 0.5 {0} else {1}).collect::<Vec<i32>>();

    naive_bayes::update_naive_bayes_model(&x_dataset,&y_dataset,&test_x_dataset,&test_y_dataset)?;
    naive_bayes::update_categorical_naive_bayes_model(&x_dataset,&y_dataset,&test_x_dataset,&test_y_dataset)?;
    //classification::test_linear_regression_model(&x_dataset,&y_dataset)?;

    Ok(())
}