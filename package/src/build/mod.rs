



pub mod data;
pub mod classification;
pub mod language_model;

use data::read_datasets;


pub async fn create_training_data(dataset_paths: Vec<&str>,topics_output_path: &str) -> anyhow::Result<()> {

    let dataset: Vec<(String,f32)> = read_datasets(&dataset_paths)?;


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


    let dataset_view: Vec<(&str,&f32)> = dataset.iter().map(|(text,label)| (text.as_str(),label)).collect();


    //language_model::embeddings::extract_embeddings(&dataset_view, Some(format!("language_model_extract_embeddings_{}",topics_output_path)))?;
    let mut topic_selection: Vec<String> = language_model::zero_shot_classification::get_n_best_fraud_indicators(100usize,&"feature_importance_random_forest_topics_only.json".to_string());
    topic_selection.push("spam".to_string());
    topic_selection.push("not spam".to_string());
    topic_selection.push("fraud".to_string());
    topic_selection.push("not fraud".to_string());
    topic_selection.push("phishing".to_string());
    topic_selection.push("not phishing".to_string());
    topic_selection.push("scam".to_string());
    topic_selection.push("not scam".to_string());
    topic_selection.push("manipulative".to_string());
    topic_selection.push("not manipulative".to_string());
    topic_selection.push("urgent".to_string());
    topic_selection.push("not urgent".to_string());
    topic_selection.push("benevolent".to_string());
    topic_selection.push("not benevolent".to_string());
    topic_selection.push("narcissistic".to_string());
    topic_selection.push("not narcissistic".to_string());
    topic_selection.push("machiavellian".to_string());
    topic_selection.push("not machiavellian".to_string());
    topic_selection.push("psychopathic".to_string());
    topic_selection.push("not psychopathic".to_string());




    language_model::zero_shot_classification::huggingface_transformers_extract_topics(&dataset_view, &topic_selection.iter().map(|x| x.as_str()).collect::<Vec<&str>>()[..], Some(format!("language_model_extract_topics_{}",topics_output_path))).await?;
    Ok(())
}
