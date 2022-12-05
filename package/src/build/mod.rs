use std::fs::File;
pub mod classification;
pub mod language_model;
pub mod feature_engineering;
pub mod sentiment;

pub const FRAUD_INDICATORS: [&str;7] = [
    "(clickbait, suspected spam, fake news)", // 2122 // 1663 // 1697 // 1726
    "(untrustworthy, not to be trusted, unreliable source, blacklisted)", // 191 // 215 // 238 // 260
    "(call to immediate action)", // 160 // 211 // 224 // 244
    "(aggressive marketing, advertising, selling, promotion, authoritative, commanding)", // 175 // 202 // 207 // 229
    "(giveaway, tokens, airdrops, rewards, gratis, claim now)", // 145 // 205 // 189 // 209
    "(written by a human, verified account, domain expert, open source, wikipedia)", // 156 // 169 // 174 // 211
    "(of importance, significant, crucial)", // 156 // 203
];
// TODO: Model can be improved by finding more/better uncorrelated fraud indicators.
// Note: Any additional topic increases the model inference time!

fn read_dataset(path: &str) -> anyhow::Result<Vec<(String,bool)>> {

    let (text_index,label_index) = if !path.contains("enron") {
        if path.contains("proposals") {
            (0,1)
        }else {
            (1,2)
        }
    }else{
        (2,3)
    };
    let mut training_data: Vec<(String,bool)> = Vec::new();

    let file = File::open(path)?; // 1896/4150
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result?;
        let record_text = record.get(text_index); // (2,3) enronSpamSubset; (1,2) lingSpam / completeSpamAssassin;
        let record_label = record.get(label_index);
        if let (Some(text),Some(label)) = (record_text,record_label) {
            let mut tuple: (String,bool) = (text.to_owned(),false);
            if label == "1" {
                tuple.1 = true;
            }
            if tuple.0 != "empty" && tuple.0 != "" {
                training_data.push(tuple);
            }
        }
    }
    Ok(training_data)
}


// https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset
// "./dataset/completeSpamAssassin.csv" 1896/4150
// "./dataset/lingSpam.csv" 433/2172
// "./dataset/enronSpamSubset.csv" 5000/5000
pub fn create_training_data(dataset_paths: Vec<&str>,topics_output_path: &str) -> anyhow::Result<()> {

    let mut dataset: Vec<(String,bool)> = read_dataset(dataset_paths.get(0).ok_or(anyhow::anyhow!("Error: dataset_paths is empty!'"))?)?;
    if dataset_paths.len() > 1 {
        for i in 1..dataset_paths.len() {
            dataset.append(&mut read_dataset(dataset_paths[i])?);
        }
    }

    println!("count spam: {:?}", dataset.iter().filter(|x| x.1).count());
    println!("count ham: {:?}", dataset.iter().filter(|x| !x.1).count());

    let indices_spam: Vec<usize> = dataset.iter().enumerate().filter(|(i,x)| x.1).map(|(i,_)| i).collect();
    let indices_spam_ham: Vec<usize> = dataset.iter().enumerate().filter(|(i,x)| !x.1)/*.take(indices_spam.len())*/.map(|(i,_)| i).collect();
    let dataset: Vec<(String,bool)> = vec![
        indices_spam.iter().map(|&i| &dataset[i]).collect::<Vec<&(String,bool)>>(),
        indices_spam_ham.iter().map(|&i| &dataset[i]).collect::<Vec<&(String,bool)>>()
    ].into_iter().flatten().map(|x| x.clone()).collect();


    println!("len dataset: {:?}", dataset.iter().count());


    sentiment::extract_sentiments(&dataset,Some(format!("sentiment_extract_sentiments_{}",topics_output_path)))?;
    language_model::extract_topics(&dataset,&FRAUD_INDICATORS,Some(format!("language_model_extract_topics_{}",topics_output_path)))?;

    Ok(())
}

pub fn create_classification_model(paths: &[&str]) -> anyhow::Result<()> {

    let (mut x_dataset, y_dataset): (Vec<Vec<f64>>,Vec<f64>) = language_model::load_topics_from_file(paths)?;
    let (x_dataset_sentiment,_) = sentiment::load_sentiments_from_file(paths)?;
    assert_eq!(x_dataset.len(),x_dataset_sentiment.len());
    for i in 0..x_dataset.len() {
        x_dataset[i].push(x_dataset_sentiment[i]);
    }

    classification::update_linear_regression_model(&x_dataset,&y_dataset)?;
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

