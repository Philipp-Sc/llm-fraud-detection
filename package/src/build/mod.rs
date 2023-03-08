use std::fs::File;
pub mod classification;
pub mod language_model;
pub mod feature_engineering;
pub mod sentiment;

pub const FRAUD_INDICATORS: [&str; 51] = [
    "(clickbait, suspected spam, fake news)",
    "(untrustworthy, not to be trusted, unreliable source, blacklisted)",
    "(call to immediate action)",
    "(aggressive marketing, advertising, selling, promotion, authoritative, commanding)",
    "(giveaway, tokens, airdrops, rewards, gratis, claim now)",
    "(written by a human, verified account, domain expert, open source, wikipedia)",
    "(of importance, significant, crucial)",
    "(misleading, deceptive, false, untruthful)",
    "(scam, fraudulent, illegitimate)",
    "(phishing, hacking, malware, ransomware)",
    "(suspicious, questionable, dubious)",
    "(bias, propaganda, spin, fake news)",
    "(spam, junk, unsolicited)",
    "(clickbait, sensationalism, hype)",
    "(unexpected, unusual behavior)",
    "(inconsistent, conflicting information)",
    "(red flags)",
    "(false endorsements, testimonials)",
    "(fake websites, domains)",
    "(fake profiles, accounts)",
    "(forged documents, signatures)",
    "(identity theft, impersonation)",
    "(money laundering, tax evasion)",
    "(bribery, corruption, influence peddling)",
    "(insider trading, market manipulation)",
    "(account takeover, unauthorized access)",
    "(denial of service attacks, cyber attacks)",
    "(bogus offers, scams)",
    "(fake reviews, ratings, feedback)",
    "Misleading or deceptive information: The product advertisement made false claims about the benefits of the product.",
    "Scams or fraudulent activity: The email was a scam asking for personal information and bank account details.",
    "Phishing, hacking, or other cyber attacks: The website asked for login information and then locked the user out of their account.",
    "Suspicious or questionable behavior: The account activity showed sudden, large transactions with no apparent reason.",
    "Bias, propaganda, or fake news: The news article was biased and contained false information.",
    "Spam or unsolicited content: The user received a large number of unsolicited emails from unknown senders.",
    "Clickbait or sensationalist headlines: The article had a sensationalist headline that didn't match the content of the article.",
    "Unexpected or unusual behavior: The user's account showed unusual patterns of activity, such as multiple login attempts from different locations.",
    "Inconsistent or conflicting information: The user provided different addresses and phone numbers in different transactions.",
    "Red flags or other indicators of potential fraud: The user received a large number of returned or declined transactions.",
    "False endorsements or testimonials: The product had fake endorsements from celebrities who had never actually used it.",
    "Fake websites or domains: The website looked like a legitimate company's website, but was actually a fake designed to trick users.",
    "Fake profiles or accounts: The social media profile was fake and used to impersonate a real person.",
    "Forged documents or signatures: The contract had a forged signature and was not valid.",
    "Identity theft or impersonation: The user's identity was stolen and used to open new accounts and make transactions.",
    "Money laundering or tax evasion: The company was involved in money laundering to hide illegal activity.",
    "Bribery, corruption, or influence peddling: The politician accepted bribes in exchange for favorable treatment.",
    "Insider trading or market manipulation: The company insider traded on non-public information to make a profit.",
    "Unauthorized access or account takeover: The hacker gained access to the user's account and made unauthorized transactions.",
    "Denial of service attacks or other cyber attacks: The website was attacked and became unavailable for users.",
    "Bogus offers or scams: The user received an offer for a free vacation that turned out to be a scam.",
    "Fake reviews, ratings, or feedback: The product had fake positive reviews that were not from real customers."];


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

