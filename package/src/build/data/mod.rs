use std::fs::File;
use rand::seq::SliceRandom;
use rand::thread_rng;


fn read_dataset(path: &str) -> anyhow::Result<Vec<(String,bool)>> {

    let (text_index,label_index) = if path.contains("enronSpamSubset") {
        (2,3)
    }else if path.contains("smsspamcollection"){
        (1,0)
    }else if path.contains("lingSpam") || path.contains("completeSpamAssassin") || path.contains("governance_proposal_spam_ham"){
        (1,2)
    }else{ // youtubeSpamCollection
        (0,1)
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


pub fn read_datasets(dataset_paths: &[&str]) -> anyhow::Result<Vec<(String,bool)>> {
    let mut dataset: Vec<(String,bool)> = read_dataset(dataset_paths.get(0).ok_or(anyhow::anyhow!("Error: dataset_paths is empty!'"))?)?;
    if dataset_paths.len() > 1 {
        for i in 1..dataset_paths.len() {
            dataset.append(&mut read_dataset(dataset_paths[i])?);
        }
    }
    Ok(dataset)
}


