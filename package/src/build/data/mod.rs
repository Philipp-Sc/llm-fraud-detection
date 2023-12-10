use std::fs::File;

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


pub fn split_vector<T>(vector: &[T], ratio: f64) -> (&[T], &[T]) {
    let split_index = (vector.len() as f64 * ratio) as usize;
    vector.split_at(split_index)
}
