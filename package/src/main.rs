
pub const SENTENCES: [&str;6] = [
    "Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];


fn main() -> anyhow::Result<()> {

    let scam_probabilities: Vec<f64> = rust_scam_detection_tools::scam_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",scam_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}

fn main_training() -> anyhow::Result<()> {

    let training_data_paths = ["data_gen_v1_(enronSpamSubset).json","data_gen_v1.json"];
    
    // generate training data for the selected topics, also generates sentiments.
    //rust_scam_detection_tools::build::create_training_data(vec![/*"./dataset/completeSpamAssassin.csv","./dataset/lingSpam.csv",*/"./dataset/enronSpamSubset.csv"],training_data_path)?;
    //return Ok(());
    rust_scam_detection_tools::build::create_classification_model(&training_data_paths)?;

    println!("test with training data");    
    rust_scam_detection_tools::build::test_classification_model(&training_data_paths)?;
    //println!("test with new data");
    //rust_scam_detection_tools::build::test_classification_model(&["data_gen_v1.json"])?;

    let scam_probabilities = rust_scam_detection_tools::scam_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",scam_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}
