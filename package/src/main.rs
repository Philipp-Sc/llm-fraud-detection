use std::thread;
use std::time::Duration;

use rust_bert_fraud_detection_tools::service::spawn_rust_bert_fraud_detection_socket_service;
use std::env;
use rust_bert_fraud_detection_socket_ipc::ipc::client_send_rust_bert_fraud_detection_request;
use rust_bert_fraud_detection_tools::build::create_naive_bayes_model;

pub const SENTENCES: [&str;6] = [
    "Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];


// To just test the fraud detection:
//      sudo docker run -it --rm -v "$(pwd)/rustbert_cache":/usr/rustbert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release

// Start service container:
//      sudo docker run -d --rm -v "$(pwd)/rustbert_cache":/usr/rustbert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release start_service
//      (To later stop the service container)
//          sudo docker container ls
//          sudo docker stop CONTAINER_ID
// Run service test:
//      sudo docker run -it --rm -v "$(pwd)/rustbert_cache":/usr/rustbert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release test_service
fn main_00() -> anyhow::Result<()> {

    let args: Vec<String> = env::args().collect();
    println!("env::args().collect(): {:?}",args);

    if args.len() <= 1 {
        println!("{:?}", &SENTENCES);
        let fraud_probabilities: Vec<f64> = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
        println!("Predictions:\n{:?}", fraud_probabilities);
        println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
        Ok(())
    }else{
        match args[1].as_str() {
            "start_service" => {
                spawn_rust_bert_fraud_detection_socket_service("./tmp/rust_bert_fraud_detection_socket").join().unwrap();
                Ok(())
            },
            "test_service" => {
                let result = client_send_rust_bert_fraud_detection_request("./tmp/rust_bert_fraud_detection_socket",SENTENCES.iter().map(|x|x.to_string()).collect::<Vec<String>>())?;
                println!("{:?}",result);
                Ok(())
            }
            _ => {
                println!("invalid command");
                Ok(())
            }
        }
    }
}

/*_training*/
fn main_9999() -> anyhow::Result<()> {

    let training_data_paths = [
      "data_gen_v3_(enronSpamSubset).json",
      "data_gen_v3_(lingSpam).json",
      "data_gen_v3_(smsspamcollection).json",
      "data_gen_v3_(completeSpamAssassin).json",
      "data_gen_v3_(governance_proposal_spam_ham).json",
      "data_gen_v3_(governance_proposal_spam_ham).json"];
    
    rust_bert_fraud_detection_tools::build::create_classification_model(&training_data_paths)?;

    println!("test with training data");
    let test_data_paths = [
      "data_gen_v3_(enronSpamSubset).json",
      "data_gen_v3_(lingSpam).json",
      "data_gen_v3_(smsspamcollection).json",
      "data_gen_v3_(completeSpamAssassin).json",
      "data_gen_v3_(governance_proposal_spam_ham).json"];
    
    rust_bert_fraud_detection_tools::build::test_classification_model(&test_data_paths)?;

    let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}

fn main() -> anyhow::Result<()>{
    let paths= vec!["./dataset/youtubeSpamCollection.csv"];
    let test_paths= vec!["./dataset/smsspamcollection.csv"];
    create_naive_bayes_model(&paths,&test_paths)
}

/*_generate_sentiments_and_topics*/
fn main_888() -> anyhow::Result<()> {

    //let training_data_path = "data_gen_v3_(enronSpamSubset).json";
    //rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/enronSpamSubset.csv"],training_data_path)?;
    //let training_data_path = "data_gen_v3_(lingSpam).json";
    //rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/lingSpam.csv"],training_data_path)?;

    let training_data_path = "data_gen_v3_(youtubeSpamCollection).json";
    rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/youtubeSpamCollection.csv"],training_data_path)?;

/*    let training_data_path = "data_gen_v3_(smsspamcollection).json";
    rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/smsspamcollection.csv"],training_data_path)?;

    let training_data_path = "data_gen_v3_(completeSpamAssassin).json";
    rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/completeSpamAssassin.csv"],training_data_path)?;
*/


    //rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/completeSpamAssassin.csv","./dataset/lingSpam.csv","./dataset/enronSpamSubset.csv"],training_data_path)?;
    return Ok(());

}
