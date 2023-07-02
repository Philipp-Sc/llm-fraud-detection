use std::thread;
use std::time::Duration;

use rust_bert_fraud_detection_tools::service::spawn_rust_bert_fraud_detection_socket_service;
use std::env;
use rust_bert_fraud_detection_socket_ipc::ipc::client_send_rust_bert_fraud_detection_request;
use rust_bert_fraud_detection_tools::build::create_naive_bayes_model;
use rust_bert_fraud_detection_tools::build::data::split_vector;

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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        println!("No command specified.");
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "naive_bayes_train" => {naive_bayes_train();},
        "naive_bayes_predict" => {naive_bayes_predict();},
        "train_and_test_final_regression_model" => {train_and_test_final_regression_model();},
        "generate_feature_vectors" => {generate_feature_vectors();},
        "service" => {service();},
        _ => {panic!()}
    }

    Ok(())
}

fn service() -> anyhow::Result<()> {

    let mut args: Vec<String> = env::args().collect();
    args.reverse();
    args.pop();
    args.reverse();
    println!("env::args().collect(): {:?}",args);

    if args.len() <= 1 {
        println!("{:?}", &SENTENCES);
        let fraud_probabilities: Vec<f64> = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
        println!("Predictions:\n{:?}", fraud_probabilities);
        println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
        Ok(())
    }else{
        match args[1].as_str() {
            "start" => {
                spawn_rust_bert_fraud_detection_socket_service("./tmp/rust_bert_fraud_detection_socket").join().unwrap();
                Ok(())
            },
            "test" => {
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

fn train_and_test_final_regression_model() -> anyhow::Result<()> {

    let data_paths = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("data_gen_v4_({}).json",x)).collect::<Vec<String>>();
    let paths = data_paths.iter().map(|x| x.as_str()).collect::<Vec<&str>>();

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&paths[..])?;

    let (x_train, x_test) = split_vector(&x_dataset,0.7);
    let x_train = x_train.to_vec();
    let x_test = x_test.to_vec();
    let (y_train, y_test) = split_vector(&y_dataset,0.7);
    let y_train = y_train.to_vec();
    let y_test = y_test.to_vec();

    rust_bert_fraud_detection_tools::build::create_classification_model(&x_train,&y_train)?;
    rust_bert_fraud_detection_tools::build::test_classification_model(&x_test,&y_test)?;

    let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}


fn naive_bayes_predict() -> anyhow::Result<()>{
    let predictions = rust_bert_fraud_detection_tools::build::naive_bayes::categorical_nb_model_predict(SENTENCES.iter().map(|&s| s.to_string()).collect::<Vec<String>>())?;
    println!("Predictions:\n{:?}",predictions);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}


fn naive_bayes_train() -> anyhow::Result<()>{
    let paths= vec![
        "./dataset/enronSpamSubset.csv",
        "./dataset/lingSpam.csv",
        "./dataset/youtubeSpamCollection.csv",
        "./dataset/smsspamcollection.csv",
        "./dataset/completeSpamAssassin.csv",
        "./dataset/governance_proposal_spam_ham.csv"];
    let test_paths= vec![
        "./dataset/enronSpamSubset.csv",
        "./dataset/lingSpam.csv",
        "./dataset/youtubeSpamCollection.csv",
        "./dataset/smsspamcollection.csv",
        "./dataset/completeSpamAssassin.csv",
        "./dataset/governance_proposal_spam_ham.csv"];
    create_naive_bayes_model(&paths,&test_paths)
}

fn generate_feature_vectors() -> anyhow::Result<()> {

    // test governance spam ham
    // let training_data_path = "new_data_gen_v4_(governance_proposal_spam_likelihood).json";
    // rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/governance_proposal_spam_likelihood.csv"],training_data_path)?;


    let datasets = [
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"];

    for dataset in datasets {
        let training_data_path = format!("data_gen_v4_({}).json",dataset);
        let dataset_path = format!("./dataset/{}.csv",dataset);

        rust_bert_fraud_detection_tools::build::create_training_data(vec![&dataset_path],&training_data_path)?;
    }


    return Ok(());

}
