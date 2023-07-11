use std::thread;
use std::time::Duration;

use rust_bert_fraud_detection_tools::service::spawn_rust_bert_fraud_detection_socket_service;
use std::env;
use rust_bert_fraud_detection_socket_ipc::ipc::client_send_rust_bert_fraud_detection_request;
use rust_bert_fraud_detection_tools::build::classification::feature_importance;
use rust_bert_fraud_detection_tools::build::create_naive_bayes_model;
use rust_bert_fraud_detection_tools::build::data::{generate_shuffled_idx, split_vector};
use rust_bert_fraud_detection_tools::build::classification::deep_learning::{feature_importance_nn, get_new_nn, Predictor, z_score_normalize};
use rust_bert_fraud_detection_tools::build::feature_engineering::get_hard_coded_feature_labels;
use rust_bert_fraud_detection_tools::build::language_model::{get_fraud_indicators, get_n_best_fraud_indicators, load_embeddings_from_file};


const JSON_DATASET: [&str;6] = [
    "data_gen_v5_(youtubeSpamCollection).json",
    "data_gen_v5_(enronSpamSubset).json",
    "data_gen_v5_(lingSpam).json",
    "data_gen_v5_(smsspamcollection).json",
    "data_gen_v5_(completeSpamAssassin).json",
    "data_gen_v5_(governance_proposal_spam_likelihood).json"
];

const CSV_DATASET: [&str;6] = [
    "./dataset/youtubeSpamCollection.csv",
    "./dataset/enronSpamSubset.csv",
    "./dataset/lingSpam.csv",
    "./dataset/smsspamcollection.csv",
    "./dataset/completeSpamAssassin.csv",
    "./dataset/governance_proposal_spam_likelihood.csv"
];

pub const SENTENCES: [&str;6] = [
    "You!!! Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];


// nn using topics
// nn using hard_coded features

// random forest with all features
// nn with all features

// linear regression or random forest of the above two models. potentially remove naive bayes from models and add to regression.

// 1) generate_feature_vectors
// 2) train_and_test_final_model
// 3) feature_selection


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
        "naive_bayes_train_and_train_and_test_final_regression_model" => {naive_bayes_train_and_train_and_test_final_regression_model();},
        "naive_bayes_train" => {naive_bayes_train();},
        "naive_bayes_predict" => {naive_bayes_predict();},
        "train_and_test_final_model_random_forest_eval" => {train_and_test_final_model(true,"random_forest".to_string());},
        "train_and_test_final_model_random_forest" => {train_and_test_final_model(false,"random_forest".to_string());},
        "train_and_test_final_model_nn" => {train_and_test_final_model(false, "nn".to_string());},
        "train_and_test_final_model_nn_eval" => {train_and_test_final_model(true,"nn".to_string());},
        "generate_feature_vectors" => {generate_feature_vectors();},
        "service" => {service();},
        "feature_selection_random_forest" => {feature_selection("random_forest".to_string());},
        "feature_selection_nn" => {feature_selection("nn".to_string());},

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

fn naive_bayes_train_and_train_and_test_final_regression_model() -> anyhow::Result<()> {

    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;

    let dataset = rust_bert_fraud_detection_tools::build::data::read_datasets_and_shuffle(&CSV_DATASET,&shuffled_idx)?;

    let (train_dataset, test_dataset) = split_vector(&dataset,0.8);
    let train_dataset = train_dataset.to_vec();
    let test_dataset = test_dataset.to_vec();

    create_naive_bayes_model(&train_dataset,&test_dataset)?;

    let topic_selection = get_fraud_indicators(false);

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&JSON_DATASET,&shuffled_idx, &topic_selection, false,false,false)?;

    let (x_train, x_test) = split_vector(&x_dataset,0.8);
    let x_train = x_train.to_vec();
    let x_test = x_test.to_vec();
    let (y_train, y_test) = split_vector(&y_dataset,0.8);
    let y_train = y_train.to_vec();
    let y_test = y_test.to_vec();

    rust_bert_fraud_detection_tools::build::create_classification_model(&x_train,&y_train)?;
    rust_bert_fraud_detection_tools::build::test_classification_model(&x_test,&y_test)?;

    let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())

}


fn feature_selection(model: String) -> anyhow::Result<()> {

    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;
    let topic_selection = get_fraud_indicators(true);

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&JSON_DATASET,&shuffled_idx, &topic_selection, false,false,false)?;

    //let hard_coded_feature_labels = get_hard_coded_feature_labels();
    //let sentiment = "Sentiment".to_string();

    let feature_labels: Vec<String> = vec![/*hard_coded_feature_labels,*/topic_selection/*,sentiment*/].into_iter().flatten().collect();

    match model.as_str() {
        "random_forest" => {
            feature_importance(&x_dataset, &y_dataset, feature_labels)
        },
        "nn" => {
            let (x_dataset, mean, std_dev) = z_score_normalize(&x_dataset, None);
            feature_importance_nn(&x_dataset, &y_dataset, feature_labels)
        }
        _ => {
            panic!()
        }
    }
}




fn train_and_test_final_model(eval: bool, model: String) -> anyhow::Result<()> {

    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;

    //let topics = get_fraud_indicators(true);
    let topic_selection = get_n_best_fraud_indicators(30usize,&"feature_importance_random_forest_topics_only.json".to_string());
    //let topics = get_n_best_fraud_indicators(30usize,&"feature_importance_nn_topics_only.json".to_string());

    let (mut x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&JSON_DATASET,&shuffled_idx, &topic_selection, true,true,false)?;

    if model.as_str() == "nn" {
        let (z_dataset, mean, std_dev) = z_score_normalize(&x_dataset, None);
        x_dataset = z_dataset;
    }

    if !eval {
        match model.as_str() {
            "random_forest" => {
                rust_bert_fraud_detection_tools::build::create_classification_model(&x_dataset,&y_dataset)?;
                rust_bert_fraud_detection_tools::build::test_classification_model(&x_dataset,&y_dataset)?;            },
            "nn" => {
                let nn = rust_bert_fraud_detection_tools::build::classification::deep_learning::train_nn(&x_dataset,&y_dataset);
                let path = std::path::Path::new("./NeuralNet.bin");
                nn.save(path).unwrap();
                let mut nn = get_new_nn(x_dataset[0].len() as i64);
                nn.load(path).unwrap();
                rust_bert_fraud_detection_tools::build::classification::deep_learning::test_nn(&nn,&x_dataset,&y_dataset);
            }
            _ => {
                panic!()
            }
        }
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

        match model.as_str() {
            "random_forest" => {
                rust_bert_fraud_detection_tools::build::create_classification_model(&x_train, &y_train)?;
                rust_bert_fraud_detection_tools::build::test_classification_model(&x_train, &y_train)?;
                rust_bert_fraud_detection_tools::build::test_classification_model(&x_test, &y_test)?;
            },
            "nn" => {
                let nn = rust_bert_fraud_detection_tools::build::classification::deep_learning::train_nn(&x_train,&y_train);
                rust_bert_fraud_detection_tools::build::classification::deep_learning::test_nn(&&nn,&x_train,&y_train);
                rust_bert_fraud_detection_tools::build::classification::deep_learning::test_nn(&&nn,&x_test,&y_test);
            }
            _ => {
                panic!()
            }
        }
    }
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


    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;

    let dataset = rust_bert_fraud_detection_tools::build::data::read_datasets_and_shuffle(&CSV_DATASET[..],&shuffled_idx)?;

/*  let (train_dataset, test_dataset) = split_vector(&dataset,0.7);
    let train_dataset = train_dataset.to_vec();
    let test_dataset = test_dataset.to_vec();

    create_naive_bayes_model(&train_dataset,&test_dataset)
*/
    create_naive_bayes_model(&dataset,&dataset)
}

fn generate_feature_vectors() -> anyhow::Result<()> {

    // test governance spam ham
    // let training_data_path = "new_data_gen_v5_(governance_proposal_spam_likelihood).json";
    // rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/governance_proposal_spam_likelihood.csv"],training_data_path)?;


    for dataset in JSON_DATASET.into_iter().zip(CSV_DATASET) {
        let training_data_path = dataset.0.to_string();
        let dataset_path = dataset.1.to_string();

        rust_bert_fraud_detection_tools::build::create_training_data(vec![&dataset_path],&training_data_path)?;
    }


    return Ok(());

}
