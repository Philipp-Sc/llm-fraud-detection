use rust_bert_fraud_detection_tools::build::classification::Model;
use rust_bert_fraud_detection_tools::build::classification::ClassificationMockModel;
use rust_bert_fraud_detection_tools::build::classification::ModelType;


use std::env;
use rust_bert_fraud_detection_tools::build::data::{generate_shuffled_idx, split_vector, DatasetKind};

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use rust_bert_fraud_detection_tools::build::language_model::zero_shot_classification::huggingface_transformers_extract_topics;
use rust_bert_fraud_detection_tools::build::language_model::zero_shot_classification::huggingface_transformers_predict_multilabel;
//use pyo3::prelude::*;
//use std::io::BufRead;

const JSON_DATASET: [&str;6] = [
    "data_gen_v7_(youtubeSpamCollection).json",
    "data_gen_v7_(enronSpamSubset).json",
    "data_gen_v7_(lingSpam).json",
    "data_gen_v7_(smsspamcollection).json",
    "data_gen_v7_(completeSpamAssassin).json",
    "data_gen_v7_(governance_proposal_spam_likelihood).json"
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
    "Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        println!("No command specified.");
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "test" => {test().unwrap();},
        "train_and_test_text_embedding_knn_regressor" => {train_and_test_text_embedding_knn_regressor(false)?;},
        "train_and_test_text_embedding_knn_regressor_eval" => {train_and_test_text_embedding_knn_regressor(true)?;},

        "train_and_test_zero_shot_classification_random_forest_regressor" => {train_and_test_zero_shot_classification_random_forest_regressor(false)?;},
        "train_and_test_zero_shot_classification_random_forest_regressor_eval" => {train_and_test_zero_shot_classification_random_forest_regressor(true)?;},

        "train_and_test_other_features_random_forest_regressor" => {train_and_test_other_features_random_forest_regressor(false)?;},
        "train_and_test_other_features_random_forest_regressor_eval" => {train_and_test_other_features_random_forest_regressor(true)?;},



        "train_and_test_mixture_model" => {train_and_test_mixture_model(false)?;},
        "train_and_test_mixture_model_eval" => {train_and_test_mixture_model(true)?;},
        "generate_feature_vectors" => {generate_feature_vectors()?;},
        "predict" => {let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
},
        _ => {panic!()}
    }

    Ok(())
}



fn test() -> anyhow::Result<()> {
    let topic_selection = rust_bert_fraud_detection_tools::build::language_model::zero_shot_classification::get_n_best_fraud_indicators(20usize,&"feature_importance_random_forest_topics_only.json".to_string());
    huggingface_transformers_predict_multilabel(&topic_selection.iter().map(|x| x.as_str()).collect::<Vec<&str>>()[..],"Don't forget our special promotion: -30% on men shoes, only today!")?;

/*
pyo3::prepare_freethreaded_python();
Python::with_gil(|py| {
        println!("Downloading and loading models...");

        let module = PyModule::from_code(
            py,
            include_str!("../huggingface.py"),
            "huggingface.py",
            "huggingface",
        )?;

        let extract_topics: Py<PyAny> = module.getattr("extract_topics")?.into();

        println!("Done! Type a sentence and hit enter. To exit hold Ctrl+C and hit Enter");

        let stdin = std::io::stdin();

        for line in stdin.lock().lines() {
            let Ok(text) = line else {
                break;
            };

            let samples: Vec<f32> = extract_topics.call1(py, (["stress","sex","apple"],"text about sex",))?.extract(py)?;
            dbg!(samples.len());
        }

        Ok(())
    })
*/
Ok(())
}


fn train_and_test_text_embedding_knn_regressor(eval: bool) -> anyhow::Result<()> {
    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(DatasetKind::Embedding, &JSON_DATASET,&shuffled_idx)?;

    if !eval {
                rust_bert_fraud_detection_tools::build::classification::update_knn_regression_model(&x_dataset,&y_dataset)?;
                rust_bert_fraud_detection_tools::build::classification::test_knn_regression_model(&x_dataset,&y_dataset)?;
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

                rust_bert_fraud_detection_tools::build::classification::update_knn_regression_model(&x_train,&y_train)?;
                rust_bert_fraud_detection_tools::build::classification::test_knn_regression_model(&x_train,&y_train)?;
                rust_bert_fraud_detection_tools::build::classification::test_knn_regression_model(&x_test,&y_test)?;
    }
    Ok(())
}
fn train_and_test_zero_shot_classification_random_forest_regressor(eval: bool) -> anyhow::Result<()> {
    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;

    let topic_selection = rust_bert_fraud_detection_tools::build::language_model::zero_shot_classification::get_n_best_fraud_indicators(30usize,&"feature_importance_random_forest_topics_only.json".to_string());

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(DatasetKind::ZeroShotClassification(topic_selection), &JSON_DATASET,&shuffled_idx)?;

    let	path = "./Topic_RandomForestRegressor.bin";

    if !eval {
              	rust_bert_fraud_detection_tools::build::classification::update_random_forest_regression_model(path, &x_dataset,&y_dataset)?;
                rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_dataset,&y_dataset)?;
    }else {
	let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

                rust_bert_fraud_detection_tools::build::classification::update_random_forest_regression_model(path, &x_train,&y_train)?;
                rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_train,&y_train)?;
                rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_test,&y_test)?;
    }
    Ok(())
}

fn train_and_test_other_features_random_forest_regressor(eval: bool) -> anyhow::Result<()> {
    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;
    
    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(DatasetKind::OtherFeatures, &JSON_DATASET,&shuffled_idx)?;

    let path = "./OtherFeatures_RandomForestRegressor.bin";

    if !eval {
        rust_bert_fraud_detection_tools::build::classification::update_random_forest_regression_model(path, &x_dataset,&y_dataset)?;
        rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_dataset,&y_dataset)?;
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

        rust_bert_fraud_detection_tools::build::classification::update_random_forest_regression_model(path, &x_train,&y_train)?;
        rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_train,&y_train)?;
        rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_test,&y_test)?;
    }
    Ok(())
}

fn train_and_test_mixture_model(eval: bool) -> anyhow::Result<()> {
    let shuffled_idx = generate_shuffled_idx(&JSON_DATASET)?;

    let topic_selection = rust_bert_fraud_detection_tools::build::language_model::zero_shot_classification::get_n_best_fraud_indicators(30usize,&"feature_importance_random_forest_topics_only.json".to_string());

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(DatasetKind::ZeroShotClassification(topic_selection), &JSON_DATASET,&shuffled_idx)?;
    let model = ClassificationMockModel { label: "./Topic_RandomForestRegressor.bin".to_string(), model_type: ModelType::RandomForest};
    let x_dataset = x_dataset.into_par_iter().map(|vec| {
         if vec.len()>0 {
            model.predict(&vec![vec])
         }else{
            vec
         }
       }).collect::<Vec<Vec<f32>>>();

    let (x_dataset2, y_dataset2) = rust_bert_fraud_detection_tools::build::data::create_dataset(DatasetKind::Embedding, &JSON_DATASET,&shuffled_idx)?;
    assert_eq!(y_dataset,y_dataset2);
    let model = ClassificationMockModel { label: "./KNNRegressor.bin".to_string(), model_type: ModelType::KNN};
    let x_dataset2 = x_dataset2.into_par_iter().map(|vec| {
         if vec.len()>0 {
            model.predict(&vec![vec])
       	 }else{
       	    vec
       	 }
       }).collect::<Vec<Vec<f32>>>();

    let x_dataset = rust_bert_fraud_detection_tools::build::data::merge_datasets(x_dataset,x_dataset2)?.into_iter().filter(|x| x.len()>0).collect();

/*
    let (x_dataset3, y_dataset3) = rust_bert_fraud_detection_tools::build::data::create_dataset(DatasetKind::OtherFeatures, &JSON_DATASET,&shuffled_idx)?;
    assert_eq!(y_dataset,y_dataset3);
    let model = ClassificationMockModel { label: "./OtherFeatures_RandomForestRegressor.bin".to_string(), model_type: ModelType::RandomForest};
    let x_dataset3 = x_dataset3.into_par_iter().map(|vec| {
         if vec.len()>0 {
            model.predict(&vec![vec])
         }else{
            vec
         }
       }).collect::<Vec<Vec<f32>>>();

    let x_dataset = rust_bert_fraud_detection_tools::build::data::merge_datasets(x_dataset,x_dataset3)?.into_iter().filter(|x| x.len()>0).collect();
*/   
/*
      let (x_dataset4, y_dataset4) = rust_bert_fraud_detection_tools::build::data::create_dataset(DatasetKind::Sentiment, &JSON_DATASET,&shuffled_idx)?;
      assert_eq!(y_dataset,y_dataset4);

      let x_dataset = rust_bert_fraud_detection_tools::build::data::merge_datasets(x_dataset,x_dataset4)?.into_iter().filter(|x| x.len()>0).collect();
*/
    let path = "./MixtureModel_RandomForestRegressor.bin";
    if !eval {
                rust_bert_fraud_detection_tools::build::classification::update_random_forest_regression_model(path, &x_dataset,&y_dataset)?;
                rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_dataset,&y_dataset)?;
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

                rust_bert_fraud_detection_tools::build::classification::update_random_forest_regression_model(path, &x_train,&y_train)?;
                rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_train,&y_train)?;
                rust_bert_fraud_detection_tools::build::classification::test_random_forest_regression_model(path, &x_test,&y_test)?;
    }
    Ok(())
}


fn generate_feature_vectors() -> anyhow::Result<()> {

    for dataset in JSON_DATASET.into_iter().zip(CSV_DATASET) {
        let training_data_path = dataset.0.to_string();
        let dataset_path = dataset.1.to_string();

        rust_bert_fraud_detection_tools::build::create_training_data(vec![&dataset_path],&training_data_path)?;
    }


    return Ok(());

}
