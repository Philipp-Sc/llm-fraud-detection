
pub mod build;
use build::data::merge_datasets;
use build::classification::*;
use build::language_model::zero_shot_classification::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use importance::score::Model;


pub fn fraud_probabilities(texts: &[&str]) ->  anyhow::Result<Vec<f32>> {

    let mut text_embeddings: Vec<Vec<f32>> = Vec::new();
    for text in texts {

       let embedding = build::language_model::embeddings::llama_cpp_embedding(text)?;
       text_embeddings.push(embedding);

    }

    let model = ClassificationMockModel { label: "./KNNRegressor.bin".to_string(), model_type: ModelType::KNN};
    let text_embeddings = text_embeddings.into_par_iter().map(|vec| {
         if vec.len()>0 {
            model.predict(&vec![vec])
         }else{
            vec
         }
       }).collect::<Vec<Vec<f32>>>();

    let topic_selection = get_n_best_fraud_indicators(30usize,&"feature_importance_random_forest_topics_only.json".to_string());
    let topic_predictions: Vec<Vec<f32>> = get_topic_predictions(texts,&topic_selection.iter().map(|x| x.as_str()).collect::<Vec<&str>>()[..])?;
    let model = ClassificationMockModel { label: "./Topic_RandomForestRegressor.bin".to_string(), model_type: ModelType::RandomForest};
    let topic_predictions = topic_predictions.into_par_iter().map(|vec| {
         if vec.len()>0 {
            model.predict(&vec![vec])
         }else{
            vec
         }
       }).collect::<Vec<Vec<f32>>>();

    let x_dataset: Vec<Vec<f32>> = merge_datasets(topic_predictions,text_embeddings)?.into_iter().filter(|x| x.len()>0).collect();
    let model = ClassificationMockModel { label: "./MixtureModel_RandomForestRegressor.bin".to_string(), model_type: ModelType::RandomForest};
    let x_dataset = x_dataset.into_par_iter().map(|vec| {
         if vec.len()>0 {
            model.predict(&vec![vec])
         }else{
            vec
         }
       }).collect::<Vec<Vec<f32>>>();


    Ok(x_dataset.into_iter().flatten().collect())
}


