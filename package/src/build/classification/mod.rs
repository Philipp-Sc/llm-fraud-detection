use std::cmp::Ordering;
use smartcore::linear::linear_regression::*;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
//use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;

use std::sync::Arc;
use std::sync::Mutex;

use std::{fs, thread};
use std::time::Duration;

use importance::*;
use importance::score::*;
use smartcore::math::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_regressor::KNNRegressor;

use smartcore::neighbors::KNNWeightFunction;


pub mod deep_learning;
lazy_static::lazy_static! {
    static ref PREDICTOR_POOL: Arc<Mutex<Vec<(String, Arc<Mutex<Box<dyn Model>>>)>>>
        = Arc::new(Mutex::new(Vec::new()));
}

pub enum ModelType {
    RandomForest,
    KNN,
}

pub struct ClassificationMockModel {
    pub label: String,
    pub model_type: ModelType,
}

struct RandomForestRegressorModel(RandomForestRegressor<f64>);
struct KNNRegressorModel(KNNRegressor<f64,Euclidian>);

impl Model for RandomForestRegressorModel {
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let x = DenseMatrix::from_2d_array(&x.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
        self.0.predict(&x).unwrap()
    }
}
impl Model for KNNRegressorModel {
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let x = DenseMatrix::from_2d_array(&x.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
        self.0.predict(&x).unwrap()
    }
}


impl Model for ClassificationMockModel {
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let predictor = get_model(&self.label, &self.model_type).expect("Failed to get a predictor from the pool.");
        let model = predictor.lock().unwrap();
        model.predict(&x)
    }
}


fn get_model(label: &String, model_type: &ModelType) -> anyhow::Result<Arc<Mutex<Box<dyn Model>>>> {
    let mut pool = PREDICTOR_POOL.lock().unwrap();

    // Check if any predictor is available in the pool
    if let Some((_, predictor)) = pool.iter().find(|(l, p)| l == label && !p.try_lock().is_err()) {
        return Ok(Arc::clone(predictor));
    }

    // If all predictors are in use, wait until one becomes available
    while pool.len() >= 500 {
        thread::sleep(Duration::from_millis(100));
        if let Some((_, predictor)) = pool.iter().find(|(l, p)| l == label && !p.try_lock().is_err()) {
            return Ok(Arc::clone(predictor));
        }
    }

    // Create a new predictor and add it to the pool
    let new_predictor: Box<dyn Model> = match model_type {
        ModelType::RandomForest => {
            let model: RandomForestRegressor<f64> = match serde_json::from_str(&fs::read_to_string(label)?)? {
                Some(lr) => lr,
                None => return Err(anyhow::anyhow!(format!("Error: unable to load '{}'", label))),
            };
            Box::new(RandomForestRegressorModel(model))
        }
        ModelType::KNN => {
            let model: KNNRegressor<f64,Euclidian> = match serde_json::from_str(&fs::read_to_string(label)?)? {
                Some(lr) => { lr },
                None => { return Err(anyhow::anyhow!(format!("Error: unable to load '{}'",label)));}
            };
            Box::new(KNNRegressorModel(model))
        }
    };

    let new_predictor = Arc::new(Mutex::new(new_predictor));
    pool.push((label.to_owned(), Arc::clone(&new_predictor)));

    Ok(new_predictor)
}

pub fn predict(x_dataset: &Vec<Vec<f64>>) ->  anyhow::Result<Vec<f64>> {

    let model = ClassificationMockModel {
        label: "./RandomForestRegressor.bin".to_string(), model_type: ModelType::RandomForest
    };

    Ok(model.predict(x_dataset))
}

pub fn update_knn_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) ->  anyhow::Result<()> {

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
    let y = y_dataset;


    let regressor = KNNRegressor::fit(&x, &y, smartcore::neighbors::knn_regressor::KNNRegressorParameters::default().with_k(3).with_weight(KNNWeightFunction::Distance)).unwrap();
    fs::write("./KNNRegressor.bin", &serde_json::to_string(&regressor)?).ok();

    Ok(())
}


pub fn test_knn_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) -> anyhow::Result<()> {

    let model: KNNRegressor<f64,Euclidian> = match serde_json::from_str(&fs::read_to_string("./KNNRegressor.bin")?)? {
        Some(lr) => { lr },
        None => { return Err(anyhow::anyhow!(format!("Error: unable to load '{}'","./KNNRegressor.bin")));}
    };
    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);

    let y_hat = model.predict(&x).unwrap();

    let thresholds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    calculate_metrics(&y_dataset,&y_hat, &thresholds);
    Ok(())
}

pub fn predict_knn_regression_model(x_dataset: &Vec<Vec<f64>>) -> anyhow::Result<Vec<f64>> {

    let model: KNNRegressor<f64,Euclidian> = match serde_json::from_str(&fs::read_to_string("./KNNRegressor.bin")?)? {
        Some(lr) => { lr },
        None => { return Err(anyhow::anyhow!(format!("Error: unable to load '{}'","./KNNRegressor.bin")));}
    };
    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);

    let y_hat = model.predict(&x).unwrap();

    Ok(y_hat)
}



pub fn update_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) ->  anyhow::Result<()> {

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
    let y = y_dataset;


    let regressor = RandomForestRegressor::fit(&x, &y, smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters::default()/*.with_max_depth(32)*/.with_n_trees(32).with_min_samples_leaf(4)).unwrap();
    fs::write("./RandomForestRegressor.bin", &serde_json::to_string(&regressor)?).ok();

    Ok(())
}


pub fn test_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) -> anyhow::Result<()> {

    let model = ClassificationMockModel {
        label: "./RandomForestRegressor.bin".to_string(), model_type: ModelType::RandomForest
    };
    let y_hat = model.predict(x_dataset);

    let thresholds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    calculate_metrics(&y_dataset,&y_hat, &thresholds);
    Ok(())
}

pub fn calculate_metrics(y: &[f64], y_hat: &[f64], thresholds: &[f64]) {
    for &threshold in thresholds {
        let true_positive_count = y_hat.iter().zip(y.iter()).filter(|(&y_hat_val, &y_val)| y_hat_val >= threshold && y_val >= 0.5).count();
        let false_positive_count = y_hat.iter().zip(y.iter()).filter(|(&y_hat_val, &y_val)| y_hat_val >= threshold && y_val < 0.5).count();
        let false_negative_count = y_hat.iter().zip(y.iter()).filter(|(&y_hat_val, &y_val)| y_hat_val < threshold && y_val >= 0.5).count();

        let precision = true_positive_count as f64 / (true_positive_count + false_positive_count) as f64;
        let recall = true_positive_count as f64 / (true_positive_count + false_negative_count) as f64;
        let f_score = 2.0 * (precision * recall) / (precision + recall);

        println!(
            "Threshold >= {}: True Positive = {}, False Positive = {}, Precision = {:.3}, Recall = {:.3}, F-Score = {:.3}",
            threshold, true_positive_count, false_positive_count, precision, recall, f_score
        );
    }
}



pub fn feature_importance(x_dataset_shuffled: &Vec<Vec<f64>>, y_dataset_shuffled: &Vec<f64>, feature_labels: Vec<String>) -> anyhow::Result<()> {

    let model = ClassificationMockModel {
        label: "./RandomForestRegressor.bin".to_string(),
        model_type: ModelType::RandomForest
    };

    let opts = Opts {
        verbose: true,
        kind: Some(ScoreKind::Mae),
        n: Some(500),
        only_means: true,
        scale: true,
    };

    let importances = importance(&model, x_dataset_shuffled.to_owned(), y_dataset_shuffled.to_owned(), opts);
    println!("Importances: {:?}", importances);

    let importances_means: Vec<f64> = importances.importances_means;
    let mut result: Vec<(f64, String)> = importances_means.into_iter().zip(feature_labels).collect();
    result.sort_by(|(a,_), (b,_)| b.partial_cmp(&a).unwrap_or(Ordering::Equal));

    let json_string = serde_json::json!({"feature_importance": &result}).to_string();

    println!("Result: \n\n{:?}", json_string);

    Ok(())

}
