use smartcore::linear::linear_regression::*;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
//use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;

use std::sync::Arc;
use std::sync::Mutex;

use std::fs;

use importance::*;
use importance::score::*;

lazy_static::lazy_static! {
        static ref MODEL: Arc<Mutex<RandomForestRegressor<f64>>> = Arc::new(Mutex::new(get_model().unwrap()));
    }

struct MockModel;

impl Model for MockModel {
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let model = get_model().unwrap();
        let x = DenseMatrix::from_2d_array(&x.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
        model.predict(&x).unwrap()
    }
}

fn get_model() -> anyhow::Result<RandomForestRegressor<f64>>{
    let model: RandomForestRegressor<f64> = match serde_json::from_str(&fs::read_to_string("./RandomForestRegressor.bin")?)? {
        Some(lr) => { lr },
        None => { return Err(anyhow::anyhow!("Error: unable to load './RandomForestRegressor.bin'"));}
    };
    Ok(model)
}

pub fn predict(x_dataset: &Vec<Vec<f64>>) ->  anyhow::Result<Vec<f64>> {

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);

    let model = MODEL.try_lock().unwrap();

    Ok(model.predict(&x)?)
}


pub fn update_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) ->  anyhow::Result<()> {

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
    let y = y_dataset;


    let regressor = RandomForestRegressor::fit(&x, &y, smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters::default()/*.with_max_depth(32)*/.with_n_trees(32).with_min_samples_leaf(4)).unwrap();
    fs::write("./RandomForestRegressor.bin", &serde_json::to_string(&regressor)?).ok();

    Ok(())
}


pub fn test_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) -> anyhow::Result<()> {

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
    let y = y_dataset;

//    let model = MODEL.try_lock().unwrap();
    let model = get_model()?;
    let y_hat = model.predict(&x).unwrap();


    for ii in vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let true_positive_count = (0..y.len()).filter(|&i| y_hat[i] >= ii && y[i] >= 0.5).count();
        let false_positive_count = (0..y.len()).filter(|&i| y_hat[i] >= ii && y[i] < 0.5).count();
        let false_negative_count = (0..y.len()).filter(|&i| y_hat[i] < ii && y[i] >= 0.5).count();

        let precision = true_positive_count as f64 / (true_positive_count + false_positive_count) as f64;
        let recall = true_positive_count as f64 / (true_positive_count + false_negative_count) as f64;
        let f_score = 2.0 * (precision * recall) / (precision + recall);

        println!(
            "Threshold >= {}: True Positive = {}, False Positive = {}, Precision = {:.3}, Recall = {:.3}, F-Score = {:.3}",
            ii, true_positive_count, false_positive_count, precision, recall, f_score
        );
    }
    Ok(())
}


pub fn feature_importance(x_dataset_shuffled: &Vec<Vec<f64>>, y_dataset_shuffled: &Vec<f64>) -> anyhow::Result<()> {

    let model = MockModel;

    let opts = Opts {
        verbose: true,
        kind: Some(ScoreKind::Smape),
        n: Some(1),
        only_means: true,
        scale: true,
    };

    let importances = importance(&model, x_dataset_shuffled.to_vec().into_iter().take(10).collect(), y_dataset_shuffled.to_vec().into_iter().take(10).collect(), opts);
    println!("Importances: {:?}", importances);

    Ok(())

}