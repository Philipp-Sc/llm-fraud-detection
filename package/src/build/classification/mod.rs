use smartcore::linear::linear_regression::*;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
//use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;

use std::fs;

pub fn predict(x_dataset: &Vec<Vec<f64>>) ->  anyhow::Result<Vec<f64>> {

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);

      // loads regression model from file.
    /*let lr: LinearRegression<f64, DenseMatrix<f64>> = match serde_json::from_str(&fs::read_to_string("./LinearRegression.bin")?)? {
        Some(lr) => { lr },
        None => { return Err(anyhow::anyhow!("Error: unable to load './LinearRegression.bin'"));}
    };*/
    /*
        let lr: RandomForestClassifier<f64> = match serde_json::from_str(&fs::read_to_string("./RandomForestClassifier.bin")?)? {
            Some(lr) => { lr },
            None => { return Err(anyhow::anyhow!("Error: unable to load './RandomForestClassifier.bin'"));}
        };*/
    let lr: RandomForestRegressor<f64> = match serde_json::from_str(&fs::read_to_string("./RandomForestRegressor.bin")?)? {
        Some(lr) => { lr },
        None => { return Err(anyhow::anyhow!("Error: unable to load './RandomForestRegressor.bin'"));}
    };

    Ok(lr.predict(&x)?)
}


pub fn update_linear_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) ->  anyhow::Result<()> {
    //let (x_dataset, y_dataset): (Vec<Vec<f64>>,Vec<f64>) = get_dataset(path)?;

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
    let y = y_dataset;


    let regressor = RandomForestRegressor::fit(&x, &y, smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters::default()/*.with_max_depth(32)*/.with_n_trees(32).with_min_samples_leaf(4)).unwrap();
    fs::write("./RandomForestRegressor.bin", &serde_json::to_string(&regressor)?).ok();
    /*
    let classifier = RandomForestClassifier::fit(&x, &y, smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default().with_max_depth(3)).unwrap();
    fs::write("./RandomForestClassifier.bin", &serde_json::to_string(&classifier)?).ok();


*/
/*
    let tree = LinearRegression::fit(&x, &y,
                                     LinearRegressionParameters::default().
                                         with_solver(LinearRegressionSolverName::QR)).unwrap();


    fs::write("./LinearRegression.bin", &serde_json::to_string(&tree)?).ok();
*/
    Ok(())
}


pub fn test_linear_regression_model(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) -> anyhow::Result<()> {
    /*
        let lr: LinearRegression<f64, DenseMatrix<f64>> = match serde_json::from_str(&fs::read_to_string("./LinearRegression.bin")?)? {
            Some(lr) => { lr },
            None => { return Err(anyhow::anyhow!("Error: unable to load './LinearRegression.bin'"));}
        };*/
    /*    let lr: RandomForestClassifier<f64> = match serde_json::from_str(&fs::read_to_string("./RandomForestClassifier.bin")?)? {
               Some(lr) => { lr },
               None => { return Err(anyhow::anyhow!("Error: unable to load './RandomForestClassifier.bin'"));}
           };
       */
    let lr: RandomForestRegressor<f64> = match serde_json::from_str(&fs::read_to_string("./RandomForestRegressor.bin")?)? {
        Some(lr) => { lr },
        None => { return Err(anyhow::anyhow!("Error: unable to load './RandomForestRegressor.bin'"));}
    };

    let x = DenseMatrix::from_2d_array(&x_dataset.iter().map(|x| &x[..]).collect::<Vec<&[f64]>>()[..]);
    let y = y_dataset;

    let y_hat = lr.predict(&x).unwrap();

    for i in 0..y.len() {
        let p = if y_hat[i] > 0.5 {1.0}else{0.0};

        /*
        if p != y[i] {
            println!("{:?}",p);
            println!("{:?}",y[i]);

        }
        */
    }
    for ii in vec![0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]{
    println!("true p(>={})==label {:?}",ii,(0..y.len()).map(|i| {
        let p = if y_hat[i] >= ii {1.0}else{0.0};
        p==y[i]
    }).filter(|x| *x).count());
    println!("false {:?}",(0..y.len()).map(|i| {
        let p = if y_hat[i] >= ii {1.0}else{0.0};
        p!=y[i]
    }).filter(|x| *x).count());
    println!("false positive {:?}\n",(0..y.len()).map(|i| {
        let p = if y_hat[i] >= ii {1.0}else{0.0};
        p!=y[i] && y[i]==0.0
    }).filter(|x| *x).count());
    }
    Ok(())
}
