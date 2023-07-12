use std::cmp::min;
use linfa_preprocessing::CountVectorizer;
use linfa_bayes::{GaussianNbParams, GaussianNbValidParams, Result};
use linfa::prelude::*;
use linfa_bayes::{GaussianNb};
use ndarray::{Axis, Dim, Array, ArrayBase, Data, Ix1, OwnedRepr, ViewRepr};
use linfa::dataset::Labels;
use std::collections::HashMap;

use std::fs;
use std::ops::Deref;
use regex::Regex;

use std::sync::Arc;
use std::sync::Mutex;
use lazy_static::lazy_static;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::naive_bayes::categorical::CategoricalNB;

lazy_static! {
    static ref GAUSSIAN_NB_POOL: Arc<Mutex<Vec<(String, Arc<Mutex<GaussianNb<f32, usize>>>)>>> = Arc::new(Mutex::new(Vec::new()));
    static ref CATEGORICAL_NB_POOL: Arc<Mutex<Vec<(String, Arc<Mutex<CategoricalNB<f32, DenseMatrix<f32>>>>)>>> = Arc::new(Mutex::new(Vec::new()));
    static ref VECTORIZER_POOL: Arc<Mutex<Vec<(String, Arc<Mutex<CountVectorizer>>)>>> = Arc::new(Mutex::new(Vec::new()));
}

fn get_gaussian_nb_model() -> anyhow::Result<Arc<Mutex<GaussianNb<f32, usize>>>> {
    let mut pool = GAUSSIAN_NB_POOL.deref().lock().unwrap();

    // Check if any predictor is available in the pool
    if let Some((_, predictor)) = pool.iter().find(|(_, p)| !p.try_lock().is_err()) {
        return Ok(Arc::clone(predictor));
    }

    // If all predictors are in use, wait until one becomes available
    while pool.len() >= 500 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if let Some((_, predictor)) = pool.iter().find(|(_, p)| !p.try_lock().is_err()) {
            return Ok(Arc::clone(predictor));
        }
    }

    // Create a new predictor and add it to the pool
    let new_predictor = match serde_json::from_str(&fs::read_to_string("./GaussianNbModel.bin")?)? {
        Some(lr) => lr,
        None => return Err(anyhow::anyhow!("Error: unable to load './GaussianNbModel.bin'"))
    };
    let new_predictor = Arc::new(Mutex::new(new_predictor));
    pool.push(("./GaussianNbModel.bin".to_owned(), Arc::clone(&new_predictor)));

    Ok(new_predictor)
}

fn get_categorical_nb_model() -> anyhow::Result<Arc<Mutex<CategoricalNB<f32, DenseMatrix<f32>>>>> {
    let mut pool = CATEGORICAL_NB_POOL.deref().lock().unwrap();

    // Check if any predictor is available in the pool
    if let Some((_, predictor)) = pool.iter().find(|(_, p)| !p.try_lock().is_err()) {
        return Ok(Arc::clone(predictor));
    }

    // If all predictors are in use, wait until one becomes available
    while pool.len() >= 500 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if let Some((_, predictor)) = pool.iter().find(|(_, p)| !p.try_lock().is_err()) {
            return Ok(Arc::clone(predictor));
        }
    }

    // Create a new predictor and add it to the pool
    let new_predictor = match serde_json::from_str(&fs::read_to_string("./CategoricalNBModel.bin")?)? {
        Some(lr) => lr,
        None => return Err(anyhow::anyhow!("Error: unable to load './CategoricalNBModel.bin'"))
    };
    let new_predictor = Arc::new(Mutex::new(new_predictor));
    pool.push(("./CategoricalNBModel.bin".to_owned(), Arc::clone(&new_predictor)));

    Ok(new_predictor)
}

fn get_vectorizer() -> anyhow::Result<Arc<Mutex<CountVectorizer>>> {
    let mut pool = VECTORIZER_POOL.deref().lock().unwrap();

    // Check if any vectorizer is available in the pool
    if let Some((_, vectorizer)) = pool.iter().find(|(_, v)| !v.try_lock().is_err()) {
        return Ok(Arc::clone(vectorizer));
    }

    // If all vectorizers are in use, wait until one becomes available
    while pool.len() >= 500 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if let Some((_, vectorizer)) = pool.iter().find(|(_, v)| !v.try_lock().is_err()) {
            return Ok(Arc::clone(vectorizer));
        }
    }

    // Create a new vectorizer and add it to the pool
    let new_vectorizer = match serde_json::from_str(&fs::read_to_string("./CountVectorizer.bin")?)? {
        Some(lr) => lr,
        None => return Err(anyhow::anyhow!("Error: unable to load './CountVectorizer.bin'"))
    };
    let new_vectorizer = Arc::new(Mutex::new(new_vectorizer));
    pool.push(("./CountVectorizer.bin".to_owned(), Arc::clone(&new_vectorizer)));

    Ok(new_vectorizer)
}

fn remove_non_letters(text: &str) -> String {
    let regex = Regex::new(r"[^a-zA-Z]").unwrap();
    let processed_text = regex.replace_all(text, " ").to_string();
    let regex = Regex::new(r"\s+").unwrap();
    let cleaned_text = regex.replace_all(&processed_text, " ").to_string();
    cleaned_text
}

pub fn gaussian_nb_model_predict(x_dataset: Vec<String>) ->  anyhow::Result<Vec<usize>> {
    let binding = get_vectorizer()?;
    let vectorizer = binding.lock().unwrap();


    let test_texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((x_dataset.len(),), x_dataset).unwrap();

    let test_records = vectorizer.transform(&test_texts).to_dense();
    let test_records = test_records.mapv(|c| c as f32);

    let binding = get_gaussian_nb_model()?;
    let model = binding.lock().unwrap();


    let prediction = model.predict(&test_records).into_raw_vec();
    Ok(prediction)

}

pub fn categorical_nb_model_predict(x_dataset: Vec<String>) ->  anyhow::Result<Vec<usize>> {
    let binding = get_vectorizer()?;
    let vectorizer = binding.lock().unwrap();

    let test_texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((x_dataset.len(),), x_dataset).unwrap();
    let test_records = vectorizer.transform(&test_texts).to_dense();

    let x_data: Vec<Vec<f32>> = test_records.outer_iter().map(|row| row.to_vec().into_iter().map(|x| x as f32).collect::<Vec<f32>>()).collect();
    let x = DenseMatrix::<f32>::from_2d_vec(&x_data);

    let binding = get_categorical_nb_model()?;
    let model = binding.lock().unwrap();


    let prediction = model.predict(&x)?.into_iter().map(|x| x as usize).collect();
    Ok(prediction)

}


pub fn update_naive_bayes_model(x_dataset: &Vec<String>, y_dataset: &Vec<i32>,test_x_dataset: &Vec<String>, test_y_dataset: &Vec<i32>) ->  anyhow::Result<()> {

    let texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((x_dataset.len(),), x_dataset.iter().map(|x| remove_non_letters(&x)).collect()).unwrap();
    let labels: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((y_dataset.len(),), y_dataset.iter().map(|&x| x as usize).collect()).unwrap();

    let test_texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((test_x_dataset.len(),), test_x_dataset.clone()).unwrap();
    let test_labels: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((test_y_dataset.len(),), test_y_dataset.iter().map(|&x| x as usize).collect()).unwrap();

    // Counting label categories in the training dataset
    let mut training_label_counts: HashMap<usize, usize> = HashMap::new();
    for &label in labels.iter() {
        let count = training_label_counts.entry(label).or_insert(0);
        *count += 1;
    }

    println!("Label category counts in the training dataset:");
    for (label, count) in &training_label_counts {
        println!("Label: {}, Count: {}", label, count);
    }

    println!();

    // Counting label categories in the test dataset
    let mut test_label_counts: HashMap<usize, usize> = HashMap::new();
    for &label in test_labels.iter() {
        let count = test_label_counts.entry(label).or_insert(0);
        *count += 1;
    }

    println!("Label category counts in the test dataset:");
    for (label, count) in &test_label_counts {
        println!("Label: {}, Count: {}", label, count);
    }
    println!();


    let min_freq = 0.0005;
    let max_freq = 0.500;
    let vectorizer = CountVectorizer::params()
        .convert_to_lowercase(true)
        .document_frequency(min_freq,max_freq)
        .n_gram_range(1,3)
        .fit(&texts).unwrap(); 

    println!(
        "We obtain a vocabulary with {} entries",
        vectorizer.nentries()
    );
/*    println!(
        "Vocabulary entries: {:?}",
        vectorizer.vocabulary()
    );
*/

    fs::write("./CountVectorizer.bin", &serde_json::to_string(&vectorizer)?).ok();


    println!();
    println!(
        "Now let's generate a matrix containing the tf-idf value of each entry in each document"
    );
    // Transforming gives a sparse dataset, we make it dense in order to be able to fit the Naive Bayes model
    let training_records = vectorizer.transform(&texts).to_dense();
    // Currently linfa only allows real valued features so we have to transform the integer counts to floats
    let training_records = training_records.mapv(|c| c as f32);

    println!(
        "We obtain a {}x{} matrix of counts for the vocabulary entries",
        training_records.dim().0,
        training_records.dim().1
    );
    println!();

    // Displaying sample entries of the training data
    println!("Sample entries of the training data:");
    let num_samples = 5;
    for i in 0..num_samples {
        let sample_entry = training_records.index_axis(Axis(0), i);
        let label = labels[i];
        let text = texts[i].to_owned();
        println!("Text: {}, Data: {:?}, Label: {}", text, sample_entry, label);
    }
    println!();

    let ds = DatasetView::new(training_records.view(), labels.view());

    // Transforming gives a sparse dataset, we make it dense in order to be able to fit the Naive Bayes model
    let test_records = vectorizer.transform(&test_texts).to_dense();
    // Currently linfa only allows real valued features so we have to transform the integer counts to floats
    let test_records = test_records.mapv(|c| c as f32);

    println!(
        "We obtain a {}x{} test matrix of counts for the vocabulary entries",
        test_records.dim().0,
        test_records.dim().1
    );
    println!();


    // Displaying sample entries of the test data
    println!("Sample entries of the test data:");
    for i in 0..num_samples {
        let sample_entry = test_records.index_axis(Axis(0), i);
        let label = test_labels[i];
        let text = test_texts[i].to_owned();
        println!("Text: {}, Data: {:?}, Label: {}", text, sample_entry, label);
    }
    println!();

    let test_ds = DatasetView::new(test_records.view(), test_labels.view());


    // create a new parameter set with variance smoothing equals `1e-5`
    let unchecked_params = GaussianNbParams::new()
        .var_smoothing(1e-5);

    // fit model with unchecked parameter set
    let model = unchecked_params.fit(&ds).unwrap();

    // transform into a verified parameter set
    let checked_params = unchecked_params.check().unwrap();

    // update model with the verified parameters, this only returns
    // errors originating from the fitting process
    let model = checked_params.fit_with(Some(model), &ds).unwrap().unwrap();
    fs::write("./GaussianNbModel.bin", &serde_json::to_string(&model)?).ok();


    let prediction = model.predict(&test_ds);

    // Displaying predictions
    println!("Predictions:");
    let num_predictions = 5; 
    for i in 0..num_predictions {
        let prediction = prediction.index_axis(Axis(0), i);
        let true_label = test_labels[i];
        let text = test_texts[i].to_owned();
        println!("Text: {}, Prediction: {}, True Label: {}", text, prediction, true_label);
    }
    println!();

    let cm = prediction
        .confusion_matrix(&test_ds)
        .unwrap();
    // 0.9944
    let accuracy = cm.f1_score();
    println!("The fitted model has a training f1 score of {}", accuracy);

    Ok(())
}




pub fn update_categorical_naive_bayes_model(x_dataset: &Vec<String>, y_dataset: &Vec<i32>,test_x_dataset: &Vec<String>, test_y_dataset: &Vec<i32>) ->  anyhow::Result<()> {

    let texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((x_dataset.len(),), x_dataset.iter().map(|x| remove_non_letters(&x)).collect()).unwrap();
    let labels: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((y_dataset.len(),), y_dataset.iter().map(|&x| x as usize).collect()).unwrap();

    let test_texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((test_x_dataset.len(),), test_x_dataset.clone()).unwrap();
    let test_labels: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((test_y_dataset.len(),), test_y_dataset.iter().map(|&x| x as usize).collect()).unwrap();

    // Counting label categories in the training dataset
    let mut training_label_counts: HashMap<usize, usize> = HashMap::new();
    for &label in labels.iter() {
        let count = training_label_counts.entry(label).or_insert(0);
        *count += 1;
    }

    println!("Label category counts in the training dataset:");
    for (label, count) in &training_label_counts {
        println!("Label: {}, Count: {}", label, count);
    }

    println!();

    let binding = get_vectorizer()?;
    let vectorizer = binding.lock().unwrap();


    /*
        let min_freq = 0.0005;
        let max_freq = 0.500;
        let vectorizer = CountVectorizer::params()
            .convert_to_lowercase(true)
            .document_frequency(min_freq,max_freq)
            .n_gram_range(1,3)
            .fit(&texts).unwrap();
    */

    println!(
        "We obtain a vocabulary with {} entries",
        vectorizer.nentries()
    );
    /*    println!(
            "Vocabulary entries: {:?}",
            vectorizer.vocabulary()
        );
    */

    //fs::write("./CountVectorizer.bin", &serde_json::to_string(&vectorizer)?).ok();

    println!();
    println!(
        "Now let's generate a matrix containing the tf-idf value of each entry in each document"
    );
    // Transforming gives a sparse dataset, we make it dense in order to be able to fit the Naive Bayes model
    let training_records = vectorizer.transform(&texts).to_dense();

    println!(
        "We obtain a {}x{} matrix of counts for the vocabulary entries",
        training_records.dim().0,
        training_records.dim().1
    );
    println!();

    // Transforming gives a sparse dataset, we make it dense in order to be able to fit the Naive Bayes model
    let test_records = vectorizer.transform(&test_texts).to_dense();

    println!(
        "We obtain a {}x{} test matrix of counts for the vocabulary entries",
        test_records.dim().0,
        test_records.dim().1
    );
    println!();

    let x_data: Vec<Vec<f32>> = training_records.outer_iter().map(|row| row.to_vec().into_iter().map(|x| x as f32).collect::<Vec<f32>>()).collect();
    let x = DenseMatrix::<f32>::from_2d_vec(&x_data);

    let nb = CategoricalNB::fit(&x, &labels.into_raw_vec().into_iter().map(|x| x as f32).collect::<Vec<f32>>(), Default::default()).unwrap();
    fs::write("./CategoricalNBModel.bin", &serde_json::to_string(&nb)?).ok();

    let x_data: Vec<Vec<f32>> = test_records.outer_iter().map(|row| row.to_vec().into_iter().map(|x| x as f32).collect::<Vec<f32>>()).collect();
    let x = DenseMatrix::<f32>::from_2d_vec(&x_data);
    let test_ds = DatasetView::new(test_records.view(), test_labels.view());

    let prediction = nb.predict(&x).unwrap();

    let usize_vec: Vec<usize> = prediction.iter().map(|&x| x as usize).collect();
    let prediction = ArrayBase::<OwnedRepr<usize>, Ix1>::from_vec(usize_vec);

    // Displaying predictions
    println!("Predictions:");
    let num_predictions = 5;
    for i in 0..num_predictions {
        let prediction = prediction.index_axis(Axis(0), i);
        let true_label = test_labels[i];
        let text = test_texts[i].to_owned();
        println!("Text: {}, Prediction: {}, True Label: {}", text, prediction, true_label);
    }
    println!();

    let cm = prediction
        .confusion_matrix(&test_ds)
        .unwrap();
    // 0.9944
    let accuracy = cm.f1_score();
    println!("The fitted model has a training f1 score of {}", accuracy);


    Ok(())
}
