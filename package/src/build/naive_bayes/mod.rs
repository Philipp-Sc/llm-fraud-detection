use linfa_preprocessing::CountVectorizer;
use linfa_bayes::{GaussianNbParams, GaussianNbValidParams, Result};
use linfa::prelude::*;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use ndarray::Axis;
use linfa::dataset::Labels;
use std::collections::HashMap;

pub fn update_naive_bayes_model(x_dataset: Vec<String>, y_dataset: Vec<i32>,test_x_dataset: Vec<String>, test_y_dataset: Vec<i32>) ->  anyhow::Result<()> {

    let texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((x_dataset.len(),), x_dataset).unwrap();
    let labels: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((y_dataset.len(),), y_dataset.into_iter().map(|x| x as usize).collect()).unwrap();

    let test_texts: ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((test_x_dataset.len(),), test_x_dataset).unwrap();
    let test_labels: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = ArrayBase::from_shape_vec((test_y_dataset.len(),), test_y_dataset.into_iter().map(|x| x as usize).collect()).unwrap();

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


    let vectorizer = CountVectorizer::params().fit(&texts).unwrap();

    println!(
        "We obtain a vocabulary with {} entries",
        vectorizer.nentries()
    );

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

    // Displaying sample entries of the training data
    println!("Sample entries of the training data:");
    let num_samples = 5;
    for i in 0..num_samples {
        let sample_entry = training_records.index_axis(Axis(0), i);
        let label = labels[i];
        let text = texts[i].to_owned();
        println!("Text: {}, Data: {:?}, Label: {}", text, sample_entry, label);
    }

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


    // Displaying sample entries of the test data
    println!("Sample entries of the test data:");
    for i in 0..num_samples {
        let sample_entry = test_records.index_axis(Axis(0), i);
        let label = test_labels[i];
        let text = test_texts[i].to_owned();
        println!("Text: {}, Data: {:?}, Label: {}", text, sample_entry, label);
    }

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

    let cm = prediction
        .confusion_matrix(&test_ds)
        .unwrap();
    // 0.9944
    let accuracy = cm.f1_score();
    println!("The fitted model has a training f1 score of {}", accuracy);

    Ok(())
}