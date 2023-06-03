use linfa_preprocessing::CountVectorizer;
use linfa_bayes::{GaussianNbParams, GaussianNbValidParams, Result};
use linfa::prelude::*;
use ndarray::array;

pub fn test_this() {

    let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
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

    let labels = array![1, 1, 1, 0];

    let ds = DatasetView::new(training_records.view(), labels.view());

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

    let training_prediction = model.predict(&ds);

    let cm = training_prediction
        .confusion_matrix(&ds)
        .unwrap();
    // 0.9944
    let accuracy = cm.f1_score();
    println!("The fitted model has a training f1 score of {}", accuracy);

}