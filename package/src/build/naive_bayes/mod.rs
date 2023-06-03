use linfa_preprocessing::CountVectorizer;

pub fn test_this() {
    let vectorizer = CountVectorizer::params()
        .fit_files(&training_filenames, ISO_8859_1, Strict)
        .unwrap();
    println!(
        "We obtain a vocabulary with {} entries",
        vectorizer.nentries()
    );

    println!();
    println!(
        "Now let's generate a matrix containing the tf-idf value of each entry in each document"
    );
// Transforming gives a sparse dataset, we make it dense in order to be able to fit the Naive Bayes model
    let training_records = vectorizer
        .transform_files(&training_filenames, ISO_8859_1, Strict)
        .to_dense();
// Currently linfa only allows real valued features so we have to transform the integer counts to floats
    let training_records = training_records.mapv(|c| c as f32);

    println!(
        "We obtain a {}x{} matrix of counts for the vocabulary entries",
        training_records.dim().0,
        training_records.dim().1
    );
}