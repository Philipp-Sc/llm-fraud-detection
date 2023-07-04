use linfa::{DatasetBase, DatasetView};
use linfa_reduction::Pca;
use linfa::prelude::Fit;
use linfa::prelude::Predict;
use ndarray::{Array2, ArrayBase, ArrayView1};


pub fn feature_importance(x_dataset_shuffled: &Vec<Vec<f64>>, y_dataset_shuffled: &Vec<f64>) -> anyhow::Result<()> {

    let rows = x_dataset_shuffled.len();
    let cols = x_dataset_shuffled[0].len();
    let data_array = Array2::from_shape_vec((rows, cols), x_dataset_shuffled.into_iter().flatten().map(|&x| x).collect()).unwrap();
    let label_array = ArrayView1::from(&y_dataset_shuffled);
    let dataset = DatasetView::new(data_array.view(), label_array.view());
    let embedding = Pca::params(1)
        .fit(&dataset).unwrap();

    let explained_variance = embedding.explained_variance_ratio();

    println!("{:?}",explained_variance);

// reduce dimensionality of the dataset
    //let dataset = embedding.predict(dataset);
    Ok(())
}
