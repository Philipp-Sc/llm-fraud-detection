use ndarray::{Array, Array1};
use tch::{nn, nn::Module, nn::OptimizerConfig, Kind, Tensor, Device};
use crate::build::classification::calculate_metrics;

use rand::seq::SliceRandom;
use rand::thread_rng;
// Define your predictor model
#[derive(Debug)]
pub struct Predictor {
    linear1: nn::Linear,
    linear2: nn::Linear,
    linear3: nn::Linear,
    linear4: nn::Linear,
    linear5: nn::Linear,
    batch_norm1: nn::BatchNorm,
    batch_norm2: nn::BatchNorm,
    batch_norm3: nn::BatchNorm,
    batch_norm4: nn::BatchNorm,
    dropout_p: f64,
}

impl Predictor {
    fn new(vs: &nn::Path, input_size: i64) -> Self {
        let factor = 2;
        let hidden_size1 = 128*factor;
        let hidden_size2 = 64*factor;
        let hidden_size3 = 32*factor;
        let hidden_size4 = 16*factor;


        let linear1 = nn::linear(vs, input_size, hidden_size1, Default::default());
        let linear2 = nn::linear(vs, hidden_size1, hidden_size2, Default::default());
        let linear3 = nn::linear(vs, hidden_size2, hidden_size3, Default::default());
        let linear4 = nn::linear(vs, hidden_size3, hidden_size4, Default::default());
        let linear5 = nn::linear(vs, hidden_size4, 1, Default::default());

        let batch_norm1 = nn::batch_norm1d(vs, hidden_size1, Default::default());
        let batch_norm2 = nn::batch_norm1d(vs, hidden_size2, Default::default());
        let batch_norm3 = nn::batch_norm1d(vs, hidden_size3, Default::default());
        let batch_norm4 = nn::batch_norm1d(vs, hidden_size4, Default::default());

        let dropout_p = 0.5;

        Predictor {
            linear1,
            linear2,
            linear3,
            linear4,
            linear5,
            batch_norm1,
            batch_norm2,
            batch_norm3,
            batch_norm4,
            dropout_p,
        }
    }
}

impl Module for Predictor {
    fn forward(&self, input: &Tensor) -> Tensor {
        let output = input
            .apply(&self.linear1)
            .apply_t(&self.batch_norm1, true)
            .relu()
            .dropout(self.dropout_p, true)
            .apply(&self.linear2)
            .apply_t(&self.batch_norm2, true)
            .relu()
            .dropout(self.dropout_p, true)
            .apply(&self.linear3)
            .apply_t(&self.batch_norm3, true)
            .relu()
            .dropout(self.dropout_p, true)
            .apply(&self.linear4)
            .apply_t(&self.batch_norm4, true)
            .relu()
            .dropout(self.dropout_p, true)
            .apply(&self.linear5)
            .sigmoid();
        output
    }
}


pub fn train_nn(x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>) -> Predictor {
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let x_len = x_dataset[0].len() as i64;
    let inputs: Vec<Vec<f32>> = x_dataset.into_iter().map(|y| y.into_iter().map(|x| *x as f32).collect()).collect();
    let targets: Vec<Vec<f32>> = y_dataset.into_iter().map(|x| vec![*x as f32] ).collect();

    let predictor = Predictor::new(&vs.root(), x_len);
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let mut rng = thread_rng();  // Initialize the random number generator

    let batch_size = 256;  // Define the desired batch size

    let input_chunks = inputs.chunks(batch_size);
    let target_chunks = targets.chunks(batch_size);

    for epoch in 0..100 {
        let mut epoch_loss = 0.0;

        let mut shuffled_chunks = input_chunks.clone().zip(target_chunks.clone()).collect::<Vec<_>>();
        let shuffled_chunks = &mut shuffled_chunks[..];
        shuffled_chunks.shuffle(&mut rng);

        for (input_chunk, target_chunk) in shuffled_chunks {

            let input_tensor = Tensor::from_slice(&input_chunk.concat()).reshape(&[-1, x_len]).to_device(device);
            let target_tensor = Tensor::from_slice(&target_chunk.concat()).reshape(&[-1, targets[0].len() as i64]).to_device(device);

            let output = predictor.forward(&input_tensor);
            //let loss = output.mse_loss(&target_tensor, tch::Reduction::Mean);
            let loss = output.binary_cross_entropy::<Tensor>(&target_tensor, None, tch::Reduction::Mean);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            epoch_loss = loss.double_value(&[]);
        }

        println!("Epoch: {:?}, Loss: {:?}", epoch, epoch_loss);

    }
    predictor
}


pub fn test_nn(predictor: &Predictor, x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>)  {
    let device = tch::Device::cuda_if_available();

    let x_len = x_dataset[0].len() as i64;
    let input_tensor = Tensor::from_slice(&x_dataset.into_iter().flatten().map(|x| *x as f32).collect::<Vec<f32>>()).reshape(&[-1, x_len ]).to_device(device);

    let predictions: Vec<Vec<f32>> = predictor.forward(&input_tensor).try_into().unwrap();
    let predictions: Vec<f64> = predictions.iter().flatten().map(|x| *x as f64).collect();
    let thresholds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    calculate_metrics(y_dataset,&predictions,&thresholds);
}



pub fn z_score_normalize(x_dataset: &Vec<Vec<f64>>, mean_std_dev: Option<(Vec<f64>,Vec<f64>)>) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {

    // Calculate mean and standard deviation of x_dataset
    let (mean, std_dev) = if let Some(tuple) = mean_std_dev {
        tuple
    }else{
        calculate_mean_std_dev(x_dataset)
    };

    // Apply z-score normalization
    let normalized_x = x_dataset
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, val)| (val - mean[i]) / std_dev[i])
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    (normalized_x, mean, std_dev)
}


fn calculate_mean_std_dev(x_dataset: &Vec<Vec<f64>>) -> (Vec<f64>, Vec<f64>) {
    let num_features = x_dataset[0].len();
    let num_samples = x_dataset.len() as f64;

    let mean: Vec<f64> = (0..num_features)
        .map(|i| {
            let sum: f64 = x_dataset.iter().map(|row| row[i]).sum();
            sum / num_samples
        })
        .collect();

    let std_dev: Vec<f64> = (0..num_features)
        .map(|i| {
            let squared_diff: f64 = x_dataset
                .iter()
                .map(|row| (row[i] - mean[i]).powi(2))
                .sum();
            (squared_diff / (num_samples - 1.0)).sqrt()
        })
        .collect();

    (mean, std_dev)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_mean_std_dev() {
        let x_dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let (mean, std_dev) = calculate_mean_std_dev(&x_dataset);

        println!("{:?}",(&mean, &std_dev));

        assert_eq!(mean, vec![2.0, 3.0, 4.0]);
        assert_eq!(std_dev, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_z_score_normalize() {
        let x_dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let (normalized_x, mean, std_dev) = z_score_normalize(&x_dataset, None);

        assert_eq!(normalized_x, vec![
            vec![-1.0, -1.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ]);
        assert_eq!(mean, vec![2.0, 3.0, 4.0]);
        assert_eq!(std_dev, vec![1.0, 1.0, 1.0]);
    }
}