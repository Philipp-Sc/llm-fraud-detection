use ndarray::{Array, Array1};
use tch::{nn, nn::Module, nn::OptimizerConfig, Kind, Tensor, Device};
use crate::build::classification::calculate_metrics;

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
        let hidden_size1 = input_size * 4;
        let hidden_size2 = hidden_size1 * 2;
        let hidden_size3 = hidden_size1;
        let hidden_size4 = hidden_size1 * 2;

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

    let inputs: Vec<Vec<f32>> = x_dataset.into_iter().map(|x| x.into_iter().map(|y| *y as f32).collect()).collect();
    let input_tensor = Tensor::from_slice(&inputs.concat()).reshape(&[-1, inputs[0].len() as i64 ]);
    let targets: Vec<Vec<f32>> = y_dataset.into_iter().map(|x| vec![(if x > &1.0 {1.0}else if x < &0.0 {0.0}else{*x}) as f32] ).collect();
    let target_tensor = Tensor::from_slice(&targets.concat()).reshape(&[-1, targets[0].len() as i64]);

    let predictor = Predictor::new(&vs.root(), inputs[0].len() as i64);
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();

    for epoch in 0..500 {
        let output = predictor.forward(&input_tensor);
        //let loss = output.mse_loss(&target_tensor, tch::Reduction::Mean);
        let loss = output.binary_cross_entropy(&output, Some(&target_tensor), tch::Reduction::Mean);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        let loss_val = loss.double_value(&[]);
        if epoch % 10 == 0 {
            println!("Epoch: {:?}, Loss: {:?}", epoch, loss_val);
        }
    }
    predictor
}


pub fn test_nn(predictor: &Predictor, x_dataset: &Vec<Vec<f64>>, y_dataset: &Vec<f64>)  {

    let x_len = x_dataset[0].len() as i64;
    let input_tensor = Tensor::from_slice(&x_dataset.into_iter().flatten().map(|x| *x as f32).collect::<Vec<f32>>()).reshape(&[-1, x_len ]);

    let predictions: Vec<Vec<f32>> = predictor.forward(&input_tensor).try_into().unwrap();
    let predictions: Vec<f64> = predictions.iter().flatten().map(|x| *x as f64).collect();
    let thresholds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    calculate_metrics(y_dataset,&predictions,&thresholds);
}