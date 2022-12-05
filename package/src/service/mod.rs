use std::thread::JoinHandle;
use rust_bert_fraud_detection_socket_ipc::ipc::{RustBertFraudDetectionRequest, RustBertFraudDetectionResult};
use rust_bert_fraud_detection_socket_ipc::ipc::socket::{Handler, spawn_socket_service};
use crate::fraud_probabilities;

pub fn spawn_rust_bert_fraud_detection_socket_service(socket_path: &str) -> JoinHandle<()> {
    println!("spawn_socket_service startup");
    let task = spawn_socket_service(socket_path,Box::new(RustBertFraudHandler{}) as Box<dyn Handler + Send>);
    println!("spawn_socket_service ready");
    task
}
pub struct RustBertFraudHandler
{
}

impl Handler for RustBertFraudHandler
{
    fn process(&self, bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {

        let request: RustBertFraudDetectionRequest = bytes.try_into()?;

        let fraud_probabilities: Vec<f64> = fraud_probabilities(&request.texts.iter().map(|x| x.as_str()).collect::<Vec<&str>>()[..])?;

        let result: Vec<u8> = RustBertFraudDetectionResult{fraud_probabilities}.try_into()?;
        Ok(result)
    }
}