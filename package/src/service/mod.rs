use std::thread::JoinHandle;
use serde::{Deserialize, Serialize};

#[cfg(feature = "server")]
use crate::fraud_probabilities;
use crate::service::socket::{client_send_request, Handler, spawn_socket_service};

pub mod socket;


#[cfg(feature = "server")]
pub fn spawn_rust_bert_fraud_detection_socket_service(socket_path: &str) -> JoinHandle<()> {
    println!("spawn_socket_service startup");
    let task = spawn_socket_service(socket_path,Box::new(RustBertFraudHandler{}) as Box<dyn Handler + Send>);
    println!("spawn_socket_service ready");
    task
}

pub fn client_send_rust_bert_fraud_detection_request(socket_path: &str, texts: Vec<String>) -> anyhow::Result<RustBertFraudDetectionResult> {
    println!("client_send_request initiating");
    client_send_request(socket_path,RustBertFraudDetectionRequest{texts})
}

#[derive(Serialize,Deserialize,Debug)]
pub struct RustBertFraudDetectionRequest {
    pub texts: Vec<String>,
}

#[cfg(feature = "server")]
impl RustBertFraudDetectionRequest {
    fn process(&self) -> anyhow::Result<RustBertFraudDetectionResult>{
        let fraud_probabilities: Vec<f64> = fraud_probabilities(&self.texts.iter().map(|x| x.as_str()).collect::<Vec<&str>>()[..])?;
        Ok(RustBertFraudDetectionResult{fraud_probabilities})
    }
}
impl TryFrom<Vec<u8>> for RustBertFraudDetectionRequest {
    type Error = anyhow::Error;
    fn try_from(item: Vec<u8>) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(&item[..])?)
    }
}

impl TryFrom<RustBertFraudDetectionRequest> for Vec<u8> {
    type Error = anyhow::Error;
    fn try_from(item: RustBertFraudDetectionRequest) -> anyhow::Result<Self> {
        Ok(bincode::serialize(&item)?)
    }
}

#[derive(Serialize,Deserialize,Debug)]
pub struct RustBertFraudDetectionResult {
    pub fraud_probabilities: Vec<f64>,
}

impl TryFrom<Vec<u8>> for RustBertFraudDetectionResult {
    type Error = anyhow::Error;
    fn try_from(item: Vec<u8>) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(&item[..])?)
    }
}

impl TryFrom<RustBertFraudDetectionResult> for Vec<u8> {
    type Error = anyhow::Error;
    fn try_from(item: RustBertFraudDetectionResult) -> anyhow::Result<Self> {
        Ok(bincode::serialize(&item)?)
    }
}

#[cfg(feature = "server")]
pub struct RustBertFraudHandler
{
}

#[cfg(feature = "server")]
impl Handler for RustBertFraudHandler
{
    fn process(&self, bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {

        let request: RustBertFraudDetectionRequest = bytes.try_into()?;
        let result: Vec<u8> = request.process()?.try_into()?;
        Ok(result)
    }
}