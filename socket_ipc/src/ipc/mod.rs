use serde::{Deserialize, Serialize};

pub mod socket;

use socket::{client_send_request, Handler, spawn_socket_service};


pub fn client_send_rust_bert_fraud_detection_request(socket_path: &str, texts: Vec<String>) -> anyhow::Result<RustBertFraudDetectionResult> {
    println!("client_send_request initiating");
    client_send_request(socket_path,RustBertFraudDetectionRequest{texts})
}

#[derive(Serialize,Deserialize,Debug)]
pub struct RustBertFraudDetectionRequest {
    pub texts: Vec<String>,
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

