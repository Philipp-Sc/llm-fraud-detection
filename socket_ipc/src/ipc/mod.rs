use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

pub mod socket;

use socket::{client_send_request, Handler, spawn_socket_service};


pub fn client_send_rust_bert_fraud_detection_request(socket_path: &str, texts: Vec<String>) -> anyhow::Result<RustBertFraudDetectionResult> {
    println!("client_send_request initiating");
    client_send_request(socket_path,RustBertFraudDetectionRequest{texts})
}

#[derive(Serialize,Deserialize,Debug,Hash,Clone)]
pub struct RustBertFraudDetectionRequest {
    pub texts: Vec<String>,
}
impl RustBertFraudDetectionRequest {
    pub fn get_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
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

#[derive(Serialize,Deserialize,Debug,Clone)]
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

