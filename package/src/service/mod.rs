use std::thread::JoinHandle;
use rust_bert_fraud_detection_socket_ipc::ipc::{RustBertFraudDetectionRequest, RustBertFraudDetectionResult};
use rust_bert_fraud_detection_socket_ipc::ipc::socket::{Handler, spawn_socket_service};
use crate::fraud_probabilities;

use lazy_static::lazy_static;
use crate::cache::HashValueStore;


lazy_static!{
   static ref FRAUD_DETECTION_STORE: HashValueStore = load_store("./tmp/rust_bert_fraud_detection_sled_db");
}

pub fn load_store(path: &str) -> HashValueStore {
    let db: sled::Db = sled::Config::default()
        .path(path)
        .cache_capacity(1024 * 1024 * 1024 / 2)
        .flush_every_ms(Some(1000))
        .open().unwrap();
    HashValueStore::new(&db)
}

pub fn spawn_rust_bert_fraud_detection_socket_service(socket_path: &str) -> JoinHandle<()> {
    println!("Starting RustBert Fraud Detection socket service on {}...", socket_path);
    let task = spawn_socket_service(socket_path,Box::new(RustBertFraudHandler{}) as Box<dyn Handler + Send>);
    println!("RustBert Fraud Detection socket service is ready and listening for incoming requests.");
    task
}
pub struct RustBertFraudHandler
{
}

impl Handler for RustBertFraudHandler
{
    fn process(&self, bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {

        println!("Processing request for RustBert Fraud Detection...");

        let request: RustBertFraudDetectionRequest = bytes.try_into()?;

        let hash = request.get_hash();

        if FRAUD_DETECTION_STORE.contains_hash(hash)? {
            let val = FRAUD_DETECTION_STORE.get_item_by_hash::<RustBertFraudDetectionResult>(hash)?.unwrap();
            let result: Vec<u8> = val.try_into()?;
            println!("Found cached result for hash {}.", hash);
            Ok(result)
        } else {
            let fraud_probabilities: Vec<f64> = fraud_probabilities(&request.texts.iter().map(|x| x.as_str()).collect::<Vec<&str>>()[..])?;
            let val = RustBertFraudDetectionResult { fraud_probabilities };

            FRAUD_DETECTION_STORE.insert_item(hash,val.clone()).ok();

            let result: Vec<u8> = val.try_into()?;
            println!("Calculated new result for hash {}.", hash);
            Ok(result)
        }
    }
}