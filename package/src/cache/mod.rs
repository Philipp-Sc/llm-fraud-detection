use serde::{Deserialize, Serialize};
use sled::IVec;


pub struct HashValueStore(SledStore);

impl HashValueStore {

    pub fn new(tree: &sled::Db) -> Self {
        let sled_store = SledStore::new(tree.clone());
        HashValueStore(sled_store)
    }

    pub fn contains_hash(&self, hash: u64) -> anyhow::Result<bool> {
        self.0.contains_key(hash.to_be_bytes().to_vec())
    }

    pub fn get_item_by_hash<S>(&self, hash: u64) -> anyhow::Result<Option<S>>
        where
            S: for<'a> Deserialize<'a> + TryFrom<Vec<u8>>,
    {
        Ok(match self.0.get(hash.to_be_bytes().to_vec())?
        {
            Some(val) => Some(val.to_vec().try_into().map_err(|_| anyhow::anyhow!("try_into() failed"))?),
            None => None,
        })
    }
    pub fn insert_item<T>(&self, hash: u64, item: T) -> anyhow::Result<()>
        where
            T: Serialize,
            Vec<u8>: TryFrom<T>,
    {
        let value: Vec<u8> = item.try_into().map_err(|_| anyhow::anyhow!("try_into() failed"))?;
        self.0.insert(hash.to_be_bytes().to_vec(),value)
    }


}

pub struct SledStore {
    db: sled::Db,
}

impl SledStore {
    pub fn new(sled_db: sled::Db) -> Self {
        SledStore {
            db: sled_db,
        }
    }

    fn contains_key<K>(&self, key: K) -> anyhow::Result<bool>
        where
            K: AsRef<Vec<u8>>,
    {
        Ok(self.db.contains_key(key.as_ref())?)
    }

    fn get<K>(&self, key: K) -> sled::Result<Option<IVec>>
        where
            K: AsRef<Vec<u8>>,
    {
        Ok(self.db.get(key.as_ref())?)
    }

    fn insert<K, V>(&self, key: K, value: V) -> anyhow::Result<()>
        where
            K: AsRef<Vec<u8>>,
            IVec: From<V>,
    {
        let _ = self
            .db
            .insert(key.as_ref(), value)?;
        Ok(())
    }

    fn remove<S>(&self, key: S) -> anyhow::Result<Option<sled::IVec>>
        where
            S: AsRef<Vec<u8>>,
    {
        Ok(self.db.remove(key.as_ref())?)
    }

}
