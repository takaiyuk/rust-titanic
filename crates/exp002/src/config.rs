use serde_json::json;

pub trait AbstractConfig {
    fn new() -> Self;
    // fn load(&mut self, path: &str) -> Result<()>;
    // fn save(&self, path: &str) -> Result<()>;
}

pub struct Config {
    pub params: serde_json::Value,
}

impl AbstractConfig for Config {
    fn new() -> Self {
        Self {
            params: json!({
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            }),
        }
    }
}
