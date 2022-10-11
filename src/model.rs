use anyhow::Result;
use lightgbm as lgb;
use serde_json;

use crate::feature::FeatureData;

pub trait Model {
    fn train(
        &mut self,
        feature_data: &[FeatureData],
        label: &[f32],
        params: &serde_json::Value,
    ) -> Result<()>;
    fn predict(&self, feature_data: &[FeatureData]) -> Result<Vec<f64>>;
}

pub struct LightGBMModel {
    booster: Option<lgb::Booster>,
    feature_names: Option<Vec<String>>,
}

impl LightGBMModel {
    pub fn new() -> Self {
        Self {
            booster: None,
            feature_names: None,
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        self.booster.as_ref().unwrap().save_file(path)?;
        Ok(())
    }

    pub fn feature_names(&self) -> Result<Option<Vec<String>>> {
        Ok(self.feature_names.clone())
    }

    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        let feature_importances = self.booster.as_ref().unwrap().feature_importance()?;
        Ok(feature_importances)
    }
}

impl Default for LightGBMModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for LightGBMModel {
    fn train(
        &mut self,
        feature_data: &[FeatureData],
        label: &[f32],
        params: &serde_json::Value,
    ) -> Result<()> {
        let data = feature_data
            .iter()
            .map(|feature_data| feature_data.features.clone())
            .collect::<Vec<_>>();
        let dataset = lgb::Dataset::from_mat(data, label.to_vec()).unwrap();
        let booster = lgb::Booster::train(dataset, params).unwrap();
        self.booster = Some(booster);

        let names = feature_data
            .iter()
            .map(|feature_data| feature_data.names.clone())
            .collect::<Vec<_>>();
        self.feature_names = Some(names[0].clone());
        Ok(())
    }

    fn predict(&self, feature_data: &[FeatureData]) -> Result<Vec<f64>> {
        let data = feature_data
            .iter()
            .map(|feature_data| feature_data.features.clone())
            .collect::<Vec<_>>();
        let result = self.booster.as_ref().unwrap().predict(data)?;
        // NOTE: `result` is a vector [[n_rows]] because of binary classification.
        // ref: https://github.com/vaaaaanquish/lightgbm-rs/blob/fdac51534170d6ff23d2628827d0d620128f4c1f/src/booster.rs#L94-L148
        Ok(result[0].clone())
    }
}