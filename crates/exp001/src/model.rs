use anyhow::Result;
use lightgbm as lgb;
use serde_json;

use crate::feature::FeatureData;

pub trait AbstractModel {
    fn train(
        &mut self,
        feature_data: &[FeatureData],
        label: &[f32],
        params: &serde_json::Value,
    ) -> Result<()>;
    fn predict(&self, feature_data: &[FeatureData]) -> Result<Vec<f64>>;
    fn save(&self, path: &str) -> Result<()>;
}

pub trait AbstractGBDTModel: AbstractModel {
    fn feature_names(&self) -> Result<Option<Vec<String>>>;
    fn feature_importances(&self) -> Result<Vec<f64>>;
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
}

impl Default for LightGBMModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractModel for LightGBMModel {
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
        Ok(result[0].clone())
    }

    fn save(&self, path: &str) -> Result<()> {
        self.booster.as_ref().unwrap().save_file(path)?;
        Ok(())
    }
}

impl AbstractGBDTModel for LightGBMModel {
    fn feature_names(&self) -> Result<Option<Vec<String>>> {
        Ok(self.feature_names.clone())
    }

    fn feature_importances(&self) -> Result<Vec<f64>> {
        Ok(self.booster.as_ref().unwrap().feature_importance()?)
    }
}
