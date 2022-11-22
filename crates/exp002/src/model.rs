use anyhow::Result;
use serde_json;
use xgboost as xgb;

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

pub struct XGBoostModel {
    booster: Option<xgb::Booster>,
    feature_names: Option<Vec<String>>,
}

impl XGBoostModel {
    pub fn new() -> Self {
        Self {
            booster: None,
            feature_names: None,
        }
    }
}

impl Default for XGBoostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractModel for XGBoostModel {
    fn train(
        &mut self,
        feature_data: &[FeatureData],
        label: &[f32],
        params: &serde_json::Value,
    ) -> Result<()> {
        todo!()
        // let data = feature_data
        //     .iter()
        //     .map(|feature_data| feature_data.features.clone())
        //     .collect::<Vec<_>>();
        // let dataset = xgb::Dataset::from_mat(data, label.to_vec()).unwrap();
        // let booster = xgb::Booster::train(dataset, params).unwrap();
        // self.booster = Some(booster);

        // let names = feature_data
        //     .iter()
        //     .map(|feature_data| feature_data.names.clone())
        //     .collect::<Vec<_>>();
        // self.feature_names = Some(names[0].clone());
        // Ok(())
    }

    fn predict(&self, feature_data: &[FeatureData]) -> Result<Vec<f64>> {
        todo!()
        // let data = feature_data
        //     .iter()
        //     .map(|feature_data| feature_data.features.clone())
        //     .collect::<Vec<_>>();
        // let result = self.booster.as_ref().unwrap().predict(data)?;
        // // NOTE: `result` is a vector [[n_rows]] because of binary classification.
        // // ref: https://github.com/vaaaaanquish/lightgbm-rs/blob/fdac51534170d6ff23d2628827d0d620128f4c1f/src/booster.rs#L94-L148
        // Ok(result[0].clone())
    }

    fn save(&self, path: &str) -> Result<()> {
        todo!()
        // self.booster.as_ref().unwrap().save_file(path)?;
        // Ok(())
    }
}

impl AbstractGBDTModel for XGBoostModel {
    fn feature_names(&self) -> Result<Option<Vec<String>>> {
        todo!()
        // Ok(self.feature_names.clone())
    }

    fn feature_importances(&self) -> Result<Vec<f64>> {
        todo!()
        // let feature_importances = self.booster.as_ref().unwrap().feature_importance()?;
        // Ok(feature_importances)
    }
}
