use anyhow::Result;
use serde_json;
use xgboost as xgb;

use crate::feature::FeatureData;

pub trait AbstractModel {
    fn train(
        &mut self,
        train_feature_data: &[FeatureData],
        valid_feature_data: &[FeatureData],
        train_label: &[f32],
        valid_label: &[f32],
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
        train_feature_data: &[FeatureData],
        valid_feature_data: &[FeatureData],
        train_label: &[f32],
        valid_label: &[f32],
        _params: &serde_json::Value,
    ) -> Result<()> {
        let mut x_train = Vec::new();
        for data in train_feature_data {
            for feature in &data.features {
                x_train.push(*feature as f32);
            }
        }
        let mut dtrain = xgb::DMatrix::from_dense(&x_train, train_feature_data.len()).unwrap();
        dtrain.set_labels(train_label).unwrap();

        let mut x_valid = Vec::new();
        for data in valid_feature_data {
            for feature in &data.features {
                x_valid.push(*feature as f32);
            }
        }
        let mut dvalid = xgb::DMatrix::from_dense(&x_valid, valid_feature_data.len()).unwrap();
        dvalid.set_labels(valid_label).unwrap();

        let learning_params = xgb::parameters::learning::LearningTaskParametersBuilder::default()
            .objective(xgb::parameters::learning::Objective::BinaryLogistic)
            .build()
            .unwrap();
        let tree_params = xgb::parameters::tree::TreeBoosterParametersBuilder::default()
            .max_depth(6)
            .eta(0.1)
            .build()
            .unwrap();
        let booster_params = xgb::parameters::BoosterParametersBuilder::default()
            .booster_type(xgb::parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let evaluation_sets = &[(&dtrain, "train"), (&dvalid, "valid")];
        let params = xgb::parameters::TrainingParametersBuilder::default()
            .dtrain(&dtrain) // dataset to train with
            .boost_rounds(100) // number of training iterations
            .booster_params(booster_params) // model parameters
            .evaluation_sets(Some(evaluation_sets)) // optional datasets to evaluate against in each iteration
            .build()
            .unwrap();

        let booster = xgb::Booster::train(&params).unwrap();
        self.booster = Some(booster);

        let names = train_feature_data
            .iter()
            .map(|feature_data| feature_data.names.clone())
            .collect::<Vec<_>>();
        self.feature_names = Some(names[0].clone());
        Ok(())
    }

    fn predict(&self, feature_data: &[FeatureData]) -> Result<Vec<f64>> {
        let mut x = Vec::new();
        for data in feature_data {
            for feature in &data.features {
                x.push(*feature as f32);
            }
        }
        let dmatrix = xgb::DMatrix::from_dense(&x, feature_data.len()).unwrap();

        let result = self.booster.as_ref().unwrap().predict(&dmatrix)?;
        Ok(result.iter().map(|v| *v as f64).collect())
    }

    fn save(&self, path: &str) -> Result<()> {
        self.booster.as_ref().unwrap().save(path)?;
        Ok(())
    }
}

impl AbstractGBDTModel for XGBoostModel {
    fn feature_names(&self) -> Result<Option<Vec<String>>> {
        Ok(self.feature_names.clone())
    }

    fn feature_importances(&self) -> Result<Vec<f64>> {
        Ok(Vec::new())
        // Ok(self.booster.as_ref().unwrap().feature_importance()?)
    }
}
