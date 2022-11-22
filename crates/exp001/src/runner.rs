use anyhow::Result;

use crate::config::Config;
use crate::feature::AbstractFeatureTransformer;
use crate::kfold::AbstractKFold;
use crate::loader::InputData;
use crate::metrics::accuracy;
use crate::model::AbstractGBDTModel;

const MODEL_PATH_PREFIX: &str = "output/models";

#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub score: f64,
    pub valid_label: Vec<f64>,
    pub pred_valid: Vec<f64>,
    pub pred_test: Vec<f64>,
    pub feature_names: Vec<String>,
    pub feature_importances: Vec<f64>,
}

pub trait AbstractRunner {
    fn run_cv(
        &mut self,
        train: &[InputData],
        labels: Vec<u32>,
        test: &[InputData],
    ) -> Result<Vec<PredictionResult>>;
}

pub struct LightGBMRunner {
    config: Config,
    feature_transformer: Box<dyn AbstractFeatureTransformer>,
    kfold: Box<dyn AbstractKFold>,
    model: Box<dyn AbstractGBDTModel>,
}

impl LightGBMRunner {
    pub fn new(
        config: Config,
        feature_transformer: Box<dyn AbstractFeatureTransformer>,
        kfold: Box<dyn AbstractKFold>,
        model: Box<dyn AbstractGBDTModel>,
    ) -> Self {
        Self {
            config,
            feature_transformer,
            kfold,
            model,
        }
    }

    fn run_fold(
        &mut self,
        fold: usize,
        train_fold: &[InputData],
        valid_fold: &[InputData],
        test: &[InputData],
    ) -> Result<PredictionResult> {
        let train_features = self.feature_transformer.transform(train_fold)?;
        let valid_features = self.feature_transformer.transform(valid_fold)?;
        let train_label: Vec<f32> = train_fold
            .iter()
            .map(|x| x.survived.unwrap() as f32)
            .collect();
        let valid_label: Vec<f64> = valid_fold
            .iter()
            .map(|x| x.survived.unwrap() as f64)
            .collect();

        self.model
            .train(&train_features, &train_label, &self.config.params)?;
        self.model
            .save(&format!("{}/fold{}.dat", MODEL_PATH_PREFIX, fold + 1))?;
        let pred_valid = self.model.predict(&valid_features)?;
        let score = accuracy(&valid_label, &pred_valid)?;
        println!("Accuracy: {:?}", score);
        let feature_names = self.model.feature_names()?.unwrap();
        let feature_importances = self.model.feature_importances()?;

        let test_feature = self.feature_transformer.transform(test)?;
        let pred_test = self.model.predict(&test_feature)?;

        let prediction_result = PredictionResult {
            score,
            valid_label,
            pred_valid,
            pred_test,
            feature_names,
            feature_importances,
        };
        Ok(prediction_result)
    }
}

impl AbstractRunner for LightGBMRunner {
    fn run_cv(
        &mut self,
        train: &[InputData],
        labels: Vec<u32>,
        test: &[InputData],
    ) -> Result<Vec<PredictionResult>> {
        let folds = self.kfold.split(train, labels);

        let mut prediction_results = vec![];
        for (n_fold, mut fold_index) in folds.into_iter().enumerate() {
            println!("Fold {:?}", n_fold + 1);

            let mut train_fold = train.to_vec();
            let mut valid_fold = vec![];
            fold_index.sort();
            fold_index.reverse();
            for i in fold_index {
                train_fold.remove(i);
                valid_fold.push(train[i].clone());
            }
            valid_fold.reverse();

            let prediction_result = self.run_fold(n_fold, &train_fold, &valid_fold, test)?;
            prediction_results.push(prediction_result)
        }
        Ok(prediction_results)
    }
}
