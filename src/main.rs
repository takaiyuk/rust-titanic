use std::path::Path;

use anyhow::Result;
use serde_json::json;

use rust_titanic::feature::{AbstractFeatureTransformer, FeatureTransformer};
use rust_titanic::kfold::{AbstractKFold, StratifiedKFold};
use rust_titanic::loader::{load_test, load_train};
use rust_titanic::metrics::accuracy;
use rust_titanic::model::{LightGBMModel, Model};
use rust_titanic::submission::generate_submission;

const TRAIN_DATA_PATH: &str = "input/titanic/train.csv";
const TEST_DATA_PATH: &str = "input/titanic/test.csv";
const MODEL_PATH_PREFIX: &str = "output/models";
const SAMPLE_SUBMISSION_DATA_PATH: &str = "input/titanic/gender_submission.csv";
const SUBMISSION_PATH: &str = "output/submissions/submission.csv";

#[derive(Debug, Clone)]
struct PredictionResult {
    valid_label: Vec<f64>,
    pred_valid: Vec<f64>,
    pred_test: Vec<f64>,
    feature_names: Vec<String>,
    feature_importances: Vec<f64>,
}

fn main() -> Result<()> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let train = load_train(project_root.join(TRAIN_DATA_PATH))?;
    let test = load_test(project_root.join(TEST_DATA_PATH))?;

    let labels = train
        .iter()
        .map(|input_data| input_data.survived.unwrap())
        .collect::<Vec<u32>>();
    let folds = StratifiedKFold::new(5, true, Some(42)).split(&train, labels);

    let mut prediction_results = vec![];
    for (n_fold, mut index) in folds.into_iter().enumerate() {
        println!("Fold {:?}", n_fold + 1);

        let mut train_fold = train.clone();
        let mut valid_fold = vec![];
        index.sort();
        index.reverse();
        for i in index {
            train_fold.remove(i);
            valid_fold.push(train[i].clone());
        }

        let feature_transformer = FeatureTransformer {};
        let train_feature = feature_transformer.transform(&train_fold)?;
        let valid_feature = feature_transformer.transform(&valid_fold)?;
        let train_label: Vec<f32> = train_fold
            .iter()
            .map(|x| x.survived.unwrap() as f32)
            .collect();
        let valid_label: Vec<f64> = valid_fold
            .iter()
            .map(|x| x.survived.unwrap() as f64)
            .collect();

        let params = json! {
           {
                "num_iterations": 100,
                "objective": "binary",
                "metric": "logloss",
                "seed": 42,
                "verbose": -1,
                "num_leaves": 31,
                "categorical_feature": vec![0, 1, 3, 4, 6, 7],
                "force_row_wise": true,
            }
        };
        let mut model = LightGBMModel::new();
        model.train(&train_feature, &train_label, &params)?;
        model.save(&format!("{}/fold{}.dat", MODEL_PATH_PREFIX, n_fold + 1))?;
        let feature_names = model.feature_names()?.unwrap();
        let feature_importances = model.feature_importances()?;

        let pred_valid = model.predict(&valid_feature)?;
        let acc = accuracy(&valid_label, &pred_valid)?;
        println!("Accuracy: {:?}", acc);

        let test_feature = feature_transformer.transform(&test)?;
        let pred_test = model.predict(&test_feature)?;

        let prediction_result = PredictionResult {
            valid_label,
            pred_valid,
            pred_test,
            feature_names,
            feature_importances,
        };
        prediction_results.push(prediction_result)
    }

    let mut valid_labels = vec![];
    let mut pred_valids = vec![];
    let mut pred_tests = vec![];
    let mut feature_importances = vec![];
    for r in prediction_results.iter() {
        valid_labels.extend(r.valid_label.iter().clone());
        pred_valids.extend(r.pred_valid.iter().clone());
        pred_tests.push(r.pred_test.clone());
        feature_importances.push(r.feature_importances.clone());
    }
    let acc = accuracy(&valid_labels, &pred_valids)?;
    println!("CV");
    println!("Accuracy: {:?}", acc);

    let mut feature_importances_mean: Vec<f64> = vec![];
    for i in 0..feature_importances[0].len() {
        let mut sum = 0.0;
        for feature_importance in feature_importances
            .iter()
            .take(feature_importances.iter().len())
        {
            sum += feature_importance[i];
        }
        feature_importances_mean.push(sum / feature_importances.iter().len() as f64);
    }
    println!("Feature names: {:?}", &prediction_results[0].feature_names);
    println!("Feature importances: {:?}", &feature_importances_mean);

    let mut pred_test_mean: Vec<f64> = vec![];
    for i in 0..pred_tests[0].len() {
        let mut sum = 0.0;
        for pred_test in pred_tests.iter().take(pred_tests.iter().len()) {
            sum += pred_test[i];
        }
        pred_test_mean.push(sum / pred_tests.iter().len() as f64);
    }

    let pred_test: Vec<i32> = pred_test_mean.iter().map(|x| i32::from(*x > 0.5)).collect();
    generate_submission(pred_test, SAMPLE_SUBMISSION_DATA_PATH, SUBMISSION_PATH)?;
    Ok(())
}
