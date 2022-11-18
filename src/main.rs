use std::path::Path;

use anyhow::Result;

use rust_titanic::config::{AbstractConfig, Config};
use rust_titanic::feature::FeatureTransformer;
use rust_titanic::kfold::StratifiedKFold;
use rust_titanic::loader::{load_test_data, load_train_data};
use rust_titanic::metrics::accuracy;
use rust_titanic::model::LightGBMModel;
use rust_titanic::runner::{AbstractRunner, LightGBMRunner};
use rust_titanic::submission::generate_submission;

const TRAIN_DATA_PATH: &str = "input/titanic/train.csv";
const TEST_DATA_PATH: &str = "input/titanic/test.csv";
const SAMPLE_SUBMISSION_DATA_PATH: &str = "input/titanic/gender_submission.csv";
const SUBMISSION_PATH: &str = "output/submissions/submission.csv";

fn calc_vec_mean(vec_vec: Vec<Vec<f64>>) -> Vec<f64> {
    let mut vec_mean: Vec<f64> = vec![];
    for i in 0..vec_vec[0].len() {
        let mut sum = 0.0;
        for feature_importance in vec_vec.iter() {
            sum += feature_importance[i];
        }
        vec_mean.push(sum / vec_vec.iter().len() as f64);
    }
    vec_mean
}

fn convert_probability_to_label(probabilities: Vec<f64>) -> Vec<i32> {
    probabilities.iter().map(|x| i32::from(*x > 0.5)).collect()
}

fn main() -> Result<()> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let train = load_train_data(project_root.join(TRAIN_DATA_PATH))?;
    let labels = train
        .iter()
        .map(|input_data| input_data.survived.unwrap())
        .collect::<Vec<u32>>();
    let test = load_test_data(project_root.join(TEST_DATA_PATH))?;

    let mut runner = LightGBMRunner::new(
        Config::new(),
        Box::new(FeatureTransformer {}),
        Box::new(StratifiedKFold::new(5, true, Some(42))),
        LightGBMModel::new(),
    );
    let prediction_results = runner.run_cv(&train, labels, &test)?;

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
    println!("CV Accuracy: {:?}", acc);

    let feature_importances_mean = calc_vec_mean(feature_importances);
    println!("Feature names: {:?}", &prediction_results[0].feature_names);
    println!("Feature importances: {:?}", &feature_importances_mean);

    let pred_test_mean = calc_vec_mean(pred_tests);
    let pred_test_label = convert_probability_to_label(pred_test_mean);
    generate_submission(
        pred_test_label,
        SAMPLE_SUBMISSION_DATA_PATH,
        SUBMISSION_PATH,
    )?;

    Ok(())
}
