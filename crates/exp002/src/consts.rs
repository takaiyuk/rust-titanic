use once_cell::sync::Lazy;

const CARGO_PKG_NAME: &str = env!("CARGO_PKG_NAME");
pub const TRAIN_DATA_PATH: &str = "../../input/titanic/train.csv";
pub const TEST_DATA_PATH: &str = "../../input/titanic/test.csv";
pub const SAMPLE_SUBMISSION_DATA_PATH: &str = "../../input/titanic/gender_submission.csv";
pub static SUBMISSION_PATH: Lazy<String> =
    Lazy::new(|| format!("../../output/{}/submissions/submission.csv", CARGO_PKG_NAME));
pub static MODEL_PATH_PREFIX: Lazy<String> =
    Lazy::new(|| format!("../../output/{}/models", CARGO_PKG_NAME));
