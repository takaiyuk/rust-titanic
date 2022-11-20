use anyhow::Result;

use crate::loader::InputData;

#[derive(Debug, Clone)]
pub struct FeatureData {
    pub names: Vec<String>,
    pub features: Vec<f64>,
}

pub trait AbstractFeatureTransformer {
    fn fit(&self, input_data: &[InputData]) -> Result<()>;
    fn transform(&self, input_data: &[InputData]) -> Result<Vec<FeatureData>>;
}

pub struct FeatureTransformer {}

impl AbstractFeatureTransformer for FeatureTransformer {
    fn fit(&self, _input_data: &[InputData]) -> Result<()> {
        Ok(())
    }
    fn transform(&self, input_data: &[InputData]) -> Result<Vec<FeatureData>> {
        let features = input_data
            .iter()
            .map(|input_data| {
                let names = vec![
                    "pclass",
                    "sex",
                    "age",
                    "sibsp",
                    "parch",
                    "fare",
                    "embarked",
                    "title",
                    "family_size",
                ]
                .iter()
                .map(|name| name.to_string())
                .collect();
                let features = vec![
                    input_data.pclass.unwrap_or(-1) as f64,
                    input_data.sex.as_ref().map_or_else(|| -1, |s| *s as i32) as f64,
                    input_data.age.unwrap_or(-1.0),
                    input_data.sibsp.unwrap_or(-1) as f64,
                    input_data.parch.unwrap_or(-1) as f64,
                    input_data.fare.unwrap_or(-1.0),
                    input_data
                        .embarked
                        .as_ref()
                        .map_or_else(|| -1, |s| *s as i32) as f64,
                    input_data.name.as_ref().map_or_else(
                        || -1,
                        |s| {
                            if s.contains("Mr.") {
                                0
                            } else if s.contains("Mrs.") {
                                1
                            } else if s.contains("Miss.") {
                                2
                            } else if s.contains("Master.") {
                                3
                            } else {
                                4
                            }
                        },
                    ) as f64,
                    (input_data.sibsp.unwrap_or(0) + input_data.parch.unwrap_or(0) + 1) as f64,
                ];
                FeatureData { names, features }
            })
            .collect();
        Ok(features)
    }
}
