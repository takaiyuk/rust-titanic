use rand::prelude::*;
use std::collections::HashSet;

use crate::loader::InputData;

pub trait AbstractKFold {
    fn split(&self, data: &[InputData], labels: Vec<u32>) -> Vec<Vec<usize>>;
}

pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl KFold {
    pub fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_splits,
            shuffle,
            random_state,
        }
    }
}

impl AbstractKFold for KFold {
    fn split(&self, data: &[InputData], _labels: Vec<u32>) -> Vec<Vec<usize>> {
        let rng = rand::thread_rng();
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_rng(rng).unwrap(),
            };
            indices.shuffle(&mut rng);
        }
        let mut folds = vec![vec![]; self.n_splits];
        for (i, &index) in indices.iter().enumerate() {
            folds[i % self.n_splits].push(index);
        }
        folds
    }
}

pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl StratifiedKFold {
    pub fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_splits,
            shuffle,
            random_state,
        }
    }
}

impl AbstractKFold for StratifiedKFold {
    fn split(&self, _data: &[InputData], labels: Vec<u32>) -> Vec<Vec<usize>> {
        let rng = rand::thread_rng();
        let mut indices = (0..labels.len()).collect::<Vec<_>>();
        let mut labels = labels;
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_rng(rng).unwrap(),
            };
            indices.shuffle(&mut rng);
            labels = indices.iter().map(|i| labels[*i]).collect();
        }

        let unique_labels = labels.iter().collect::<HashSet<_>>();
        let mut folds = vec![vec![]; self.n_splits];
        for unique_label in unique_labels {
            let mut count = 0;
            for (i, is_unique_label) in labels.iter().map(|x| x == unique_label).enumerate() {
                if is_unique_label {
                    folds[count % self.n_splits].push(indices[i]);
                    count += 1;
                }
            }
        }
        folds
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    use rstest::*;

    use crate::loader::{Embarked, Sex};

    #[fixture]
    pub fn fixture_input_data() -> InputData {
        InputData {
            passenger_id: 0,
            survived: None,
            pclass: Some(3),
            name: Some("Alice".to_string()),
            sex: Some(Sex::Female),
            age: Some(22.0),
            sibsp: Some(1),
            parch: Some(0),
            ticket: Some("A/5 21171".to_string()),
            fare: Some(7.25),
            cabin: None,
            embarked: Some(Embarked::S),
        }
    }

    #[rstest]
    fn test_KFold_split(fixture_input_data: InputData) {
        fn test(input_data: InputData) {
            let data = vec![input_data; 100];
            let mut labels = vec![0; 10];
            labels.extend(vec![1; 90].iter().copied());
            let mut rng = rand::thread_rng();
            labels.shuffle(&mut rng);

            let kfold = KFold::new(2, true, None);
            let folds = kfold.split(&data, labels.clone());
            assert_eq!(folds.len(), 2);
            for index in folds.iter() {
                assert_eq!(index.len(), 50);
            }
        }
        test(fixture_input_data);
    }

    #[rstest]
    fn test_StratifiedKFold_split(fixture_input_data: InputData) {
        fn test(input_data: InputData) {
            let data = vec![input_data; 100];
            let mut labels = vec![0; 10];
            labels.extend(vec![1; 90].iter().copied());
            let mut rng = rand::thread_rng();
            labels.shuffle(&mut rng);

            let kfold = StratifiedKFold::new(2, true, None);
            let folds = kfold.split(&data, labels.clone());
            assert_eq!(folds.len(), 2);
            for index in folds.iter() {
                let fold_labels = index.iter().map(|i| labels[*i]).collect::<Vec<_>>();
                let mut expected = vec![0; 5];
                expected.extend(vec![1; 45].iter().copied());
                assert_eq!(index.len(), 50);
                assert_eq!(
                    fold_labels.iter().filter(|x| **x == 0).count(),
                    expected.iter().filter(|x| **x == 0).count()
                );
                assert_eq!(
                    fold_labels.iter().filter(|x| **x == 1).count(),
                    expected.iter().filter(|x| **x == 1).count()
                );
            }
        }
        // seed によってテストが通る場合があるので、何度か試す
        for _ in 0..10 {
            test(fixture_input_data.clone());
        }
    }
}
