use std::collections::HashSet;
use std::hash::Hash;

use rand::prelude::*;

pub trait AbstractKFold<T, U> {
    fn split(&self, data: &[T], labels: Vec<U>) -> Vec<Vec<usize>>;
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

impl<T, U> AbstractKFold<T, U> for KFold {
    fn split(&self, data: &[T], _labels: Vec<U>) -> Vec<Vec<usize>> {
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

impl<T, U> AbstractKFold<T, U> for StratifiedKFold
where
    U: Copy + Eq + Hash,
{
    fn split(&self, _data: &[T], labels: Vec<U>) -> Vec<Vec<usize>> {
        let rng = rand::thread_rng();
        let mut indices = (0..labels.len()).collect::<Vec<_>>();
        let mut labels = labels;
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_rng(rng).unwrap(),
            };
            indices.shuffle(&mut rng);
            labels = indices.iter().map(|&i| labels[i]).collect();
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

    #[test]
    fn test_StratifiedKFold_split() {
        fn test() {
            let data = vec![0; 100];
            let mut labels = vec![0; 10];
            labels.extend(vec![1; 90].iter().copied());
            let mut rng = rand::thread_rng();
            labels.shuffle(&mut rng);

            let kfold = StratifiedKFold::new(2, true, None);
            let folds = kfold.split(&data, labels.clone());
            assert_eq!(folds.len(), 2);
            for index in folds.iter() {
                assert_eq!(index.len(), 50);
                let fold_labels = index.iter().map(|i| labels[*i]).collect::<Vec<_>>();
                let mut expected = vec![0; 5];
                expected.extend(vec![1; 45].iter().copied());
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
            test();
        }
    }
}
