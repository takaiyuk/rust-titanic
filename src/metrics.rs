use anyhow::Result;

pub fn accuracy(t: &[f64], p: &[f64]) -> Result<f64> {
    let mut correct = 0;
    for (t, p) in t.iter().zip(p.iter()) {
        if *p > 0.5 {
            if *t == 1.0 {
                correct += 1;
            }
        } else if *t == 0.0 {
            correct += 1;
        }
    }
    Ok(correct as f64 / t.len() as f64)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    use rstest::*;

    #[rstest]
    #[case(
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0],
        0.5
    )]
    fn test_accuracy(#[case] t: Vec<f64>, #[case] p: Vec<f64>, #[case] expected: f64) {
        let t = t.as_slice();
        let p = p.as_slice();
        let actual = accuracy(t, p).unwrap();
        assert_eq!(actual, expected);
    }
}
