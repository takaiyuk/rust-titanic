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
