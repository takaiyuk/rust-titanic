use std::path::Path;

use anyhow::Result;
use csv::{Reader, Writer};

pub fn generate_submission<P: AsRef<Path>>(
    pred_test: Vec<i32>,
    input_path: P,
    output_path: P,
) -> Result<()> {
    let mut rdr = Reader::from_path(input_path)?;
    let mut wtr = Writer::from_path(output_path)?;
    for (i, (r, p)) in rdr
        .records()
        .into_iter()
        .zip(pred_test.into_iter())
        .enumerate()
    {
        let r = r.unwrap();
        let id = &r[0];
        if i == 0 {
            wtr.write_record(["PassengerId", "Survived"])?;
        }
        wtr.write_record([id, &p.to_string()])?;
    }
    Ok(())
}
