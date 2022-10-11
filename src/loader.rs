use std::path::Path;

use anyhow::Result;
use csv::{Reader, StringRecord};

#[derive(Debug, Copy, Clone)]
pub enum Sex {
    Female,
    Male,
}

#[derive(Debug, Copy, Clone)]
pub enum Embarked {
    C, // Cherbourg
    Q, // Queenstown
    S, // Southampton
}

#[derive(Debug, Clone)]
pub struct InputData {
    pub passenger_id: u32,
    pub survived: Option<u32>,
    pub pclass: Option<i32>,
    pub name: Option<String>,
    pub sex: Option<Sex>,
    pub age: Option<f64>,
    pub sibsp: Option<i32>,
    pub parch: Option<i32>,
    pub ticket: Option<String>,
    pub fare: Option<f64>,
    pub cabin: Option<String>,
    pub embarked: Option<Embarked>,
}

impl InputData {
    fn from_record(record: &StringRecord) -> Result<InputData> {
        let is_train = if record.len() == 12 {
            true
        } else if record.len() == 11 {
            false
        } else {
            panic!("Invalid record length: {}", record.len())
        };

        let passenger_id = record[0].parse::<u32>()?;
        let survived = if is_train {
            Some(record[1].parse::<u32>()?)
        } else {
            None
        };
        let pclass_index = if is_train { 2 } else { 1 };
        let pclass = record[pclass_index]
            .parse::<i32>()
            .map_or_else(|_| None, Some);
        let name_index = if is_train { 3 } else { 2 };
        let name = if !record[name_index].is_empty() {
            Some(record[name_index].to_string())
        } else {
            None
        };
        let sex_index = if is_train { 4 } else { 3 };
        let sex = match &record[sex_index] {
            "female" => Some(Sex::Female),
            "male" => Some(Sex::Male),
            _ => None,
        };
        let age_index = if is_train { 5 } else { 4 };
        let age = record[age_index].parse::<f64>().map_or_else(|_| None, Some);
        let sibsp_index = if is_train { 6 } else { 5 };
        let sibsp = record[sibsp_index]
            .parse::<i32>()
            .map_or_else(|_| None, Some);
        let parch_index = if is_train { 7 } else { 6 };
        let parch = record[parch_index]
            .parse::<i32>()
            .map_or_else(|_| None, Some);
        let ticket_index = if is_train { 8 } else { 7 };
        let ticket = if !record[ticket_index].is_empty() {
            Some(record[ticket_index].to_string())
        } else {
            None
        };
        let fare_index = if is_train { 9 } else { 8 };
        let fare = record[fare_index]
            .parse::<f64>()
            .map_or_else(|_| None, Some);
        let cabin_index = if is_train { 10 } else { 9 };
        let cabin = if !record[cabin_index].is_empty() {
            Some(record[cabin_index].to_string())
        } else {
            None
        };
        let embarked_index = if is_train { 11 } else { 10 };
        let embarked = match &record[embarked_index] {
            "C" => Some(Embarked::C),
            "Q" => Some(Embarked::Q),
            "S" => Some(Embarked::S),
            _ => None,
        };

        Ok(InputData {
            passenger_id,
            survived,
            pclass,
            name,
            sex,
            age,
            sibsp,
            parch,
            ticket,
            fare,
            cabin,
            embarked,
        })
    }
}

pub fn load_train<P: AsRef<Path>>(path: P) -> Result<Vec<InputData>> {
    let mut rdr = Reader::from_path(path)?;
    let records = rdr
        .records()
        .into_iter()
        .map(|r| InputData::from_record(&r.unwrap()).unwrap())
        .collect();
    Ok(records)
}

pub fn load_test<P: AsRef<Path>>(path: P) -> Result<Vec<InputData>> {
    let mut rdr = Reader::from_path(path)?;
    let records = rdr
        .records()
        .into_iter()
        .map(|r| InputData::from_record(&r.unwrap()).unwrap())
        .collect();
    Ok(records)
}
