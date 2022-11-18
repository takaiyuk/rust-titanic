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
    fn from_train_record(record: &StringRecord) -> Result<InputData> {
        let passenger_id = InputData::parse_passenger_id(&record[0])?;
        let survived = InputData::parse_survived(&record[1])?;
        let pclass = InputData::parse_pclass(&record[2])?;
        let name = InputData::parse_name(&record[3])?;
        let sex = InputData::parse_sex(&record[4])?;
        let age = InputData::parse_age(&record[5])?;
        let sibsp = InputData::parse_sibsp(&record[6])?;
        let parch = InputData::parse_parch(&record[7])?;
        let ticket = InputData::parse_ticket(&record[8])?;
        let fare = InputData::parse_fare(&record[9])?;
        let cabin = InputData::parse_cabin(&record[10])?;
        let embarked = InputData::parse_embarked(&record[11])?;
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

    fn from_test_record(record: &StringRecord) -> Result<InputData> {
        let passenger_id = InputData::parse_passenger_id(&record[0])?;
        let survived = None;
        let pclass = InputData::parse_pclass(&record[1])?;
        let name = InputData::parse_name(&record[2])?;
        let sex = InputData::parse_sex(&record[3])?;
        let age = InputData::parse_age(&record[4])?;
        let sibsp = InputData::parse_sibsp(&record[5])?;
        let parch = InputData::parse_parch(&record[6])?;
        let ticket = InputData::parse_ticket(&record[7])?;
        let fare = InputData::parse_fare(&record[8])?;
        let cabin = InputData::parse_cabin(&record[9])?;
        let embarked = InputData::parse_embarked(&record[10])?;
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

    fn parse_passenger_id(v: &str) -> Result<u32> {
        Ok(v.parse::<u32>()?)
    }

    fn parse_survived(v: &str) -> Result<Option<u32>> {
        Ok(Some(v.parse::<u32>()?))
    }

    fn parse_pclass(v: &str) -> Result<Option<i32>> {
        Ok(v.parse::<i32>().map_or_else(|_| None, Some))
    }

    fn parse_name(v: &str) -> Result<Option<String>> {
        if !v.is_empty() {
            Ok(Some(v.to_string()))
        } else {
            Ok(None)
        }
    }

    fn parse_sex(v: &str) -> Result<Option<Sex>> {
        let sex = match v {
            "female" => Some(Sex::Female),
            "male" => Some(Sex::Male),
            _ => None,
        };
        Ok(sex)
    }

    fn parse_age(v: &str) -> Result<Option<f64>> {
        Ok(v.parse::<f64>().map_or_else(|_| None, Some))
    }

    fn parse_sibsp(v: &str) -> Result<Option<i32>> {
        Ok(v.parse::<i32>().map_or_else(|_| None, Some))
    }

    fn parse_parch(v: &str) -> Result<Option<i32>> {
        Ok(v.parse::<i32>().map_or_else(|_| None, Some))
    }

    fn parse_ticket(v: &str) -> Result<Option<String>> {
        if !v.is_empty() {
            Ok(Some(v.to_string()))
        } else {
            Ok(None)
        }
    }

    fn parse_fare(v: &str) -> Result<Option<f64>> {
        Ok(v.parse::<f64>().map_or_else(|_| None, Some))
    }

    fn parse_cabin(v: &str) -> Result<Option<String>> {
        if !v.is_empty() {
            Ok(Some(v.to_string()))
        } else {
            Ok(None)
        }
    }

    fn parse_embarked(v: &str) -> Result<Option<Embarked>> {
        let embarked = match v {
            "C" => Some(Embarked::C),
            "Q" => Some(Embarked::Q),
            "S" => Some(Embarked::S),
            _ => None,
        };
        Ok(embarked)
    }
}

pub fn load_train_data<P: AsRef<Path>>(path: P) -> Result<Vec<InputData>> {
    let mut rdr = Reader::from_path(path)?;
    let records = rdr
        .records()
        .into_iter()
        .map(|r| InputData::from_train_record(&r.unwrap()).unwrap())
        .collect();
    Ok(records)
}

pub fn load_test_data<P: AsRef<Path>>(path: P) -> Result<Vec<InputData>> {
    let mut rdr = Reader::from_path(path)?;
    let records = rdr
        .records()
        .into_iter()
        .map(|r| InputData::from_test_record(&r.unwrap()).unwrap())
        .collect();
    Ok(records)
}
