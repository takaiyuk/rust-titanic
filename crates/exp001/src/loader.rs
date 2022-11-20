use std::path::Path;

use anyhow::{anyhow, Result};
use csv::{Reader, StringRecord};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Sex {
    Female,
    Male,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Embarked {
    C, // Cherbourg
    Q, // Queenstown
    S, // Southampton
}

#[derive(Debug, Clone, PartialEq)]
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
        // * parse -> OK
        // * cannot parse -> Err
        v.parse::<u32>()
            .map_err(|_| anyhow!("failed to parse passenger_id"))
    }

    fn parse_survived(v: &str) -> Result<Option<u32>> {
        // * parse and result is 0 or 1 -> OK
        // * parse and result is not 0 or 1 -> Err
        // * cannot parse -> Err
        match v.parse::<u32>() {
            Ok(0) => Ok(Some(0)),
            Ok(1) => Ok(Some(1)),
            Ok(_) => Err(anyhow!("survived must be 0 or 1")),
            Err(_) => Err(anyhow!("failed to parse survived")),
        }
    }

    fn parse_pclass(v: &str) -> Result<Option<i32>> {
        // * parse -> OK(Some)
        // * cannot parse due to empty string -> Ok(None)
        // * cannot parse due to not integer -> Err
        if v.is_empty() {
            Ok(None)
        } else {
            v.parse::<i32>()
                .map(Some)
                .map_err(|_| anyhow!("failed to parse pclass"))
        }
    }

    fn parse_name(v: &str) -> Result<Option<String>> {
        // * parse -> OK(Some)
        // * empty string -> Ok(None)
        if v.is_empty() {
            Ok(None)
        } else {
            Ok(Some(v.to_string()))
        }
    }

    fn parse_sex(v: &str) -> Result<Option<Sex>> {
        // * female or male -> OK(Some)
        // * neither female nor male -> Ok(None)
        let sex = match v {
            "female" => Some(Sex::Female),
            "male" => Some(Sex::Male),
            _ => None,
        };
        Ok(sex)
    }

    fn parse_age(v: &str) -> Result<Option<f64>> {
        // * parse -> OK(Some)
        // * cannot parse due to empty string -> Ok(None)
        // * cannot parse due to not float -> Err
        if v.is_empty() {
            Ok(None)
        } else {
            v.parse::<f64>()
                .map(Some)
                .map_err(|_| anyhow!("failed to parse age"))
        }
    }

    fn parse_sibsp(v: &str) -> Result<Option<i32>> {
        // * parse -> OK(Some)
        // * cannot parse due to empty string -> Ok(None)
        // * cannot parse due to not integer -> Err
        if v.is_empty() {
            Ok(None)
        } else {
            v.parse::<i32>()
                .map(Some)
                .map_err(|_| anyhow!("failed to parse sibsp"))
        }
    }

    fn parse_parch(v: &str) -> Result<Option<i32>> {
        // * parse -> OK(Some)
        // * cannot parse due to empty string -> Ok(None)
        // * cannot parse due to not integer -> Err
        if v.is_empty() {
            Ok(None)
        } else {
            v.parse::<i32>()
                .map(Some)
                .map_err(|_| anyhow!("failed to parse parch"))
        }
        // Ok(v.parse::<i32>().map_or_else(|_| None, Some))
    }

    fn parse_ticket(v: &str) -> Result<Option<String>> {
        // * parse -> OK(Some)
        // * empty string -> Ok(None)
        if v.is_empty() {
            Ok(None)
        } else {
            Ok(Some(v.to_string()))
        }
    }

    fn parse_fare(v: &str) -> Result<Option<f64>> {
        // * parse -> OK(Some)
        // * cannot parse due to empty string -> Ok(None)
        // * cannot parse due to not float -> Err
        if v.is_empty() {
            Ok(None)
        } else {
            v.parse::<f64>()
                .map(Some)
                .map_err(|_| anyhow!("failed to parse fare"))
        }
    }

    fn parse_cabin(v: &str) -> Result<Option<String>> {
        // * parse -> OK(Some)
        // * empty string -> Ok(None)
        if v.is_empty() {
            Ok(None)
        } else {
            Ok(Some(v.to_string()))
        }
    }

    fn parse_embarked(v: &str) -> Result<Option<Embarked>> {
        // * C or Q or S -> OK(Some)
        // * neither C nor Q nor S -> Ok(None)
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

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    use rstest::*;

    fn assert_result_eq<T>(actual: Result<T>, expected: Result<T>)
    where
        T: std::fmt::Debug + PartialEq,
    {
        match (&actual, &expected) {
            (Ok(actual), Ok(expected)) => assert_eq!(actual, expected),
            (Err(actual), Err(expected)) => assert_eq!(actual.to_string(), expected.to_string()),
            _ => panic!("unexpected result: {:?}, {:?}", actual, expected),
        }
    }

    #[rstest]
    #[case(
        &StringRecord::from(vec!["1","0","3","Braund, Mr. Owen Harris","male","22","1","0","A/5 21171","7.25","","S"]),
        Ok(InputData{passenger_id: 1, survived: Some(0), pclass: Some(3), name: Some("Braund, Mr. Owen Harris".to_string()), sex: Some(Sex::Male), age: Some(22.0), sibsp: Some(1), parch: Some(0), ticket: Some("A/5 21171".to_string()), fare: Some(7.25), cabin: None, embarked: Some(Embarked::S)}),
    )]
    #[case(
        &StringRecord::from(vec!["","0","3","Braund, Mr. Owen Harris","male","22","1","0","A/5 21171","7.25","","S"]),
        Err(anyhow!("failed to parse passenger_id")),
    )]
    fn test_InputData_from_train_record(
        #[case] record: &StringRecord,
        #[case] expected: Result<InputData>,
    ) {
        let actual = InputData::from_train_record(record);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case(
        &StringRecord::from(vec!["1","3","Braund, Mr. Owen Harris","male","22","1","0","A/5 21171","7.25","","S"]),
        Ok(InputData{passenger_id: 1, survived: None, pclass: Some(3), name: Some("Braund, Mr. Owen Harris".to_string()), sex: Some(Sex::Male), age: Some(22.0), sibsp: Some(1), parch: Some(0), ticket: Some("A/5 21171".to_string()), fare: Some(7.25), cabin: None, embarked: Some(Embarked::S)}),
    )]
    #[case(
        &StringRecord::from(vec!["","3","Braund, Mr. Owen Harris","male","22","1","0","A/5 21171","7.25","","S"]),
        Err(anyhow!("failed to parse passenger_id")),
    )]
    fn test_InputData_from_test_record(
        #[case] record: &StringRecord,
        #[case] expected: Result<InputData>,
    ) {
        let actual = InputData::from_test_record(record);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("1", Ok(1))]
    #[case("", Err(anyhow!("failed to parse passenger_id")))]
    fn test_InputData_parse_passenger_id(#[case] v: &str, #[case] expected: Result<u32>) {
        let actual = InputData::parse_passenger_id(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("0", Ok(Some(0)))]
    #[case("1", Ok(Some(1)))]
    #[case("2", Err(anyhow!("survived must be 0 or 1")))]
    #[case("", Err(anyhow!("failed to parse survived")))]
    #[case("a", Err(anyhow!("failed to parse survived")))]
    fn test_InputData_parse_survived(#[case] v: &str, #[case] expected: Result<Option<u32>>) {
        let actual = InputData::parse_survived(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("1", Ok(Some(1)))]
    #[case("", Ok(None))]
    #[case("a", Err(anyhow!("failed to parse pclass")))]
    fn test_InputData_parse_pclass(#[case] v: &str, #[case] expected: Result<Option<i32>>) {
        let actual = InputData::parse_pclass(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("Alice", Ok(Some("Alice".to_string())))]
    #[case("", Ok(None))]
    fn test_InputData_parse_name(#[case] v: &str, #[case] expected: Result<Option<String>>) {
        let actual = InputData::parse_name(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("female", Ok(Some(Sex::Female)))]
    #[case("male", Ok(Some(Sex::Male)))]
    #[case("", Ok(None))]
    fn test_InputData_parse_sex(#[case] v: &str, #[case] expected: Result<Option<Sex>>) {
        let actual = InputData::parse_sex(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("20", Ok(Some(20.0)))]
    #[case("", Ok(None))]
    #[case("a", Err(anyhow!("failed to parse age")))]
    fn test_InputData_parse_age(#[case] v: &str, #[case] expected: Result<Option<f64>>) {
        let actual = InputData::parse_age(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("2", Ok(Some(2)))]
    #[case("", Ok(None))]
    #[case("a", Err(anyhow!("failed to parse sibsp")))]
    fn test_InputData_parse_sibsp(#[case] v: &str, #[case] expected: Result<Option<i32>>) {
        let actual = InputData::parse_sibsp(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("2", Ok(Some(2)))]
    #[case("", Ok(None))]
    #[case("a", Err(anyhow!("failed to parse parch")))]
    fn test_InputData_parse_parch(#[case] v: &str, #[case] expected: Result<Option<i32>>) {
        let actual = InputData::parse_parch(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("A/5 21171", Ok(Some("A/5 21171".to_string())))]
    #[case("", Ok(None))]
    fn test_InputData_parse_ticket(#[case] v: &str, #[case] expected: Result<Option<String>>) {
        let actual = InputData::parse_ticket(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("7.25", Ok(Some(7.25)))]
    #[case("", Ok(None))]
    #[case("a", Err(anyhow!("failed to parse fare")))]
    fn test_InputData_parse_fare(#[case] v: &str, #[case] expected: Result<Option<f64>>) {
        let actual = InputData::parse_fare(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("C85", Ok(Some("C85".to_string())))]
    #[case("", Ok(None))]
    fn test_InputData_parse_cabin(#[case] v: &str, #[case] expected: Result<Option<String>>) {
        let actual = InputData::parse_cabin(v);
        assert_result_eq(actual, expected);
    }

    #[rstest]
    #[case("C", Ok(Some(Embarked::C)))]
    #[case("Q", Ok(Some(Embarked::Q)))]
    #[case("S", Ok(Some(Embarked::S)))]
    #[case("", Ok(None))]
    fn test_InputData_parse_embarked(#[case] v: &str, #[case] expected: Result<Option<Embarked>>) {
        let actual = InputData::parse_embarked(v);
        assert_result_eq(actual, expected);
    }
}
