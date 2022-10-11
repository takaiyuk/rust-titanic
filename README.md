# rust-titanic

## Setup

Download titanic data from [kaggle](https://www.kaggle.com/competitions/titanic), or use Kaggle API to download the data as below.

```
$ kaggle competitions download -c titanic -p .
$ mkdir -p input
$ unzip titanic.zip -d input/titanic
$ rm titanic.zip
```

## Execute

```
$ cargo run
```
