# rust-titanic

Repository which is an attempt to solve the [Titanic ML competition](https://www.kaggle.com/competitions/titanic) with Rust.

## Setup

### Data

Download [titanic data from kaggle](https://www.kaggle.com/competitions/titanic/data), or use Kaggle API to download the data as below.

```
$ kaggle competitions download -c titanic -p .
$ mkdir -p input
$ unzip titanic.zip -d input/titanic
$ rm titanic.zip
```

### Cargo-make (Optional)

Install `cargo-make` as a task runner

```
cargo install --force cargo-make
```

## Execute

```
$ cargo run
```

If you installed `cargo-make`, you can execute as below

```
$ cargo make run expXXX
```

### Result of exp001

```
Fold 1
Accuracy: 0.8435754189944135
Fold 2
Accuracy: 0.8324022346368715
Fold 3
Accuracy: 0.8426966292134831
Fold 4
Accuracy: 0.8539325842696629
Fold 5
Accuracy: 0.751412429378531
CV Accuracy: 0.8249158249158249
Feature names: ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "title", "family_size"]
Feature importances: [108.4, 57.0, 697.6, 43.6, 30.4, 819.2, 96.8, 118.8, 93.6]
```

```
Public Score: 0.75119
```

## Development

See the output of `cargo make --list-all-steps`

```
cargo make --list-all-steps
[cargo-make] INFO - cargo make 0.36.3
[cargo-make] INFO - Project: rust-titanic
[cargo-make] INFO - Build File: Makefile.toml
[cargo-make] INFO - Task: default
[cargo-make] INFO - Profile: development
Build
----------
build - cargo build
run - cargo run

Development
----------
format - cargo fmt

Test
----------
check - cargo check
clippy - cargo clippy
coverage - cargo llvm-cov
coverage-html - cargo llvm-cov --html
coverage-open - cargo llvm-cov --open
test - cargo nextest
```
