# Burn Central Example (MNIST)

This repository shows how to adapt the standard Burn MNIST example into a Burn Central-ready project. It includes both:

- a local executable that runs the experiment on your machine
- a registered Burn Central training routine for remote execution

## What This Example Covers

- defining a custom model artifact with `BundleEncode` and `BundleDecode`
- exposing training configuration through `Args<MnistTrainingConfig>`
- registering a training routine with `#[register(training)]`
- wiring metrics, checkpoints, and interruption handling through `ExperimentRun`
- reusing the same training logic for both local and Burn Central execution

## Project Layout

- [`src/main.rs`](src/main.rs): local runner using `ExperimentRun::local("./experiments")`
- [`src/training.rs`](src/training.rs): training routine registration, learner setup, evaluation, and artifact upload
- [`src/model.rs`](src/model.rs): model definition and artifact bundle serialization
- [`src/data.rs`](src/data.rs): MNIST batching and data augmentation

## Run Locally

By default the example uses the `NdArray` backend. Optional backends can be enabled with Cargo features.

```bash
cargo run
cargo run --features wgpu
cargo run --features cuda
```

The local runner writes experiment data and artifacts to `./experiments`.

## Run Through Burn Central

Install the CLI first:

```bash
cargo install burn-central-cli
```

Then initialize and authenticate the project:

```bash
burn init
burn login
```

This example registers the training routine as `train_mnist`. Run it through the CLI:

```bash
burn train
burn train train_mnist
burn train train_mnist --override num_epochs=1
```

The CLI discovers registered routines and executes the job through the standard Burn Central workflow.

## More Details

- Burn Central SDK docs: [docs.rs/burn-central](https://docs.rs/burn-central/latest/burn_central/)
