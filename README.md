# Burn Central Example (MNIST)

The first commit of this repository is a clasic burn example project. It doesn't not containt anything related to burn-central. The second commit add everything we need for burn-central. We highly recommand to look at it in on [Github](https://github.com/tracel-ai/burn-central-example/commit/afcfbce95c5ce9fef27342a3f351555d4e242ff0).

## Step

To go from a non burn-central project to a burn-central ready project we recommand following this step.

### 0. Import dependencies

First thing first run `cargo install burn-central`!

### 1. Define a artifact

The first thing you should do is define the final artifact. The artifact you define must be return by the training function. It will be automaticly upload by the SDK to burn-central.

The goal of the artifact is to define what you need to do inference. Most of the time you should included your trained model and probably your training configuration. Note that you will have to implemente `BundleEncode` and `BundleDecode` for your artifact.

In this example the code can be find [here](https://github.com/tracel-ai/burn-central-example/blob/afcfbce95c5ce9fef27342a3f351555d4e242ff0/src/model.rs#L159)! We included a preview below.

```rust 
pub struct MnistModelArtifact<B: Backend> {
    pub model_record: MnistModelRecord<B>,
    pub config: MnistTrainingConfig,
}

impl<B: Backend> BundleEncode for MnistModelArtifact<B> {
    type Settings = ();
    type Error = String;

    fn encode<O: BundleSink>(
        self,
        sink: &mut O,
        _settings: &Self::Settings,
    ) -> Result<(), Self::Error> {
        ...
    }
}

impl<B: Backend> BundleDecode for MnistModelArtifact<B> {
    type Settings = ();
    type Error = String;

    fn decode<I: BundleSource>(
        source: &I,
        _settings: &Self::Settings,
    ) -> Result<Self, Self::Error> {
        ...
    }
}
```

### 2. Define a configuration

If you have a training conifugration that make sens to override one run to another or parameter you want to tweak make sure you define a structure that containts those parameter and make this struct implement `Default` and `Serialize/Deserialize`


### 3. Add annotation to define training function

Convert your training function to burn-central

Before:
```rust
pub fn run<B: AutodiffBackend>(device: B::Device) {
    /// Training logic
}
```

After:
```rust
#[register(training, name = "train_mnist")] // Name parameter not require (default to function name)
pub fn run<B: AutodiffBackend>(
    client: &ExperimentRun, // Client to used burn-central feature
    config: Args<MnistTrainingConfig>, // The configuration you want to override
    MultiDevice(devices): MultiDevice<B>, // Device are injected as they are define by the CLI
) -> burn_central::runtime::Model<MnistModelArtifact<B::InnerBackend>> { // Make sure to use Model from burn-central runtime
    /// Training logic
}
```


### 4. Inject burn-central tooling

Right now you must inject the remote implementation of burn trait to send data to our platform.

```rust
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        ...
        .with_metric_logger(RemoteMetricLogger::new(client))
        .with_file_checkpointer(RemoteCheckpointRecorder::new(client))
        .with_application_logger(Some(Box::new(RemoteExperimentLoggerInstaller::new(client))))
        ...
```
