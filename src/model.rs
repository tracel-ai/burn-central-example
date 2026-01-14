use crate::{data::MnistBatch, training::MnistTrainingConfig};
use burn::{
    nn::{
        BatchNorm, PaddingConfig2d,
        loss::CrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_central::bundle::{BundleDecode, BundleEncode, BundleSink, BundleSource};

#[derive(Module, Debug)]
pub struct MnistModel<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    dropout: nn::Dropout,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
    activation: nn::Gelu,
}

impl<B: Backend> Default for MnistModel<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}

const NUM_CLASSES: usize = 10;

impl<B: Backend> MnistModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = ConvBlock::new([1, 64], [3, 3], device, true); // out: max_pool -> [Batch,32,13,13]
        let conv2 = ConvBlock::new([64, 64], [3, 3], device, true); // out: max_pool -> [Batch,64,5,5]
        let hidden_size = 64 * 5 * 5;
        let fc1 = nn::LinearConfig::new(hidden_size, 128).init(device);
        let fc2 = nn::LinearConfig::new(128, 128).init(device);
        let fc3 = nn::LinearConfig::new(128, NUM_CLASSES).init(device);

        let dropout = nn::DropoutConfig::new(0.25).init();

        Self {
            conv1,
            conv2,
            dropout,
            fc1,
            fc2,
            fc3,
            activation: nn::Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.fc3.forward(x)
    }

    pub fn forward_classification(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B>,
    pool: Option<MaxPool2d>,
    activation: nn::Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        device: &B::Device,
        pool: bool,
    ) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Valid)
            .init(device);
        let norm = nn::BatchNormConfig::new(channels[1]).init(device);
        let pool = if pool {
            Some(MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init())
        } else {
            None
        };

        Self {
            conv,
            norm,
            pool,
            activation: nn::Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        let x = self.activation.forward(x);

        if let Some(pool) = &self.pool {
            pool.forward(x)
        } else {
            x
        }
    }
}

impl<B: AutodiffBackend> TrainStep for MnistModel<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for MnistModel<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

// Define the model artifact (put in that everything you will need for inference)
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
        let config_bytes = serde_json::to_vec(&self.config)
            .map_err(|e| format!("Failed to serialize config: {e}"))?;
        sink.put_bytes("config.json", &config_bytes)
            .map_err(|e| format!("Failed to write config: {e}"))?;

        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let model_bytes = recorder
            .record(self.model_record, ())
            .map_err(|e| format!("Failed to record model: {e}"))?;

        sink.put_bytes("model.mpk", &model_bytes)
            .map_err(|e| format!("Failed to write model: {e}"))?;

        Ok(())
    }
}

impl<B: Backend> BundleDecode for MnistModelArtifact<B> {
    type Settings = ();
    type Error = String;

    fn decode<I: BundleSource>(
        source: &I,
        _settings: &Self::Settings,
    ) -> Result<Self, Self::Error> {
        let config_reader = source
            .open("config.json")
            .map_err(|e| format!("Failed to read config: {e}"))?;
        let config: MnistTrainingConfig = serde_json::from_reader(config_reader)
            .map_err(|e| format!("Failed to deserialize config: {e}"))?;

        let mut model_reader = source
            .open("model.mpk")
            .map_err(|e| format!("Failed to read model: {e}"))?;
        let mut model_bytes = Vec::new();

        model_reader
            .read_to_end(&mut model_bytes)
            .map_err(|e| format!("Failed to read model: {e}"))?;

        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let (model_record, ()) = recorder
            .load(model_bytes, &B::Device::default())
            .map_err(|e| format!("Failed to read model: {e}"))?;

        Ok(MnistModelArtifact {
            model_record,
            config,
        })
    }
}
