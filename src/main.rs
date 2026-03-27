use burn_central_example::training::MnistTrainingConfig;

#[cfg(feature = "cuda")]
type TBackend = burn::backend::Cuda;
#[cfg(feature = "wgpu")]
type TBackend = burn::backend::Wgpu;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
type TBackend = burn::backend::NdArray;

type Device = <TBackend as burn::tensor::backend::Backend>::Device;
type TAutodiffBackend = burn::backend::Autodiff<TBackend>;

fn main() {
    let api_key = std::env::var("BURN_CENTRAL_API_KEY")
        .expect("BURN_CENTRAL_API_KEY environment variable must be set");

    let namespace = "jwric";
    let project_name = "burn-central-example";
    // This is a problem we are currently bound to the digest as we use it in remote launching features.
    // We could make it optional
    let digest = "489b819dc87e8fa398d3e6b44b63fe242c90f6b537293c8a85b3e886dee15139".to_string();
    let routine = "my_training_routine".to_string();
    let exp_metadata = serde_json::json!({
        "namespace": namespace,
        "project_name": project_name,
        "digest": digest,
    });

    let device = Device::default();

    let experiment =
        burn_central::BurnCentral::login(burn_central::BurnCentralCredentials::new(api_key))
            .unwrap()
            .start_experiment(namespace, project_name, digest, routine)
            .unwrap();

    experiment.log_config("metadata", &exp_metadata).unwrap();

    let config = MnistTrainingConfig::default();

    let _ = burn_central_example::training::run_manual::<TAutodiffBackend>(
        Some(&experiment),
        config,
        vec![device],
    );
}
