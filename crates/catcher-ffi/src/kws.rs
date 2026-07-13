use std::path::{Path, PathBuf};

use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig, OnlineStream};

pub const EXPECTED_KEYWORD: &str = "TIPPI_GO";

pub struct KeywordDetection {
    pub keyword: String,
    pub start_ms: u64,
}

pub struct KeywordSpotterSession {
    // Fields drop in declaration order; release the stream before its owner.
    stream: OnlineStream,
    spotter: KeywordSpotter,
    latched: bool,
}

impl KeywordSpotterSession {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let mut config = KeywordSpotterConfig::default();
        config.model_config.transducer.encoder = Some(path(
            directory,
            "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
        )?);
        config.model_config.transducer.decoder = Some(path(
            directory,
            "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        )?);
        config.model_config.transducer.joiner = Some(path(
            directory,
            "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
        )?);
        config.model_config.tokens = Some(path(directory, "tokens.txt")?);
        config.model_config.modeling_unit = Some("cjkchar".to_string());
        config.model_config.num_threads = 1;
        config.keywords_file = Some(path(directory, "keywords.txt")?);

        let spotter = KeywordSpotter::create(&config)
            .ok_or_else(|| "sherpa-onnx could not create keyword spotter".to_string())?;
        let stream = spotter.create_stream();
        Ok(Self {
            stream,
            spotter,
            latched: false,
        })
    }

    pub fn reset(&mut self) {
        self.spotter.reset(&self.stream);
        self.latched = false;
    }

    pub fn push(&mut self, samples: &[f32]) -> Option<KeywordDetection> {
        if self.latched {
            return None;
        }

        self.stream.accept_waveform(16_000, samples);
        while self.spotter.is_ready(&self.stream) {
            self.spotter.decode(&self.stream);
        }

        let result = self.spotter.get_result(&self.stream)?;
        if result.keyword != EXPECTED_KEYWORD {
            return None;
        }
        // sherpa-onnx 1.13.4 leaves `start_time` at zero for this KWS model,
        // while the first decoded token timestamp carries the actual keyword
        // start. Prefer the dedicated field when populated and retain the
        // token timestamp as the runtime-compatible fallback.
        let start_time = if result.start_time > 0.0 {
            result.start_time
        } else {
            result.timestamps.first().copied().unwrap_or(0.0)
        };
        self.latched = true;
        Some(KeywordDetection {
            keyword: result.keyword,
            start_ms: (start_time.max(0.0) * 1_000.0).round() as u64,
        })
    }
}

fn path(directory: &Path, name: &str) -> Result<String, String> {
    let path: PathBuf = directory.join(name);
    if !path.is_file() {
        return Err(format!("missing KWS model file: {}", path.display()));
    }
    Ok(path.to_string_lossy().into_owned())
}
