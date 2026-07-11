use mlx_rs::Array;

use super::{ModelError, ModelResult, QuantizedLinear, Tensor3};

/// One checkpoint language-prompt slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LanguagePrompt {
    code: &'static str,
    id: usize,
}

impl LanguagePrompt {
    /// Resolves exactly the aliases stored in `processor_config.json`.
    pub fn from_code(code: &str) -> ModelResult<Self> {
        PROMPT_DICTIONARY
            .iter()
            .find(|(candidate, _)| *candidate == code)
            .map(|(code, id)| Self { code, id: *id })
            .ok_or_else(|| ModelError::UnsupportedLanguage(code.to_string()))
    }

    pub const fn id(self) -> usize {
        self.id
    }

    pub const fn code(self) -> &'static str {
        self.code
    }

    pub fn supported_codes() -> &'static [(&'static str, usize)] {
        PROMPT_DICTIONARY
    }
}

/// Quantized two-layer MLP that fuses encoder frames with a broadcast prompt one-hot.
#[derive(Debug)]
pub struct PromptProjector {
    hidden_size: usize,
    num_prompts: usize,
    linear1: QuantizedLinear,
    linear2: QuantizedLinear,
}

impl PromptProjector {
    pub fn from_artifact(artifact: &crate::weights::Artifact) -> ModelResult<Self> {
        let linear1 = QuantizedLinear::from_artifact(
            artifact,
            "prompt_projector.linear_1.weight",
            Some("prompt_projector.linear_1.bias"),
        )?;
        let linear2 = QuantizedLinear::from_artifact(
            artifact,
            "prompt_projector.linear_2.weight",
            Some("prompt_projector.linear_2.bias"),
        )?;
        let hidden_size = linear2.output_dims();
        let num_prompts = linear1
            .input_dims()
            .checked_sub(hidden_size)
            .ok_or_else(|| {
                ModelError::InvalidShape(
                    "prompt projector input must include hidden state and prompt one-hot"
                        .to_string(),
                )
            })?;
        Ok(Self {
            hidden_size,
            num_prompts,
            linear1,
            linear2,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_f32(
        weight1: &[f32],
        bias1: &[f32],
        weight2: &[f32],
        bias2: &[f32],
        hidden_size: usize,
        num_prompts: usize,
        intermediate_size: usize,
        group_size: usize,
    ) -> ModelResult<Self> {
        Ok(Self {
            hidden_size,
            num_prompts,
            linear1: QuantizedLinear::from_f32(
                weight1,
                intermediate_size,
                hidden_size + num_prompts,
                bias1,
                group_size,
            )?,
            linear2: QuantizedLinear::from_f32(
                weight2,
                hidden_size,
                intermediate_size,
                bias2,
                group_size,
            )?,
        })
    }

    pub fn forward_f32(
        &self,
        hidden_states: &[f32],
        batch: usize,
        time: usize,
        prompt: LanguagePrompt,
    ) -> ModelResult<Tensor3> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| ModelError::InvalidShape("prompt row count overflow".to_string()))?;
        if hidden_states.len() != rows * self.hidden_size || prompt.id >= self.num_prompts {
            return Err(ModelError::InvalidShape(format!(
                "prompt input requires [{batch},{time},{}] and a prompt below {}",
                self.hidden_size, self.num_prompts
            )));
        }

        let fused_dims = self.hidden_size + self.num_prompts;
        let mut fused = vec![0.0; rows * fused_dims];
        for row in 0..rows {
            let input_start = row * self.hidden_size;
            let output_start = row * fused_dims;
            fused[output_start..output_start + self.hidden_size]
                .copy_from_slice(&hidden_states[input_start..input_start + self.hidden_size]);
            fused[output_start + self.hidden_size + prompt.id] = 1.0;
        }

        let input = Array::from_slice(&fused, &[rows as i32, fused_dims as i32]);
        let hidden = mlx_rs::nn::relu(self.linear1.forward_array(&input)?)?;
        let output = self.linear2.forward_array(&hidden)?.as_type::<f32>()?;
        output.eval()?;
        Ok(Tensor3 {
            shape: [batch, time, self.hidden_size],
            values: output.try_as_slice::<f32>()?.to_vec(),
        })
    }
}

const PROMPT_DICTIONARY: &[(&str, usize)] = &[
    ("af-ZA", 54),
    ("am-ET", 49),
    ("ar", 7),
    ("ar-AR", 7),
    ("auto", 101),
    ("ay-BO", 81),
    ("az-AZ", 66),
    ("bg", 30),
    ("bg-BG", 30),
    ("bn-IN", 36),
    ("cs", 22),
    ("cs-CZ", 22),
    ("da", 25),
    ("da-DK", 25),
    ("de", 9),
    ("de-DE", 9),
    ("el", 21),
    ("el-GR", 21),
    ("en", 0),
    ("en-GB", 1),
    ("en-US", 0),
    ("enGB", 1),
    ("es", 3),
    ("es-ES", 2),
    ("es-US", 3),
    ("esES", 2),
    ("et", 60),
    ("et-EE", 60),
    ("fa-IR", 38),
    ("fi", 26),
    ("fi-FI", 26),
    ("fr", 8),
    ("fr-CA", 100),
    ("fr-FR", 8),
    ("gn-PY", 82),
    ("gu-IN", 42),
    ("ha-NG", 50),
    ("haw-US", 97),
    ("he-IL", 64),
    ("hi", 6),
    ("hi-HI", 6),
    ("hi-IN", 6),
    ("hr", 29),
    ("hr-HR", 29),
    ("hu", 23),
    ("hu-HU", 23),
    ("hy-AM", 68),
    ("id-ID", 34),
    ("ig-NG", 53),
    ("it", 15),
    ("it-IT", 15),
    ("ja-JA", 10),
    ("ja-JP", 10),
    ("ka-GE", 67),
    ("km-KH", 47),
    ("kn-IN", 43),
    ("ko", 14),
    ("ko-KO", 14),
    ("ko-KR", 14),
    ("ku-TR", 65),
    ("ky-KG", 71),
    ("ln-CD", 58),
    ("lt", 31),
    ("lt-LT", 31),
    ("lv", 61),
    ("lv-LV", 61),
    ("mi-NZ", 96),
    ("ml-IN", 44),
    ("mr-IN", 41),
    ("ms-MY", 35),
    ("mt-MT", 102),
    ("nah-MX", 83),
    ("nb", 103),
    ("nb-NO", 103),
    ("ne-NP", 46),
    ("nl", 16),
    ("nl-NL", 16),
    ("nn", 104),
    ("nn-NO", 104),
    ("no", 27),
    ("no-NO", 27),
    ("ny-MW", 57),
    ("or-KE", 59),
    ("pl", 17),
    ("pl-PL", 17),
    ("pt", 13),
    ("pt-BR", 12),
    ("pt-PT", 13),
    ("qu-PE", 80),
    ("ro", 20),
    ("ro-RO", 20),
    ("ru", 11),
    ("ru-RU", 11),
    ("rw-RW", 55),
    ("si-LK", 45),
    ("sk", 28),
    ("sk-SK", 28),
    ("sl", 62),
    ("sl-SI", 62),
    ("sm-WS", 98),
    ("so-SO", 56),
    ("sv", 24),
    ("sv-SE", 24),
    ("sw-KE", 48),
    ("ta-IN", 39),
    ("te-IN", 40),
    ("tg-TJ", 70),
    ("th-TH", 32),
    ("to-TO", 99),
    ("tr", 18),
    ("tr-TR", 18),
    ("uk", 19),
    ("uk-UA", 19),
    ("ur-PK", 37),
    ("uz-UZ", 69),
    ("vi-VN", 33),
    ("yo-NG", 52),
    ("zh-CN", 4),
    ("zh-TW", 5),
    ("zh-ZH", 4),
    ("zu-ZA", 51),
];
