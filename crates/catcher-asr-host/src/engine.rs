//! 推論引擎抽象。Session 只認識這個 trait,
//! 讓狀態機能在沒有 Metal 與模型檔的環境(CI、驗收測試)用 FakeEngine 驗證。

pub trait AsrEngine {
    /// 餵入 16 kHz mono f32 samples,回傳新產生的 token ids。
    fn push(&mut self, samples: &[f32]) -> Result<Vec<u32>, String>;
    /// 沖洗解碼器,回傳最後一批 token ids。
    fn finish(&mut self) -> Result<Vec<u32>, String>;
    /// 把「會話累積的全部 ids」解成文字(host 端已含繁化)。
    fn decode(&self, ids: &[u32]) -> Result<String, String>;
    fn backend(&self) -> &'static str;
}

/// 決定性假引擎:每滿 1600 samples 產出一個遞增 id;
/// decode 把每個 id 映成 "字N",讓測試能精確斷言 partial/final 內容。
pub struct FakeEngine {
    buffered: usize,
    next_id: u32,
}

impl FakeEngine {
    pub fn new() -> Self {
        Self { buffered: 0, next_id: 0 }
    }
}

impl AsrEngine for FakeEngine {
    fn push(&mut self, samples: &[f32]) -> Result<Vec<u32>, String> {
        self.buffered += samples.len();
        let mut ids = Vec::new();
        while self.buffered >= 1600 {
            self.buffered -= 1600;
            ids.push(self.next_id);
            self.next_id += 1;
        }
        Ok(ids)
    }

    fn finish(&mut self) -> Result<Vec<u32>, String> {
        self.buffered = 0;
        Ok(Vec::new())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        Ok(ids.iter().map(|id| format!("字{id}")).collect())
    }

    fn backend(&self) -> &'static str {
        "fake"
    }
}
