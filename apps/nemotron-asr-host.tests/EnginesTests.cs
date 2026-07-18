// Final-review fix (兩份 host 不一致修復清單項目 1+5):驗證 InferenceBackend enum →
// 協定線上字串("dml"/"cpu")的映射本身,不經過 NemotronEngineAdapter/NemotronEngine
// (兩者都需要真正的 onnxruntime-genai 模型目錄才能建構,不適合 fast suite)。
// BackendWireName 是 Engines.cs 抽出的 static mapper,見該檔案註解:NemotronEngineAdapter.Backend
// 現在一律讀 inner.Backend 現算,不再接受建構子傳入的 InferenceBackend 參數
// (修掉 BackendProber.LoadEngine 靜默 DML→CPU 回退時「回報值」與「實際引擎」不一致的 bug),
// 因此原本規劃的「new NemotronEngineAdapter(null!, ..., InferenceBackend.DirectML).Backend」
// 寫法已不適用(inner 是 null! 會在讀 inner.Backend 時 NPE)——改測映射本身所在的新位置。
using NemotronAsrHost;
using Tippi.Windows.Services;
using Xunit;

namespace NemotronAsrHost.Tests;

public class EnginesTests
{
    [Fact]
    public void BackendWireNameMapsDirectMlToDml()
    {
        Assert.Equal("dml", BackendWireName.For(InferenceBackend.DirectML));
    }

    [Fact]
    public void BackendWireNameMapsCpuToCpu()
    {
        Assert.Equal("cpu", BackendWireName.For(InferenceBackend.Cpu));
    }
}
