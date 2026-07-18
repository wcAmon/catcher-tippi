// 對應 mac crates/catcher-asr-host/src/protocol.rs 的 #[cfg(test)] mod tests。
using NemotronAsrHost;
using Xunit;

namespace NemotronAsrHost.Tests;

public class ProtocolTests
{
    // 對應 parses_start_command
    [Fact]
    public void ParsesStartCommand()
    {
        Assert.True(Protocol.TryParse(
            "{\"cmd\":\"start\",\"lang\":\"auto\",\"sample_rate\":16000}", out var command, out _));
        Assert.True(command is Command.Start { Lang: "auto", SampleRate: 16000 });
    }

    // 對應 parses_audio_and_stop
    [Fact]
    public void ParsesAudioAndStop()
    {
        Assert.True(Protocol.TryParse(
            "{\"cmd\":\"audio\",\"pcm16_b64\":\"AAA=\"}", out var audioCommand, out _));
        Assert.IsType<Command.Audio>(audioCommand);

        Assert.True(Protocol.TryParse("{\"cmd\":\"stop\"}", out var stopCommand, out _));
        Assert.IsType<Command.Stop>(stopCommand);
    }

    // 對應 rejects_unknown_and_malformed
    [Fact]
    public void RejectsUnknownAndMalformed()
    {
        Assert.False(Protocol.TryParse("{\"cmd\":\"dance\"}", out _, out _));
        Assert.False(Protocol.TryParse("not json", out _, out _));
    }

    // 對應協定文件「指令物件拒絕未知欄位」;mac 用 serde deny_unknown_fields 天生涵蓋,
    // C# 手動檢查故獨立成一條測試。
    [Fact]
    public void RejectsUnknownField()
    {
        Assert.False(Protocol.TryParse(
            "{\"cmd\":\"start\",\"lang\":\"auto\",\"sample_rate\":16000,\"x\":1}", out _, out _));
    }

    // 對應 emits_events_as_single_json_lines(ready 分支)
    [Fact]
    public void EmitsReadyVerbatim()
    {
        Assert.Equal("{\"event\":\"ready\",\"backend\":\"fake\"}", Protocol.EmitReady("fake"));
    }

    // 對應 emits_events_as_single_json_lines(partial 分支,含中文驗證原始 UTF-8 不跳脫)
    [Fact]
    public void EmitsPartialVerbatim()
    {
        Assert.Equal("{\"event\":\"partial\",\"text\":\"你好\"}", Protocol.EmitPartial("你好"));
    }

    // final 事件補齊(mac 測試三種之外的第四種;協定文件四事件皆需逐字驗證)
    [Fact]
    public void EmitsFinalVerbatim()
    {
        Assert.Equal("{\"event\":\"final\",\"text\":\"你好\"}", Protocol.EmitFinal("你好"));
    }

    // 對應 emits_events_as_single_json_lines(error 分支)
    [Fact]
    public void EmitsErrorVerbatim()
    {
        Assert.Equal("{\"event\":\"error\",\"message\":\"x\"}", Protocol.EmitError("x"));
    }
}
