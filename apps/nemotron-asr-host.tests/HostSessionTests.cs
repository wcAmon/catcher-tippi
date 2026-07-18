// 對應 mac crates/catcher-asr-host/src/session.rs 的 #[cfg(test)] mod tests(用 FakeEngine)。
using System.Text.Json;
using NemotronAsrHost;
using Xunit;

namespace NemotronAsrHost.Tests;

public class HostSessionTests
{
    private static string B64Pcm(int samples) => Convert.ToBase64String(new byte[samples * 2]);

    private static string StartLine(int sampleRate = 16000) =>
        $"{{\"cmd\":\"start\",\"lang\":\"auto\",\"sample_rate\":{sampleRate}}}";

    private static string AudioLine(string b64) => $"{{\"cmd\":\"audio\",\"pcm16_b64\":\"{b64}\"}}";

    private const string StopLine = "{\"cmd\":\"stop\"}";

    private static string EventType(string json) =>
        JsonDocument.Parse(json).RootElement.GetProperty("event").GetString()!;

    private static string Field(string json, string name) =>
        JsonDocument.Parse(json).RootElement.GetProperty(name).GetString()!;

    // 對應 happy_path_start_audio_stop
    [Fact]
    public void HappyPathStartAudioStop()
    {
        var session = new HostSession(new FakeEngine());
        Assert.Empty(session.Handle(StartLine()));

        var events = session.Handle(AudioLine(B64Pcm(1600)));
        var partial = Assert.Single(events);
        Assert.Equal("partial", EventType(partial));
        Assert.Equal("字0", Field(partial, "text"));

        events = session.Handle(StopLine);
        var final = Assert.Single(events);
        Assert.Equal("final", EventType(final));
        Assert.Equal("字0", Field(final, "text"));
    }

    // 對應 partial_only_when_new_tokens
    [Fact]
    public void PartialOnlyWhenNewTokens()
    {
        var session = new HostSession(new FakeEngine());
        session.Handle(StartLine());
        // 不足 1600 samples → FakeEngine 不吐 token → 不得有 partial
        var events = session.Handle(AudioLine(B64Pcm(100)));
        Assert.Empty(events);
    }

    // 對應 state_errors_do_not_kill_session
    [Fact]
    public void StateErrorsDoNotKillSession()
    {
        var session = new HostSession(new FakeEngine());
        Assert.Equal("error", EventType(Assert.Single(session.Handle(StopLine))));
        Assert.Equal("error", EventType(Assert.Single(session.Handle(AudioLine(B64Pcm(1600))))));

        session.Handle(StartLine());
        Assert.Equal("error", EventType(Assert.Single(session.Handle(StartLine()))));

        // 原會話仍活著
        var events = session.Handle(AudioLine(B64Pcm(1600)));
        Assert.Equal("partial", EventType(Assert.Single(events)));
    }

    // 對應 rejects_bad_audio_payload
    [Fact]
    public void RejectsBadAudioPayload()
    {
        var session = new HostSession(new FakeEngine());
        session.Handle(StartLine());

        // 非法 base64
        Assert.Equal("error", EventType(Assert.Single(session.Handle(AudioLine("!!!")))));

        // 奇數位元組
        var odd = Convert.ToBase64String(new byte[3]);
        Assert.Equal("error", EventType(Assert.Single(session.Handle(AudioLine(odd)))));
    }

    // 對應 rejects_wrong_sample_rate
    [Fact]
    public void RejectsWrongSampleRate()
    {
        var session = new HostSession(new FakeEngine());
        var events = session.Handle(StartLine(44100));
        Assert.Equal("error", EventType(Assert.Single(events)));
    }

    // 對應 second_session_starts_fresh
    [Fact]
    public void SecondSessionStartsFresh()
    {
        var session = new HostSession(new FakeEngine());

        session.Handle(StartLine());
        var events = session.Handle(AudioLine(B64Pcm(1600)));
        Assert.Equal("字0", Field(Assert.Single(events), "text"));
        events = session.Handle(StopLine);
        Assert.Equal("字0", Field(Assert.Single(events), "text"));

        // 第二個會話:引擎必須重置,id 從頭再來,partial/final 再度是「字0」。
        session.Handle(StartLine());
        events = session.Handle(AudioLine(B64Pcm(1600)));
        Assert.Equal("字0", Field(Assert.Single(events), "text"));
        events = session.Handle(StopLine);
        Assert.Equal("字0", Field(Assert.Single(events), "text"));
    }

    /// finish() 永遠失敗的測試引擎,用來驗證 stop 失敗分支:
    /// 會話視為已結束,不會有 final,後續指令都收到 error。
    private sealed class FailingEngine : IAsrEngine
    {
        public string Backend => "failing";
        public void Begin() { }
        public string? Push(float[] samples) => null;
        public string Finish() => throw new InvalidOperationException("boom");
    }

    // 對應 stop_failure_ends_session_without_final
    [Fact]
    public void StopFailureEndsSessionWithoutFinal()
    {
        var session = new HostSession(new FailingEngine());

        session.Handle(StartLine());
        var events = session.Handle(StopLine);
        Assert.Equal("error", EventType(Assert.Single(events)));

        // 會話已結束:後續 audio 收到 error,而不是被當作進行中的會話處理。
        events = session.Handle(AudioLine(B64Pcm(1600)));
        Assert.Equal("error", EventType(Assert.Single(events)));
    }
}
