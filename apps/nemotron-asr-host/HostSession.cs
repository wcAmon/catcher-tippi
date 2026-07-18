// 會話狀態機:一行指令 → 零至多行已序列化事件。不做 I/O(Program.cs 負責)。
// 語義鏡射 crates/catcher-asr-host/src/session.rs;差異僅在引擎 API 形狀
// (Push 直接回全文,因此不需要另外 decode)。
namespace NemotronAsrHost;

public sealed class HostSession(IAsrEngine engine)
{
    private bool _active;
    private string _lastEmitted = "";

    public string Backend => engine.Backend;

    public IReadOnlyList<string> Handle(string line)
    {
        if (!Protocol.TryParse(line, out var command, out var parseError))
        {
            return [Protocol.EmitError(parseError)];
        }
        switch (command)
        {
            case Command.Start start when start.SampleRate != 16000:
                return [Protocol.EmitError($"sample_rate 僅支援 16000,收到 {start.SampleRate}")];
            case Command.Start when _active:
                return [Protocol.EmitError("會話進行中,請先 stop")];
            case Command.Start:
                // start.lang 為 v1 保留欄位:接受但忽略(語言由 --language 決定)。
                try { engine.Begin(); }
                catch (Exception e) { return [Protocol.EmitError($"引擎重置失敗:{e.Message}")]; }
                _active = true;
                _lastEmitted = "";
                return [];
            case Command.Audio when !_active:
                return [Protocol.EmitError("尚未 start")];
            case Command.Audio audio:
            {
                float[] samples;
                try { samples = DecodePcm16(audio.Pcm16B64); }
                catch (FormatException) { return [Protocol.EmitError("pcm16_b64 非法 base64")]; }
                catch (ArgumentException e) { return [Protocol.EmitError(e.Message)]; }
                string? text;
                try { text = engine.Push(samples); }
                catch (Exception e) { return [Protocol.EmitError(e.Message)]; }
                if (text is null || text == _lastEmitted)
                {
                    return [];   // 協定:partial 僅在內容變化時輸出
                }
                _lastEmitted = text;
                return [Protocol.EmitPartial(text)];
            }
            case Command.Stop when !_active:
                return [Protocol.EmitError("尚未 start")];
            case Command.Stop:
            {
                _active = false;   // 協定:stop 失敗 → error 且會話視為已結束、無 final
                try { return [Protocol.EmitFinal(engine.Finish())]; }
                catch (Exception e) { return [Protocol.EmitError(e.Message)]; }
            }
            default:
                return [Protocol.EmitError("未知指令")];
        }
    }

    /// PCM16-LE bytes → float samples(±1.0)。奇數位元組數為格式錯誤。
    private static float[] DecodePcm16(string b64)
    {
        byte[] bytes = Convert.FromBase64String(b64);
        if (bytes.Length % 2 != 0)
        {
            throw new ArgumentException("pcm16_b64 位元組數必須為偶數");
        }
        var samples = new float[bytes.Length / 2];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = BitConverter.ToInt16(bytes, i * 2) / 32768f;
        }
        return samples;
    }
}
