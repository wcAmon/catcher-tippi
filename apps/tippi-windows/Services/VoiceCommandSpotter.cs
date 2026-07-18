using SherpaOnnx;

namespace Tippi.Windows.Services;

public sealed class VoiceCommandSpotter : IDisposable
{
    public const string SubmitEvent = "SUBMIT_ZH";
    private readonly KeywordSpotter _spotter;
    private OnlineStream? _stream;

    public VoiceCommandSpotter(KeywordModelInstaller installer)
    {
        var config = new KeywordSpotterConfig();
        config.FeatConfig.SampleRate = 16_000;
        config.FeatConfig.FeatureDim = 80;
        config.ModelConfig.Transducer.Encoder = installer.EncoderPath;
        config.ModelConfig.Transducer.Decoder = installer.DecoderPath;
        config.ModelConfig.Transducer.Joiner = installer.JoinerPath;
        config.ModelConfig.Tokens = installer.TokensPath;
        config.ModelConfig.Provider = "cpu";
        config.ModelConfig.NumThreads = 1;
        config.ModelConfig.Debug = 0;
        config.MaxActivePaths = 4;
        config.NumTrailingBlanks = 1;
        config.KeywordsScore = 1.5f;
        config.KeywordsThreshold = 0.25f;
        config.KeywordsFile = installer.KeywordsPath;
        _spotter = new KeywordSpotter(config);
    }

    public void BeginSession()
    {
        _stream?.Dispose();
        _stream = _spotter.CreateStream();
    }

    public bool Process(float[] samples)
    {
        OnlineStream stream = _stream
            ?? throw new InvalidOperationException("尚未開始 Voice Command 工作階段。");
        stream.AcceptWaveform(16_000, samples);
        return DecodeReady(stream);
    }

    public bool Finish()
    {
        if (_stream is null)
        {
            return false;
        }
        _stream.AcceptWaveform(16_000, new float[4_800]);
        _stream.InputFinished();
        return DecodeReady(_stream);
    }

    private bool DecodeReady(OnlineStream stream)
    {
        bool detected = false;
        while (_spotter.IsReady(stream))
        {
            _spotter.Decode(stream);
            KeywordResult result = _spotter.GetResult(stream);
            if (string.Equals(result.Keyword, SubmitEvent, StringComparison.Ordinal) ||
                string.Equals(result.Keyword, "幫我送出", StringComparison.Ordinal))
            {
                detected = true;
                _spotter.Reset(stream);
            }
        }
        return detected;
    }

    public void Dispose()
    {
        _stream?.Dispose();
        _stream = null;
        _spotter.Dispose();
    }
}
