using Microsoft.ML.OnnxRuntimeGenAI;
using OpenccNetLib;

namespace Tippi.Windows.Services;

public sealed record TranscriptionUpdate(string RawText, string DisplayText);

public sealed class NemotronEngine : IDisposable
{
    private static readonly IReadOnlyDictionary<string, int> LanguageIds = new Dictionary<string, int>
    {
        ["en-US"] = 0,
        ["en-GB"] = 1,
        ["es-ES"] = 2,
        ["es-US"] = 3,
        ["zh-CN"] = 4,
        ["hi-IN"] = 6,
        ["ar-AR"] = 7,
        ["fr-FR"] = 8,
        ["de-DE"] = 9,
        ["ja-JP"] = 10,
        ["ru-RU"] = 11,
        ["pt-BR"] = 12,
        ["pt-PT"] = 13,
        ["ko-KR"] = 14,
        ["it-IT"] = 15,
        ["nl-NL"] = 16,
        ["pl-PL"] = 17,
        ["tr-TR"] = 18,
        ["uk-UA"] = 19,
        ["auto"] = 101,
    };

    private readonly Model _model;
    private readonly Opencc _opencc = new("s2twp");
    private StreamingProcessor? _processor;
    private Tokenizer? _tokenizer;
    private TokenizerStream? _tokenizerStream;
    private GeneratorParams? _generatorParams;
    private Generator? _generator;
    private string _rawText = string.Empty;
    private bool _traditionalChinese;

    public NemotronEngine(string modelDirectory)
        : this(modelDirectory, InferenceBackend.Cpu)
    {
    }

    public NemotronEngine(string modelDirectory, InferenceBackend backend)
    {
        using var config = new Config(modelDirectory);
        config.ClearProviders();
        if (backend == InferenceBackend.DirectML)
        {
            config.AppendProvider("DML");
        }

        _model = new Model(config);
        Backend = backend;
    }

    public InferenceBackend Backend { get; }

    public void BeginSession(string language, bool useVad, bool traditionalChinese)
    {
        EndSessionObjects();
        _rawText = string.Empty;
        _traditionalChinese = traditionalChinese;
        _processor = new StreamingProcessor(_model);
        _processor.SetOption("use_vad", useVad ? "true" : "false");
        _tokenizer = new Tokenizer(_model);
        _tokenizerStream = _tokenizer.CreateStream();
        _generatorParams = new GeneratorParams(_model);
        _generator = new Generator(_model, _generatorParams);
        int languageId = LanguageIds.TryGetValue(language, out int id) ? id : LanguageIds["auto"];
        GeneratorRuntimeOptions.Set(
            _generator,
            "lang_id",
            languageId.ToString(System.Globalization.CultureInfo.InvariantCulture));
    }

    public TranscriptionUpdate? Process(float[] samples)
    {
        EnsureSession();
        using NamedTensors? inputs = _processor!.Process(samples);
        return inputs is null ? null : Decode(inputs);
    }

    public TranscriptionUpdate? Flush()
    {
        EnsureSession();
        using NamedTensors? inputs = _processor!.Flush();
        return inputs is null ? CurrentUpdate() : Decode(inputs);
    }

    private TranscriptionUpdate Decode(NamedTensors inputs)
    {
        _generator!.SetInputs(inputs);
        while (!_generator.IsDone())
        {
            _generator.GenerateNextToken();
            ReadOnlySpan<int> tokens = _generator.GetNextTokens();
            if (tokens.Length > 0)
            {
                _rawText += _tokenizerStream!.Decode(tokens[0]);
            }
        }
        return CurrentUpdate();
    }

    private TranscriptionUpdate CurrentUpdate()
    {
        string display = _traditionalChinese ? _opencc.Convert(_rawText) : _rawText;
        return new(_rawText, display);
    }

    private void EnsureSession()
    {
        if (_processor is null || _generator is null)
        {
            throw new InvalidOperationException("尚未開始語音辨識工作階段。");
        }
    }

    private void EndSessionObjects()
    {
        _generator?.Dispose();
        _generator = null;
        _generatorParams?.Dispose();
        _generatorParams = null;
        _tokenizerStream?.Dispose();
        _tokenizerStream = null;
        _tokenizer?.Dispose();
        _tokenizer = null;
        _processor?.Dispose();
        _processor = null;
    }

    public void Dispose()
    {
        EndSessionObjects();
        _model.Dispose();
    }
}
