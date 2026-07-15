using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Win32;

namespace Tippi.Windows.Services;

public sealed record InferenceLoadResult(
    NemotronEngine Engine,
    InferenceBackendPreference Preference,
    InferenceBackend SelectedBackend,
    string Detail,
    bool UsedCachedProfile);

public sealed class InferenceBackendLoader
{
    private const int ProfileVersion = 1;
    private const int ChunkSamples = 8_960;
    private readonly string _runtimeProfilePath;

    public InferenceBackendLoader(string? runtimeProfilePath = null)
    {
        _runtimeProfilePath = runtimeProfilePath ?? DefaultRuntimeProfilePath;
    }

    public Task<InferenceLoadResult> LoadAsync(
        string modelDirectory,
        InferenceBackendPreference preference,
        bool forceProbe,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        return Task.Run(
            () => Load(modelDirectory, preference, forceProbe, progress, cancellationToken),
            cancellationToken);
    }

    public void InvalidateProfile()
    {
        try
        {
            File.Delete(_runtimeProfilePath);
        }
        catch (IOException)
        {
            // A stale profile is only a performance hint. A failed deletion
            // must never prevent the model from loading.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private InferenceLoadResult Load(
        string modelDirectory,
        InferenceBackendPreference preference,
        bool forceProbe,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string signature = RuntimeSignature();

        if (preference == InferenceBackendPreference.Cpu)
        {
            progress?.Report("正在載入 CPU 語音模型…");
            return LoadSelected(modelDirectory, preference, InferenceBackend.Cpu, "已依設定使用 CPU。", false);
        }

        if (!forceProbe && preference == InferenceBackendPreference.Auto
            && TryReadProfile(signature, out RuntimeProfile? cached))
        {
            RuntimeProfile cachedProfile = cached!;
            progress?.Report($"正在載入上次測得的 {InferenceBackendPolicy.DisplayName(cachedProfile.Backend)}…");
            InferenceLoadResult loaded = LoadSelected(
                modelDirectory,
                preference,
                cachedProfile.Backend,
                $"沿用這台電腦的基準測試結果：{ProfileSummary(cachedProfile)}",
                true);
            if (loaded.SelectedBackend == cachedProfile.Backend)
            {
                return loaded;
            }

            loaded.Engine.Dispose();
            InvalidateProfile();
            progress?.Report("顯示驅動或 GPU 狀態已改變，正在重新測試…");
        }

        progress?.Report("正在測試 GPU（DirectML）相容性與速度…");
        BackendProbeResult directMl = Probe(modelDirectory, InferenceBackend.DirectML, cancellationToken);

        if (preference == InferenceBackendPreference.DirectML)
        {
            if (directMl.Succeeded)
            {
                return LoadSelected(
                    modelDirectory,
                    preference,
                    InferenceBackend.DirectML,
                    $"GPU 測試完成（{directMl.Elapsed.TotalMilliseconds:F0} ms）。",
                    false);
            }

            progress?.Report("GPU 無法使用，正在安全回退 CPU…");
            return LoadSelected(
                modelDirectory,
                preference,
                InferenceBackend.Cpu,
                $"GPU 無法使用，已回退 CPU：{directMl.Error}",
                false);
        }

        progress?.Report("正在測量 CPU，完成後會自動選擇較適合的後端…");
        BackendProbeResult cpu = Probe(modelDirectory, InferenceBackend.Cpu, cancellationToken);
        InferenceBackend selected = InferenceBackendPolicy.Select(preference, directMl, cpu);
        var profile = new RuntimeProfile(
            ProfileVersion,
            signature,
            selected,
            directMl.Succeeded ? directMl.Elapsed.TotalMilliseconds : null,
            cpu.Elapsed.TotalMilliseconds,
            DateTimeOffset.UtcNow);
        WriteProfile(profile);

        string detail = directMl.Succeeded
            ? $"自動測試：GPU {directMl.Elapsed.TotalMilliseconds:F0} ms、CPU {cpu.Elapsed.TotalMilliseconds:F0} ms。"
            : $"DirectML 不相容，使用 CPU：{directMl.Error}";
        return LoadSelected(modelDirectory, preference, selected, detail, false);
    }

    private static InferenceLoadResult LoadSelected(
        string modelDirectory,
        InferenceBackendPreference preference,
        InferenceBackend backend,
        string detail,
        bool cached)
    {
        try
        {
            return new(new NemotronEngine(modelDirectory, backend), preference, backend, detail, cached);
        }
        catch (Exception ex) when (backend == InferenceBackend.DirectML)
        {
            return new(
                new NemotronEngine(modelDirectory, InferenceBackend.Cpu),
                preference,
                InferenceBackend.Cpu,
                $"DirectML 載入失敗，已回退 CPU：{ex.Message}",
                cached);
        }
    }

    private static BackendProbeResult Probe(
        string modelDirectory,
        InferenceBackend backend,
        CancellationToken cancellationToken)
    {
        try
        {
            string probePath = Path.Combine(AppContext.BaseDirectory, "Assets", "backend-probe.wav");
            float[] probeAudio = AudioFileLoader.LoadMono16Khz(probePath);
            using var engine = new NemotronEngine(modelDirectory, backend);
            engine.BeginSession("zh-CN", useVad: false, traditionalChinese: false);

            var stopwatch = Stopwatch.StartNew();
            TranscriptionUpdate? latest = null;
            foreach (float[] chunk in probeAudio.Chunk(ChunkSamples))
            {
                cancellationToken.ThrowIfCancellationRequested();
                latest = engine.Process(chunk) ?? latest;
            }
            latest = engine.Flush() ?? latest;
            stopwatch.Stop();

            if (latest is null || string.IsNullOrWhiteSpace(latest.RawText))
            {
                throw new InvalidDataException("後端探測沒有產生任何語音辨識文字。");
            }

            return new(backend, true, stopwatch.Elapsed);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            return new(backend, false, TimeSpan.MaxValue, CompactError(ex));
        }
    }

    private bool TryReadProfile(string signature, out RuntimeProfile? profile)
    {
        profile = null;
        try
        {
            if (!File.Exists(_runtimeProfilePath))
            {
                return false;
            }

            profile = JsonSerializer.Deserialize<RuntimeProfile>(File.ReadAllText(_runtimeProfilePath));
            return profile is not null
                && profile.Version == ProfileVersion
                && profile.Signature == signature;
        }
        catch (JsonException)
        {
            return false;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private void WriteProfile(RuntimeProfile profile)
    {
        try
        {
            string? directory = Path.GetDirectoryName(_runtimeProfilePath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            File.WriteAllText(_runtimeProfilePath, JsonSerializer.Serialize(profile));
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static string RuntimeSignature()
    {
        string runtimePath = Path.Combine(AppContext.BaseDirectory, "onnxruntime-genai.dll");
        var runtime = new FileInfo(runtimePath);
        string raw = string.Join('|',
            ModelManifest.Revision,
            Environment.OSVersion.Version,
            runtime.Exists ? runtime.Length : 0,
            runtime.Exists ? runtime.LastWriteTimeUtc.Ticks : 0,
            DisplayDriverSignature());
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(raw))).ToLowerInvariant();
    }

    private static string DisplayDriverSignature()
    {
        try
        {
            using RegistryKey? video = Registry.LocalMachine.OpenSubKey(
                @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}");
            if (video is null)
            {
                return "unknown";
            }

            var entries = new List<string>();
            foreach (string name in video.GetSubKeyNames())
            {
                using RegistryKey? adapter = video.OpenSubKey(name);
                string? description = adapter?.GetValue("DriverDesc")?.ToString();
                string? version = adapter?.GetValue("DriverVersion")?.ToString();
                if (!string.IsNullOrWhiteSpace(description))
                {
                    entries.Add($"{description}:{version}");
                }
            }
            entries.Sort(StringComparer.Ordinal);
            return string.Join(';', entries);
        }
        catch
        {
            return "unknown";
        }
    }

    private static string ProfileSummary(RuntimeProfile profile)
    {
        string dml = profile.DirectMlMilliseconds is double value ? $"GPU {value:F0} ms、" : string.Empty;
        return $"{dml}CPU {profile.CpuMilliseconds:F0} ms";
    }

    private static string CompactError(Exception exception)
    {
        string message = exception.GetBaseException().Message.ReplaceLineEndings(" ").Trim();
        return message.Length <= 240 ? message : message[..240] + "…";
    }

    private static string DefaultRuntimeProfilePath
    {
        get
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(local, "Tippi", "runtime-profile-v1.json");
        }
    }

    private sealed record RuntimeProfile(
        int Version,
        string Signature,
        InferenceBackend Backend,
        double? DirectMlMilliseconds,
        double CpuMilliseconds,
        DateTimeOffset MeasuredAt);
}
