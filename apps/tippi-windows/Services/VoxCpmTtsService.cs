using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace Tippi.Windows.Services;

public enum TtsBackendPreference
{
    Auto,
    Vulkan,
    Cpu,
}

public enum TtsBackend
{
    Vulkan,
    Cpu,
}

public sealed record TtsSynthesisResult(byte[] WaveData, TtsBackend Backend, TimeSpan Elapsed);

public sealed class VoxCpmTtsService : IAsyncDisposable
{
    private static readonly TimeSpan StartupTimeout = TimeSpan.FromMinutes(5);
    private readonly HttpClient _httpClient = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly StringBuilder _processLog = new();
    private Process? _process;
    private Uri? _endpoint;

    public TtsBackend? Backend { get; private set; }

    public async Task<TtsBackend> EnsureStartedAsync(
        VoxCpmModelInstaller installer,
        TtsBackendPreference preference,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        if (_process is { HasExited: false } && Backend is not null &&
            (preference == TtsBackendPreference.Auto || Matches(preference, Backend.Value)))
        {
            return Backend.Value;
        }

        await StopAsync();
        IReadOnlyList<TtsBackend> candidates = preference switch
        {
            TtsBackendPreference.Vulkan => [TtsBackend.Vulkan],
            TtsBackendPreference.Cpu => [TtsBackend.Cpu],
            _ => [TtsBackend.Vulkan, TtsBackend.Cpu],
        };

        var failures = new List<string>();
        foreach (TtsBackend candidate in candidates)
        {
            string executable = RuntimeExecutable(candidate);
            if (!File.Exists(executable))
            {
                failures.Add($"{DisplayName(candidate)} runtime 不存在：{executable}");
                continue;
            }

            try
            {
                progress?.Report($"正在用 {DisplayName(candidate)} 載入 VoxCPM2…");
                await StartCandidateAsync(executable, candidate, installer, cancellationToken);
                if (candidate == TtsBackend.Vulkan &&
                    !RecentLog().Contains("backend=Vulkan", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidOperationException("Vulkan runtime 未找到可用 GPU backend。");
                }
                Backend = candidate;
                return candidate;
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                failures.Add($"{DisplayName(candidate)}：模型載入超過 {StartupTimeout.TotalMinutes:F0} 分鐘");
                await StopAsync();
            }
            catch (OperationCanceledException)
            {
                await StopAsync();
                throw;
            }
            catch (Exception ex)
            {
                failures.Add($"{DisplayName(candidate)}：{ex.GetBaseException().Message}");
                await StopAsync();
            }
        }

        throw new InvalidOperationException(
            "VoxCPM2 無法啟動。" + Environment.NewLine + string.Join(Environment.NewLine, failures));
    }

    public async Task<TtsSynthesisResult> SynthesizeAsync(
        string text,
        int inferenceTimesteps,
        CancellationToken cancellationToken)
    {
        if (_process is null || _process.HasExited || _endpoint is null || Backend is null)
        {
            throw new InvalidOperationException("VoxCPM2 TTS runtime 尚未啟動。");
        }

        var payload = new
        {
            model = "voxcpm",
            input = text,
            voice = "default",
            response_format = "wav",
            seed = 42,
            cfg_value = 2.0,
            inference_timesteps = inferenceTimesteps,
            max_steps = 200,
            temperature = 1.0,
        };
        using var request = new HttpRequestMessage(HttpMethod.Post, new Uri(_endpoint, "/v1/audio/speech"))
        {
            Content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json"),
        };
        var stopwatch = Stopwatch.StartNew();
        using HttpResponseMessage response = await _httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);
        byte[] content = await response.Content.ReadAsByteArrayAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            string message = Encoding.UTF8.GetString(content);
            throw new InvalidOperationException($"VoxCPM2 合成失敗 ({(int)response.StatusCode})：{message}");
        }
        stopwatch.Stop();
        if (content.Length < 44 || content[0] != (byte)'R' || content[1] != (byte)'I')
        {
            throw new InvalidDataException("VoxCPM2 回傳的資料不是有效 WAV 音訊。");
        }
        return new(content, Backend.Value, stopwatch.Elapsed);
    }

    public async Task<TtsSynthesisResult> SynthesizeWithFallbackAsync(
        VoxCpmModelInstaller installer,
        TtsBackendPreference preference,
        string text,
        int inferenceTimesteps,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        TtsBackend backend = await EnsureStartedAsync(
            installer,
            preference,
            progress,
            cancellationToken);
        progress?.Report($"正在用 {DisplayName(backend)} 合成（{inferenceTimesteps} steps）…");
        try
        {
            return await SynthesizeAsync(text, inferenceTimesteps, cancellationToken);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception gpuFailure) when (
            preference == TtsBackendPreference.Auto && backend == TtsBackend.Vulkan)
        {
            progress?.Report("Vulkan 合成失敗，正在卸載 GPU 模型並改用 CPU…");
            await StopAsync();
            try
            {
                await EnsureStartedAsync(
                    installer,
                    TtsBackendPreference.Cpu,
                    progress,
                    cancellationToken);
                progress?.Report($"正在用 CPU 合成（{inferenceTimesteps} steps）…");
                return await SynthesizeAsync(text, inferenceTimesteps, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception cpuFailure)
            {
                throw new InvalidOperationException(
                    $"VoxCPM2 的 Vulkan 與 CPU 推論都失敗。{Environment.NewLine}" +
                    $"Vulkan：{gpuFailure.GetBaseException().Message}{Environment.NewLine}" +
                    $"CPU：{cpuFailure.GetBaseException().Message}",
                    new AggregateException(gpuFailure, cpuFailure));
            }
        }
    }

    private async Task StartCandidateAsync(
        string executable,
        TtsBackend candidate,
        VoxCpmModelInstaller installer,
        CancellationToken cancellationToken)
    {
        int port = ReserveLoopbackPort();
        _endpoint = new Uri($"http://127.0.0.1:{port}");
        lock (_processLog)
        {
            _processLog.Clear();
        }

        var startInfo = new ProcessStartInfo(executable)
        {
            WorkingDirectory = Path.GetDirectoryName(executable)!,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        startInfo.ArgumentList.Add("--voxcpm2-base-lm");
        startInfo.ArgumentList.Add(installer.BaseModelPath);
        startInfo.ArgumentList.Add("--voxcpm2-acoustic");
        startInfo.ArgumentList.Add(installer.AcousticModelPath);
        startInfo.ArgumentList.Add("--voxcpm2-n-gpu-layers");
        startInfo.ArgumentList.Add(candidate == TtsBackend.Vulkan ? "99" : "0");
        startInfo.ArgumentList.Add("--host");
        startInfo.ArgumentList.Add("127.0.0.1");
        startInfo.ArgumentList.Add("--port");
        startInfo.ArgumentList.Add(port.ToString(System.Globalization.CultureInfo.InvariantCulture));

        _process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        _process.OutputDataReceived += CaptureProcessLog;
        _process.ErrorDataReceived += CaptureProcessLog;
        if (!_process.Start())
        {
            throw new InvalidOperationException("無法啟動 llama-tts-server。");
        }
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();
        if (candidate == TtsBackend.Cpu)
        {
            try { _process.PriorityClass = ProcessPriorityClass.BelowNormal; }
            catch { /* Priority is an optimization, not a startup requirement. */ }
        }

        using var startupTimeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        startupTimeout.CancelAfter(StartupTimeout);
        while (true)
        {
            startupTimeout.Token.ThrowIfCancellationRequested();
            if (_process.HasExited)
            {
                throw new InvalidOperationException(
                    $"llama-tts-server 提前結束 (exit {_process.ExitCode})。{Environment.NewLine}{RecentLog()}");
            }
            try
            {
                using var healthTimeout = CancellationTokenSource.CreateLinkedTokenSource(startupTimeout.Token);
                healthTimeout.CancelAfter(TimeSpan.FromSeconds(2));
                using HttpResponseMessage response = await _httpClient.GetAsync(
                    new Uri(_endpoint, "/health"),
                    healthTimeout.Token);
                if (response.IsSuccessStatusCode)
                {
                    return;
                }
            }
            catch (OperationCanceledException) when (!startupTimeout.IsCancellationRequested)
            {
            }
            catch (HttpRequestException)
            {
            }
            await Task.Delay(500, startupTimeout.Token);
        }
    }

    private void CaptureProcessLog(object sender, DataReceivedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(e.Data))
        {
            return;
        }
        lock (_processLog)
        {
            _processLog.AppendLine(e.Data);
            if (_processLog.Length > 24_000)
            {
                _processLog.Remove(0, _processLog.Length - 16_000);
            }
        }
    }

    private string RecentLog()
    {
        lock (_processLog)
        {
            return _processLog.ToString().Trim();
        }
    }

    public async Task StopAsync()
    {
        Process? process = _process;
        _process = null;
        _endpoint = null;
        Backend = null;
        if (process is null)
        {
            return;
        }
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                await process.WaitForExitAsync(timeout.Token);
            }
        }
        catch
        {
            // Process shutdown must not block the app from closing or falling back.
        }
        finally
        {
            process.Dispose();
        }
    }

    private static int ReserveLoopbackPort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static string RuntimeExecutable(TtsBackend backend)
    {
        string directory = backend == TtsBackend.Vulkan ? "vulkan" : "cpu";
        return Path.Combine(AppContext.BaseDirectory, "TtsRuntime", directory, "llama-tts-server.exe");
    }

    private static bool Matches(TtsBackendPreference preference, TtsBackend backend) =>
        (preference == TtsBackendPreference.Vulkan && backend == TtsBackend.Vulkan) ||
        (preference == TtsBackendPreference.Cpu && backend == TtsBackend.Cpu);

    public static string DisplayName(TtsBackend backend) => backend switch
    {
        TtsBackend.Vulkan => "GPU / Vulkan",
        _ => "CPU",
    };

    public async ValueTask DisposeAsync()
    {
        await StopAsync();
        _httpClient.Dispose();
    }
}
