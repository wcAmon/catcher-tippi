using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading.Channels;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using Microsoft.Win32;
using Tippi.Windows.Services;

namespace Tippi.Windows;

public partial class MainWindow : Window
{
    private readonly ModelInstaller _installer = new();
    private readonly DiarizationModelInstaller _diarizationInstaller = new();
    private readonly InferenceBackendLoader _backendLoader = new();
    private readonly InferencePreferenceStore _preferenceStore = new();
    private readonly CancellationTokenSource _lifetime = new();
    private NemotronEngine? _engine;
    private SpeakerDiarizer? _diarizer;
    private MicrophoneCapture? _microphone;
    private Channel<float[]>? _audioChannel;
    private Task<RecognitionSessionResult>? _recognitionWorker;
    private TextInjectionCoordinator? _injectionCoordinator;
    private bool _recordingDiarization;
    private bool _recordingTraditional;
    private bool _busy;
    private bool _shutdownComplete;
    private bool _windowLoaded;

    public MainWindow()
    {
        InitializeComponent();
        SetBackendBox(_preferenceStore.Load());
    }

    private async void Window_Loaded(object sender, RoutedEventArgs e)
    {
        _windowLoaded = true;
        nint handle = new WindowInteropHelper(this).Handle;
        _injectionCoordinator = new TextInjectionCoordinator(new WindowsTextInjector(handle));
        await PrepareModelAsync(repair: false);
    }

    private async Task PrepareModelAsync(bool repair)
    {
        if (_busy)
        {
            return;
        }

        SetBusy(true);
        PrepareButton.IsEnabled = false;
        try
        {
            long allModelBytes = ModelManifest.TotalBytes + DiarizationModelManifest.TotalBytes;
            IProgress<ModelInstallProgress> asrProgress = CombinedProgress(0, allModelBytes, "語音辨識");
            IProgress<ModelInstallProgress> diarizationProgress = CombinedProgress(
                ModelManifest.TotalBytes,
                allModelBytes,
                "說話者分離");

            if (repair)
            {
                StatusText.Text = "正在校驗並修復本機模型…";
                await _installer.RepairAsync(asrProgress, _lifetime.Token);
                await _diarizationInstaller.RepairAsync(diarizationProgress, _lifetime.Token);
            }
            else
            {
                if (!_installer.IsInstalled())
                {
                    StatusText.Text = $"首次使用：正在下載本機模型（合計約 {FormatBytes(allModelBytes)}）…";
                    await _installer.InstallAsync(asrProgress, _lifetime.Token);
                }
                if (!_diarizationInstaller.IsInstalled())
                {
                    StatusText.Text = "正在下載 CPU 說話者分離模型（約 39.9 MiB）…";
                    await _diarizationInstaller.InstallAsync(diarizationProgress, _lifetime.Token);
                }
            }

            if (_engine is null)
            {
                ProgressText.Text = "第一次載入需要一些時間；建議至少 8 GB RAM，並保留約 4 GB 可用記憶體。";
                InferenceLoadResult loaded = await _backendLoader.LoadAsync(
                    _installer.ModelDirectory,
                    SelectedBackendPreference(),
                    forceProbe: repair,
                    new Progress<string>(message => StatusText.Text = message),
                    _lifetime.Token);
                _engine = loaded.Engine;
                UpdateRuntimePresentation(loaded);
            }

            ModelProgress.Value = 1;
            StatusText.Text = $"準備完成：{CurrentBackendName()}，可開始錄音或選擇音訊檔。";
            ProgressText.Text += $"  模型位於：{ModelsRootDirectory()}";
        }
        catch (OperationCanceledException) when (_lifetime.IsCancellationRequested)
        {
            return;
        }
        catch (Exception ex)
        {
            StatusText.Text = "模型準備失敗。";
            ProgressText.Text = ex.Message;
            MessageBox.Show(this, ex.Message, "模型準備失敗", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            SetBusy(false);
            PrepareButton.IsEnabled = true;
        }
    }

    private Progress<ModelInstallProgress> CombinedProgress(long offset, long allModelBytes, string category)
    {
        return new Progress<ModelInstallProgress>(value =>
        {
            long completed = offset + value.CompletedBytes;
            ModelProgress.Value = (double)completed / allModelBytes;
            ProgressText.Text = $"{category} · {value.FileName}  {FormatBytes(completed)} / {FormatBytes(allModelBytes)}";
        });
    }

    private async void PrepareButton_Click(object sender, RoutedEventArgs e)
    {
        if (_busy)
        {
            return;
        }

        _diarizer?.Dispose();
        _diarizer = null;
        _engine?.Dispose();
        _engine = null;
        _backendLoader.InvalidateProfile();
        await PrepareModelAsync(repair: true);
    }

    private async void BackendBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_windowLoaded)
        {
            return;
        }

        InferenceBackendPreference preference = SelectedBackendPreference();
        _preferenceStore.Save(preference);
        if (_busy || !_installer.IsInstalled())
        {
            return;
        }

        SetBusy(true);
        try
        {
            _engine?.Dispose();
            _engine = null;
            if (preference == InferenceBackendPreference.Auto)
            {
                _backendLoader.InvalidateProfile();
            }

            InferenceLoadResult loaded = await _backendLoader.LoadAsync(
                _installer.ModelDirectory,
                preference,
                forceProbe: true,
                new Progress<string>(message => StatusText.Text = message),
                _lifetime.Token);
            _engine = loaded.Engine;
            UpdateRuntimePresentation(loaded);
            StatusText.Text = $"已切換至 {CurrentBackendName()}。";
        }
        catch (OperationCanceledException) when (_lifetime.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            ShowError("無法切換運算後端", ex);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void StartButton_Click(object sender, RoutedEventArgs e)
    {
        if (_engine is null || _busy)
        {
            return;
        }

        TranscriptBox.Clear();
        SetBusy(true);
        StopButton.IsEnabled = true;
        StatusText.Text = "正在錄音與即時辨識；停止後會分析不同說話者。切到其他程式即可使用語音輸入。";

        string language = SelectedLanguage();
        bool useVad = VadCheck.IsChecked == true;
        bool inject = InjectCheck.IsChecked == true;
        _recordingTraditional = TraditionalCheck.IsChecked == true;
        _recordingDiarization = DiarizationCheck.IsChecked == true;
        _injectionCoordinator?.Reset();
        _audioChannel = Channel.CreateUnbounded<float[]>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
        });

        try
        {
            await Task.Run(
                () => BeginSessionWithFallback(language, useVad, _recordingTraditional),
                _lifetime.Token);
            _recognitionWorker = RunRecognitionWorkerAsync(
                _audioChannel.Reader,
                language,
                useVad,
                _recordingTraditional,
                inject,
                _lifetime.Token);
            _microphone = new MicrophoneCapture();
            _microphone.SamplesAvailable += samples => _audioChannel.Writer.TryWrite(samples);
            _microphone.Stopped += error => _audioChannel.Writer.TryComplete(error);
            _microphone.Start();
        }
        catch (Exception ex)
        {
            _audioChannel.Writer.TryComplete(ex);
            FinishBusyOperation();
            ShowError("無法開始錄音", ex);
        }
    }

    private async void StopButton_Click(object sender, RoutedEventArgs e)
    {
        StopButton.IsEnabled = false;
        StatusText.Text = "正在完成最後一段辨識…";
        _microphone?.Stop();

        try
        {
            if (_recognitionWorker is not null)
            {
                RecognitionSessionResult result = await _recognitionWorker;
                await ApplySpeakerLabelsAsync(result, _recordingDiarization, _recordingTraditional);
            }
        }
        catch (Exception ex)
        {
            ShowError("語音辨識發生錯誤", ex);
        }
        finally
        {
            FinishBusyOperation();
        }
    }

    private Task<RecognitionSessionResult> RunRecognitionWorkerAsync(
        ChannelReader<float[]> reader,
        string language,
        bool useVad,
        bool traditional,
        bool inject,
        CancellationToken cancellationToken)
    {
        return Task.Run(async () =>
        {
            var audio = new List<float>();
            var collector = new TimedTranscriptCollector();
            TranscriptionUpdate latest = new(string.Empty, string.Empty);

            await foreach (float[] samples in reader.ReadAllAsync(cancellationToken))
            {
                audio.AddRange(samples);
                TranscriptionUpdate? update;
                try
                {
                    update = _engine!.Process(samples);
                }
                catch (Exception ex) when (_engine?.Backend == InferenceBackend.DirectML)
                {
                    SwitchToCpu(language, useVad, traditional, ex);
                    collector = new TimedTranscriptCollector();
                    latest = new(string.Empty, string.Empty);
                    int replayedSamples = 0;
                    foreach (float[] replayChunk in audio.ToArray().Chunk(8_960))
                    {
                        update = _engine.Process(replayChunk);
                        replayedSamples += replayChunk.Length;
                        if (update is not null)
                        {
                            latest = update;
                            collector.Update(update.RawText, replayedSamples / 16_000d);
                            PublishUpdate(update, inject);
                        }
                    }
                    continue;
                }
                if (update is not null)
                {
                    latest = update;
                    collector.Update(update.RawText, audio.Count / 16_000d);
                    PublishUpdate(update, inject);
                }
            }

            TranscriptionUpdate? final;
            try
            {
                final = _engine!.Flush();
            }
            catch (Exception ex) when (_engine?.Backend == InferenceBackend.DirectML)
            {
                SwitchToCpu(language, useVad, traditional, ex);
                collector = new TimedTranscriptCollector();
                latest = new(string.Empty, string.Empty);
                int replayedSamples = 0;
                foreach (float[] replayChunk in audio.ToArray().Chunk(8_960))
                {
                    TranscriptionUpdate? replay = _engine.Process(replayChunk);
                    replayedSamples += replayChunk.Length;
                    if (replay is not null)
                    {
                        latest = replay;
                        collector.Update(replay.RawText, replayedSamples / 16_000d);
                        PublishUpdate(replay, inject);
                    }
                }
                final = _engine.Flush();
            }
            if (final is not null)
            {
                latest = final;
                collector.Update(final.RawText, audio.Count / 16_000d);
                PublishUpdate(final, inject);
            }
            if (inject)
            {
                _injectionCoordinator?.Finish();
            }

            return new RecognitionSessionResult(
                audio.ToArray(),
                collector.Chunks.ToArray(),
                latest.RawText,
                latest.DisplayText);
        }, cancellationToken);
    }

    private void PublishUpdate(TranscriptionUpdate update, bool inject)
    {
        if (inject)
        {
            _injectionCoordinator?.Update(update.DisplayText);
        }
        Dispatcher.BeginInvoke(() =>
        {
            TranscriptBox.Text = update.DisplayText;
            TranscriptBox.ScrollToEnd();
        });
    }

    private async void FileButton_Click(object sender, RoutedEventArgs e)
    {
        if (_engine is null || _busy)
        {
            return;
        }
        var dialog = new OpenFileDialog
        {
            Title = "選擇要轉錄的音訊檔",
            Filter = "音訊檔|*.wav;*.mp3;*.m4a;*.wma|所有檔案|*.*",
        };
        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        SetBusy(true);
        TranscriptBox.Clear();
        StatusText.Text = "正在讀取並辨識音訊檔…";
        bool traditional = TraditionalCheck.IsChecked == true;
        bool diarization = DiarizationCheck.IsChecked == true;
        try
        {
            string language = SelectedLanguage();
            bool vad = VadCheck.IsChecked == true;
            RecognitionSessionResult result = await Task.Run(
                () => TranscribeAudioFile(dialog.FileName, language, vad, traditional, _lifetime.Token),
                _lifetime.Token);
            await ApplySpeakerLabelsAsync(result, diarization, traditional);
        }
        catch (Exception ex)
        {
            ShowError("音訊檔轉錄失敗", ex);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private RecognitionSessionResult TranscribeAudioFile(
        string fileName,
        string language,
        bool vad,
        bool traditional,
        CancellationToken cancellationToken)
    {
        float[] audio = AudioFileLoader.LoadMono16Khz(fileName);
        try
        {
            return TranscribeAudio(audio, language, vad, traditional, cancellationToken);
        }
        catch (Exception ex) when (_engine?.Backend == InferenceBackend.DirectML)
        {
            SwitchToCpu(language, vad, traditional, ex);
            return TranscribeAudio(audio, language, vad, traditional, cancellationToken, sessionAlreadyStarted: true);
        }
    }

    private RecognitionSessionResult TranscribeAudio(
        float[] audio,
        string language,
        bool vad,
        bool traditional,
        CancellationToken cancellationToken,
        bool sessionAlreadyStarted = false)
    {
        var collector = new TimedTranscriptCollector();
        TranscriptionUpdate latest = new(string.Empty, string.Empty);
        if (!sessionAlreadyStarted)
        {
            BeginSessionWithFallback(language, vad, traditional);
        }

        const int chunkSamples = 8_960;
        for (int start = 0; start < audio.Length; start += chunkSamples)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSamples, audio.Length - start);
            var chunk = new float[count];
            Array.Copy(audio, start, chunk, 0, count);
            TranscriptionUpdate? update = _engine!.Process(chunk);
            if (update is not null)
            {
                latest = update;
                collector.Update(update.RawText, (start + count) / 16_000d);
                PublishUpdate(update, inject: false);
            }
        }

        TranscriptionUpdate? final = _engine!.Flush();
        if (final is not null)
        {
            latest = final;
            collector.Update(final.RawText, audio.Length / 16_000d);
            PublishUpdate(final, inject: false);
        }
        return new(audio, collector.Chunks.ToArray(), latest.RawText, latest.DisplayText);
    }

    private async Task ApplySpeakerLabelsAsync(
        RecognitionSessionResult result,
        bool diarization,
        bool traditional)
    {
        if (!diarization || result.Audio.Length == 0 || string.IsNullOrWhiteSpace(result.RawText))
        {
            TranscriptBox.Text = result.DisplayText;
            StatusText.Text = $"轉錄完成（{CurrentBackendName()}）。";
            return;
        }

        StatusText.Text = "正在以 CPU 分析說話者；較長的錄音需要多一點時間…";
        try
        {
            if (_diarizer is null)
            {
                _diarizer = await Task.Run(
                    () => new SpeakerDiarizer(
                        _diarizationInstaller.SegmentationModelPath,
                        _diarizationInstaller.EmbeddingModelPath),
                    _lifetime.Token);
            }

            IReadOnlyList<SpeakerTimeSegment> segments = await Task.Run(
                () => _diarizer.Process(result.Audio),
                _lifetime.Token);
            TranscriptBox.Text = SpeakerTranscriptFormatter.Format(
                result.Chunks,
                segments,
                result.RawText,
                traditional);
            TranscriptBox.ScrollToEnd();
            int speakerCount = segments.Select(segment => segment.Speaker).Distinct().Count();
            StatusText.Text = $"轉錄與說話者分離完成（ASR：{CurrentBackendName()}；說話者：CPU）。";
            ProgressText.Text = speakerCount > 0
                ? $"偵測到 {speakerCount} 位說話者；標籤為自動推測，可直接編修儲存後的文字檔。"
                : "沒有偵測到可標記的說話片段，已保留完整逐字稿。";
        }
        catch (OperationCanceledException) when (_lifetime.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            // Diarization is an enhancement. Never discard a successful ASR
            // transcript when the optional second pass cannot complete.
            TranscriptBox.Text = result.DisplayText;
            StatusText.Text = "轉錄完成；說話者分離失敗，已保留逐字稿。";
            ProgressText.Text = ex.Message;
        }
    }

    private void FinishBusyOperation()
    {
        _microphone?.Dispose();
        _microphone = null;
        _audioChannel = null;
        _recognitionWorker = null;
        SetBusy(false);
    }

    private string SelectedLanguage()
    {
        return (LanguageBox.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "auto";
    }

    private InferenceBackendPreference SelectedBackendPreference()
    {
        return (BackendBox.SelectedItem as ComboBoxItem)?.Tag?.ToString() switch
        {
            "dml" => InferenceBackendPreference.DirectML,
            "cpu" => InferenceBackendPreference.Cpu,
            _ => InferenceBackendPreference.Auto,
        };
    }

    private void SetBackendBox(InferenceBackendPreference preference)
    {
        string tag = preference switch
        {
            InferenceBackendPreference.DirectML => "dml",
            InferenceBackendPreference.Cpu => "cpu",
            _ => "auto",
        };
        BackendBox.SelectedItem = BackendBox.Items
            .OfType<ComboBoxItem>()
            .First(item => string.Equals(item.Tag?.ToString(), tag, StringComparison.Ordinal));
    }

    private void BeginSessionWithFallback(string language, bool useVad, bool traditional)
    {
        try
        {
            _engine!.BeginSession(language, useVad, traditional);
        }
        catch (Exception ex) when (_engine?.Backend == InferenceBackend.DirectML)
        {
            SwitchToCpu(language, useVad, traditional, ex);
        }
    }

    private void SwitchToCpu(string language, bool useVad, bool traditional, Exception cause)
    {
        _engine?.Dispose();
        _engine = new NemotronEngine(_installer.ModelDirectory, InferenceBackend.Cpu);
        _engine.BeginSession(language, useVad, traditional);
        _backendLoader.InvalidateProfile();
        string reason = cause.GetBaseException().Message.ReplaceLineEndings(" ");
        Dispatcher.BeginInvoke(() =>
        {
            RuntimeSubtitle.Text = "Nemotron 3.5 ASR · CPU 安全回退";
            StatusText.Text = "GPU 推論中斷，已切換 CPU 並繼續辨識。";
            ProgressText.Text = reason;
        });
    }

    private string CurrentBackendName() => _engine is null
        ? "尚未載入"
        : InferenceBackendPolicy.DisplayName(_engine.Backend);

    private void UpdateRuntimePresentation(InferenceLoadResult loaded)
    {
        RuntimeSubtitle.Text = $"Nemotron 3.5 ASR · {InferenceBackendPolicy.DisplayName(loaded.SelectedBackend)} · 說話者分離 CPU";
        ProgressText.Text = loaded.Detail;
    }

    private void SetBusy(bool busy)
    {
        _busy = busy;
        StartButton.IsEnabled = !busy && _engine is not null;
        FileButton.IsEnabled = !busy && _engine is not null;
        PrepareButton.IsEnabled = !busy;
        LanguageBox.IsEnabled = !busy;
        BackendBox.IsEnabled = !busy;
        VadCheck.IsEnabled = !busy;
        TraditionalCheck.IsEnabled = !busy;
        DiarizationCheck.IsEnabled = !busy;
        InjectCheck.IsEnabled = !busy;
        if (!busy)
        {
            StopButton.IsEnabled = false;
        }
    }

    private void ClearButton_Click(object sender, RoutedEventArgs e) => TranscriptBox.Clear();

    private void CopyButton_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(TranscriptBox.Text))
        {
            Clipboard.SetText(TranscriptBox.Text);
        }
    }

    private void SaveButton_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(TranscriptBox.Text))
        {
            return;
        }
        var dialog = new SaveFileDialog
        {
            Title = "儲存逐字稿",
            Filter = "文字檔|*.txt|所有檔案|*.*",
            DefaultExt = ".txt",
            FileName = $"Tippi-{DateTime.Now:yyyyMMdd-HHmmss}.txt",
        };
        if (dialog.ShowDialog(this) == true)
        {
            File.WriteAllText(dialog.FileName, TranscriptBox.Text, new System.Text.UTF8Encoding(false));
        }
    }

    private void OpenModelFolder_Click(object sender, RoutedEventArgs e)
    {
        string directory = ModelsRootDirectory();
        Directory.CreateDirectory(directory);
        Process.Start(new ProcessStartInfo("explorer.exe", directory) { UseShellExecute = true });
    }

    private static string ModelsRootDirectory()
    {
        string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(local, "Tippi", "Models");
    }

    private void ShowError(string title, Exception exception)
    {
        StatusText.Text = title;
        ProgressText.Text = exception.Message;
        MessageBox.Show(this, exception.Message, title, MessageBoxButton.OK, MessageBoxImage.Error);
    }

    private static string FormatBytes(long bytes) => $"{bytes / 1024d / 1024d:F1} MiB";

    private async void Window_Closing(object? sender, CancelEventArgs e)
    {
        if (_shutdownComplete)
        {
            return;
        }
        e.Cancel = true;
        _lifetime.Cancel();
        _microphone?.Stop();
        _audioChannel?.Writer.TryComplete();
        if (_recognitionWorker is not null)
        {
            try { await _recognitionWorker; }
            catch { /* The process is shutting down. */ }
        }
        _microphone?.Dispose();
        _diarizer?.Dispose();
        _engine?.Dispose();
        _diarizationInstaller.Dispose();
        _installer.Dispose();
        _lifetime.Dispose();
        _shutdownComplete = true;
        Close();
    }

    private sealed record RecognitionSessionResult(
        float[] Audio,
        IReadOnlyList<TimedTextChunk> Chunks,
        string RawText,
        string DisplayText);
}
