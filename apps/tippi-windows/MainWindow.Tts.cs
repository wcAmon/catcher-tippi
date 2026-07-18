using System.Windows;
using System.Windows.Controls;
using Tippi.Windows.Services;

namespace Tippi.Windows;

public partial class MainWindow
{
    private readonly VoxCpmModelInstaller _ttsInstaller = new();
    private readonly VoxCpmTtsService _ttsService = new();
    private readonly WavPlaybackService _ttsPlayback = new();
    private CancellationTokenSource? _ttsOperation;
    private bool _ttsBusy;

    private void RefreshTtsPresentation()
    {
        bool installed = _ttsInstaller.IsInstalled();
        TtsProgress.Value = installed ? 1 : 0;
        TtsStatusText.Text = installed
            ? "VoxCPM2 Q8 模型已就緒；第一次合成時才載入 runtime"
            : "VoxCPM2 Q8 模型尚未準備";
        TtsProgressText.Text = installed
            ? $"模型位置：{_ttsInstaller.ModelDirectory}"
            : $"需要下載 {FormatBytes(VoxCpmModelManifest.TotalBytes)}，下載可續傳。";
        TtsPrepareButton.Content = installed ? "校驗／修復" : "下載 TTS 模型";
        UpdateTtsControls();
    }

    private async void TtsPrepareButton_Click(object sender, RoutedEventArgs e)
    {
        if (_ttsBusy)
        {
            return;
        }

        BeginTtsOperation();
        try
        {
            TtsStatusText.Text = _ttsInstaller.IsInstalled()
                ? "正在校驗 VoxCPM2 模型…"
                : "正在下載 VoxCPM2 Q8 模型…";
            var progress = new Progress<ModelInstallProgress>(value =>
            {
                TtsProgress.Value = value.Fraction;
                TtsProgressText.Text = $"{value.FileName}  {FormatBytes(value.CompletedBytes)} / {FormatBytes(value.TotalBytes)}";
            });
            if (_ttsInstaller.IsInstalled())
            {
                await _ttsInstaller.RepairAsync(progress, _ttsOperation!.Token);
            }
            else
            {
                await _ttsInstaller.InstallAsync(progress, _ttsOperation!.Token);
            }
            TtsProgress.Value = 1;
            TtsStatusText.Text = "VoxCPM2 Q8 模型已就緒";
            TtsProgressText.Text = $"模型位置：{_ttsInstaller.ModelDirectory}";
        }
        catch (OperationCanceledException)
        {
            TtsStatusText.Text = "TTS 模型作業已停止；下次可從中斷處續傳";
        }
        catch (Exception ex)
        {
            TtsStatusText.Text = "VoxCPM2 模型準備失敗";
            TtsProgressText.Text = ex.Message;
            MessageBox.Show(this, ex.Message, "VoxCPM2 模型準備失敗", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            EndTtsOperation();
            RefreshTtsPresentation();
        }
    }

    private async void TtsSpeakButton_Click(object sender, RoutedEventArgs e)
    {
        string text = TtsInputBox.Text.Trim();
        if (_ttsBusy || !_ttsInstaller.IsInstalled() || text.Length == 0)
        {
            return;
        }

        BeginTtsOperation();
        TtsStopButton.IsEnabled = true;
        try
        {
            var startupProgress = new Progress<string>(message =>
            {
                TtsStatusText.Text = message;
                TtsProgressText.Text = "ASR、Voice Command 與 Diarization 可在載入及合成期間繼續使用。";
            });
            int timesteps = SelectedTtsTimesteps();
            TtsSynthesisResult result = await _ttsService.SynthesizeWithFallbackAsync(
                _ttsInstaller,
                SelectedTtsBackend(),
                text,
                timesteps,
                startupProgress,
                _ttsOperation!.Token);
            TtsStatusText.Text = $"合成完成，正在播放 · {VoxCpmTtsService.DisplayName(result.Backend)}";
            TtsProgressText.Text = $"生成時間 {result.Elapsed.TotalSeconds:F1} 秒 · WAV {FormatBytes(result.WaveData.Length)}";
            await _ttsPlayback.PlayAsync(result.WaveData, _ttsOperation.Token);
            TtsStatusText.Text = $"播放完成 · VoxCPM2 常駐於 {VoxCpmTtsService.DisplayName(result.Backend)}";
        }
        catch (OperationCanceledException)
        {
            TtsStatusText.Text = "TTS 已停止並卸載";
        }
        catch (Exception ex)
        {
            TtsStatusText.Text = "VoxCPM2 合成失敗";
            TtsProgressText.Text = ex.Message;
            MessageBox.Show(this, ex.Message, "VoxCPM2 合成失敗", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            EndTtsOperation();
        }
    }

    private async void TtsStopButton_Click(object sender, RoutedEventArgs e)
    {
        _ttsOperation?.Cancel();
        _ttsPlayback.Stop();
        await _ttsService.StopAsync();
        TtsStatusText.Text = _ttsInstaller.IsInstalled()
            ? "VoxCPM2 已卸載；模型仍保留在磁碟"
            : "VoxCPM2 Q8 模型尚未準備";
        TtsProgressText.Text = _ttsInstaller.IsInstalled()
            ? $"模型位置：{_ttsInstaller.ModelDirectory}"
            : "下載中斷檔會保留，下次可續傳。";
        UpdateTtsControls();
    }

    private TtsBackendPreference SelectedTtsBackend()
    {
        return (TtsBackendBox.SelectedItem as ComboBoxItem)?.Tag?.ToString() switch
        {
            "vulkan" => TtsBackendPreference.Vulkan,
            "cpu" => TtsBackendPreference.Cpu,
            _ => TtsBackendPreference.Auto,
        };
    }

    private int SelectedTtsTimesteps()
    {
        string? value = (TtsQualityBox.SelectedItem as ComboBoxItem)?.Tag?.ToString();
        return int.TryParse(value, out int timesteps) ? timesteps : 6;
    }

    private void BeginTtsOperation()
    {
        _ttsOperation?.Dispose();
        _ttsOperation = CancellationTokenSource.CreateLinkedTokenSource(_lifetime.Token);
        _ttsBusy = true;
        UpdateTtsControls();
    }

    private void EndTtsOperation()
    {
        _ttsOperation?.Dispose();
        _ttsOperation = null;
        _ttsBusy = false;
        UpdateTtsControls();
    }

    private void UpdateTtsControls()
    {
        bool installed = _ttsInstaller.IsInstalled();
        TtsPrepareButton.IsEnabled = !_ttsBusy;
        TtsSpeakButton.IsEnabled = !_ttsBusy && installed;
        TtsBackendBox.IsEnabled = !_ttsBusy;
        TtsQualityBox.IsEnabled = !_ttsBusy;
        TtsStopButton.IsEnabled = _ttsBusy || _ttsService.Backend is not null;
    }

    private async Task ShutdownTtsAsync()
    {
        _ttsOperation?.Cancel();
        _ttsPlayback.Dispose();
        await _ttsService.DisposeAsync();
        _ttsInstaller.Dispose();
        _ttsOperation?.Dispose();
        _ttsOperation = null;
    }
}
