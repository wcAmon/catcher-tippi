using Tippi.Windows.Services;

namespace Tippi.Windows;

public partial class MainWindow
{
    private readonly KeywordModelInstaller _keywordInstaller = new();
    private VoiceCommandSpotter? _voiceCommandSpotter;

    private void BeginVoiceCommandSession(bool enabled)
    {
        if (enabled)
        {
            _voiceCommandSpotter?.BeginSession();
        }
    }

    private bool ProcessVoiceCommand(float[] samples, bool enabled)
    {
        return enabled && _voiceCommandSpotter?.Process(samples) == true;
    }

    private bool FinishVoiceCommand(bool enabled)
    {
        return enabled && _voiceCommandSpotter?.Finish() == true;
    }

    private void PublishVoiceCommandDetection(bool inject)
    {
        if (!inject)
        {
            return;
        }
        _injectionCoordinator?.RequestSubmit();
        Dispatcher.BeginInvoke(() =>
        {
            StatusText.Text = "Voice Command 已偵測「幫我送出」，等待 ASR 完成安全尾端後送出";
        });
    }

    private void DisposeVoiceCommand()
    {
        _voiceCommandSpotter?.Dispose();
        _voiceCommandSpotter = null;
        _keywordInstaller.Dispose();
    }
}
