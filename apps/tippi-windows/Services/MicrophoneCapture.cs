using NAudio.Wave;

namespace Tippi.Windows.Services;

public sealed class MicrophoneCapture : IDisposable
{
    private WaveInEvent? _waveIn;

    public event Action<float[]>? SamplesAvailable;
    public event Action<Exception?>? Stopped;

    public void Start()
    {
        if (_waveIn is not null)
        {
            throw new InvalidOperationException("麥克風已經在錄音。");
        }

        _waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(16_000, 16, 1),
            BufferMilliseconds = 40,
            NumberOfBuffers = 4,
        };
        _waveIn.DataAvailable += OnDataAvailable;
        _waveIn.RecordingStopped += OnRecordingStopped;
        _waveIn.StartRecording();
    }

    public void Stop() => _waveIn?.StopRecording();

    private void OnDataAvailable(object? sender, WaveInEventArgs args)
    {
        int sampleCount = args.BytesRecorded / sizeof(short);
        var samples = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            short value = BitConverter.ToInt16(args.Buffer, i * sizeof(short));
            samples[i] = value / 32768f;
        }
        SamplesAvailable?.Invoke(samples);
    }

    private void OnRecordingStopped(object? sender, StoppedEventArgs args)
    {
        WaveInEvent? waveIn = _waveIn;
        _waveIn = null;
        if (waveIn is not null)
        {
            waveIn.DataAvailable -= OnDataAvailable;
            waveIn.RecordingStopped -= OnRecordingStopped;
            waveIn.Dispose();
        }
        Stopped?.Invoke(args.Exception);
    }

    public void Dispose()
    {
        if (_waveIn is null)
        {
            return;
        }
        _waveIn.DataAvailable -= OnDataAvailable;
        _waveIn.RecordingStopped -= OnRecordingStopped;
        _waveIn.Dispose();
        _waveIn = null;
    }
}
