using System.IO;
using NAudio.Wave;

namespace Tippi.Windows.Services;

public sealed class WavPlaybackService : IDisposable
{
    private readonly object _gate = new();
    private WaveOutEvent? _output;
    private WaveFileReader? _reader;
    private MemoryStream? _stream;
    private TaskCompletionSource? _completion;

    public Task PlayAsync(byte[] waveData, CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            if (_output is not null)
            {
                throw new InvalidOperationException("已有 TTS 音訊正在播放。");
            }
            _stream = new MemoryStream(waveData, writable: false);
            _reader = new WaveFileReader(_stream);
            _output = new WaveOutEvent();
            _completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            WaveOutEvent current = _output;
            current.PlaybackStopped += (_, args) => CompletePlayback(current, args.Exception);
            current.Init(_reader);
            current.Play();
            TaskCompletionSource completion = _completion;
            cancellationToken.Register(() =>
            {
                completion.TrySetCanceled(cancellationToken);
                Stop();
            });
            return completion.Task;
        }
    }

    public void Stop()
    {
        WaveOutEvent? output;
        lock (_gate)
        {
            output = _output;
        }
        output?.Stop();
    }

    private void CompletePlayback(WaveOutEvent sender, Exception? exception)
    {
        TaskCompletionSource? completion;
        lock (_gate)
        {
            if (!ReferenceEquals(_output, sender))
            {
                return;
            }
            completion = _completion;
            _completion = null;
            _output.Dispose();
            _reader?.Dispose();
            _stream?.Dispose();
            _output = null;
            _reader = null;
            _stream = null;
        }
        if (exception is null)
        {
            completion?.TrySetResult();
        }
        else
        {
            completion?.TrySetException(exception);
        }
    }

    public void Dispose()
    {
        Stop();
        lock (_gate)
        {
            _output?.Dispose();
            _reader?.Dispose();
            _stream?.Dispose();
            _output = null;
            _reader = null;
            _stream = null;
            _completion?.TrySetCanceled();
            _completion = null;
        }
    }
}
