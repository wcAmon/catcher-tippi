using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Tippi.Windows.Services;

public static class AudioFileLoader
{
    public static float[] LoadMono16Khz(string path)
    {
        using var reader = new AudioFileReader(path);
        ISampleProvider source = reader;

        if (source.WaveFormat.Channels == 2)
        {
            source = new StereoToMonoSampleProvider(source);
        }
        else if (source.WaveFormat.Channels != 1)
        {
            source = new DownmixToMonoSampleProvider(source);
        }

        if (source.WaveFormat.SampleRate != 16_000)
        {
            source = new WdlResamplingSampleProvider(source, 16_000);
        }

        var samples = new List<float>();
        var buffer = new float[16_000];
        int read;
        while ((read = source.Read(buffer, 0, buffer.Length)) > 0)
        {
            samples.AddRange(buffer.AsSpan(0, read).ToArray());
        }
        return samples.ToArray();
    }

    private sealed class DownmixToMonoSampleProvider : ISampleProvider
    {
        private readonly ISampleProvider _source;
        private readonly int _channels;
        private float[] _sourceBuffer = [];

        public DownmixToMonoSampleProvider(ISampleProvider source)
        {
            _source = source;
            _channels = source.WaveFormat.Channels;
            WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(source.WaveFormat.SampleRate, 1);
        }

        public WaveFormat WaveFormat { get; }

        public int Read(float[] buffer, int offset, int count)
        {
            int requested = checked(count * _channels);
            if (_sourceBuffer.Length < requested)
            {
                _sourceBuffer = new float[requested];
            }

            int sourceSamples = _source.Read(_sourceBuffer, 0, requested);
            int frames = sourceSamples / _channels;
            for (int frame = 0; frame < frames; frame++)
            {
                float sum = 0;
                int sourceOffset = frame * _channels;
                for (int channel = 0; channel < _channels; channel++)
                {
                    sum += _sourceBuffer[sourceOffset + channel];
                }
                buffer[offset + frame] = sum / _channels;
            }
            return frames;
        }
    }
}
