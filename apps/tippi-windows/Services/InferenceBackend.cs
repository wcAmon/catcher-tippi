namespace Tippi.Windows.Services;

public enum InferenceBackendPreference
{
    Auto,
    DirectML,
    Cpu,
}

public enum InferenceBackend
{
    DirectML,
    Cpu,
}

public sealed record BackendProbeResult(
    InferenceBackend Backend,
    bool Succeeded,
    TimeSpan Elapsed,
    string? Error = null);

public static class InferenceBackendPolicy
{
    // DirectML must win by a meaningful margin. Moving tensors between CPU and
    // an integrated GPU can otherwise make a tiny benchmark win disappear in
    // normal streaming use.
    public const double AutoGpuThreshold = 0.85;

    public static InferenceBackend Select(
        InferenceBackendPreference preference,
        BackendProbeResult directMl,
        BackendProbeResult cpu)
    {
        if (!cpu.Succeeded)
        {
            throw new InvalidOperationException(cpu.Error ?? "CPU 語音模型無法載入。");
        }

        if (preference == InferenceBackendPreference.Cpu)
        {
            return InferenceBackend.Cpu;
        }

        if (preference == InferenceBackendPreference.DirectML)
        {
            return directMl.Succeeded ? InferenceBackend.DirectML : InferenceBackend.Cpu;
        }

        if (!directMl.Succeeded)
        {
            return InferenceBackend.Cpu;
        }

        return directMl.Elapsed.TotalMilliseconds
                <= cpu.Elapsed.TotalMilliseconds * AutoGpuThreshold
            ? InferenceBackend.DirectML
            : InferenceBackend.Cpu;
    }

    public static string DisplayName(InferenceBackend backend) => backend switch
    {
        InferenceBackend.DirectML => "GPU（DirectML）",
        _ => "CPU",
    };
}
