using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class InferenceBackendPolicyTests
{
    private static readonly BackendProbeResult FastGpu = new(
        InferenceBackend.DirectML,
        true,
        TimeSpan.FromMilliseconds(50));

    private static readonly BackendProbeResult Cpu = new(
        InferenceBackend.Cpu,
        true,
        TimeSpan.FromMilliseconds(100));

    [Fact]
    public void AutoChoosesGpuOnlyWhenItIsMeaningfullyFaster()
    {
        Assert.Equal(
            InferenceBackend.DirectML,
            InferenceBackendPolicy.Select(InferenceBackendPreference.Auto, FastGpu, Cpu));

        var marginalGpu = FastGpu with { Elapsed = TimeSpan.FromMilliseconds(90) };
        Assert.Equal(
            InferenceBackend.Cpu,
            InferenceBackendPolicy.Select(InferenceBackendPreference.Auto, marginalGpu, Cpu));
    }

    [Fact]
    public void FailedGpuAlwaysFallsBackToCpu()
    {
        var failedGpu = new BackendProbeResult(
            InferenceBackend.DirectML,
            false,
            TimeSpan.MaxValue,
            "No DirectML adapter");

        Assert.Equal(
            InferenceBackend.Cpu,
            InferenceBackendPolicy.Select(InferenceBackendPreference.Auto, failedGpu, Cpu));
        Assert.Equal(
            InferenceBackend.Cpu,
            InferenceBackendPolicy.Select(InferenceBackendPreference.DirectML, failedGpu, Cpu));
    }

    [Fact]
    public void ExplicitPreferenceOverridesBenchmarkWhenBackendWorks()
    {
        var slowGpu = FastGpu with { Elapsed = TimeSpan.FromMilliseconds(500) };

        Assert.Equal(
            InferenceBackend.DirectML,
            InferenceBackendPolicy.Select(InferenceBackendPreference.DirectML, slowGpu, Cpu));
        Assert.Equal(
            InferenceBackend.Cpu,
            InferenceBackendPolicy.Select(InferenceBackendPreference.Cpu, FastGpu, Cpu));
    }

    [Fact]
    public void CpuFailureIsFatalBecauseItIsTheSafetyBackend()
    {
        var failedCpu = Cpu with { Succeeded = false, Error = "CPU failed" };

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            InferenceBackendPolicy.Select(InferenceBackendPreference.Auto, FastGpu, failedCpu));

        Assert.Contains("CPU failed", error.Message, StringComparison.Ordinal);
    }
}
