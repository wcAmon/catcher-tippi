using System.Buffers;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace Tippi.Windows.Services;

public sealed record ModelInstallProgress(string FileName, long CompletedBytes, long TotalBytes)
{
    public double Fraction => TotalBytes == 0 ? 0 : (double)CompletedBytes / TotalBytes;
}

public sealed class ModelInstaller : IDisposable
{
    private readonly VerifiedModelInstaller _installer;

    public ModelInstaller()
    {
        _installer = new VerifiedModelInstaller(
            ModelManifest.DefaultDirectory,
            ModelManifest.Files,
            file =>
            {
                string escapedName = Uri.EscapeDataString(file.Name)
                    .Replace("%2F", "/", StringComparison.OrdinalIgnoreCase);
                return $"https://huggingface.co/{ModelManifest.Repository}/resolve/{ModelManifest.Revision}/{escapedName}";
            });
    }

    public string ModelDirectory => _installer.ModelDirectory;
    public bool IsInstalled() => _installer.IsInstalled();
    public Task InstallAsync(IProgress<ModelInstallProgress>? progress, CancellationToken cancellationToken) =>
        _installer.InstallAsync(progress, cancellationToken);
    public Task<bool> VerifyAsync(CancellationToken cancellationToken) =>
        _installer.VerifyAsync(cancellationToken);
    public Task RepairAsync(IProgress<ModelInstallProgress>? progress, CancellationToken cancellationToken) =>
        _installer.RepairAsync(progress, cancellationToken);
    public void Dispose() => _installer.Dispose();
}

internal sealed class VerifiedModelInstaller : IDisposable
{
    private readonly IReadOnlyList<ModelFile> _files;
    private readonly Func<ModelFile, string> _urlForFile;
    private readonly HttpClient _httpClient;

    public VerifiedModelInstaller(
        string modelDirectory,
        IReadOnlyList<ModelFile> files,
        Func<ModelFile, string> urlForFile)
    {
        ModelDirectory = modelDirectory;
        _files = files;
        _urlForFile = urlForFile;
        TotalBytes = files.Sum(file => file.Size);
        _httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("Tippi-Windows/1.0");
    }

    public string ModelDirectory { get; }
    private long TotalBytes { get; }

    public bool IsInstalled()
    {
        return _files.All(IsValidSize);
    }

    public async Task InstallAsync(
        IProgress<ModelInstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(ModelDirectory);
        long completedBeforeCurrent = _files.Where(IsValidSize).Sum(file => file.Size);

        foreach (ModelFile file in _files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (IsValidSize(file))
            {
                progress?.Report(new(file.Name, completedBeforeCurrent, TotalBytes));
                continue;
            }

            string destination = Path.Combine(ModelDirectory, file.Name);
            string partial = destination + ".part";
            await DownloadVerifiedFileAsync(file, partial, completedBeforeCurrent, progress, cancellationToken);

            File.Move(partial, destination, true);
            completedBeforeCurrent += file.Size;
            progress?.Report(new(file.Name, completedBeforeCurrent, TotalBytes));
        }
    }

    public async Task<bool> VerifyAsync(CancellationToken cancellationToken)
    {
        foreach (ModelFile file in _files)
        {
            if (!IsValidSize(file) ||
                !await HasExpectedHashAsync(Path.Combine(ModelDirectory, file.Name), file.Sha256, cancellationToken))
            {
                return false;
            }
        }
        return true;
    }

    public async Task RepairAsync(
        IProgress<ModelInstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(ModelDirectory);
        foreach (ModelFile file in _files)
        {
            string path = Path.Combine(ModelDirectory, file.Name);
            if (File.Exists(path) &&
                (!IsValidSize(file) || !await HasExpectedHashAsync(path, file.Sha256, cancellationToken)))
            {
                File.Delete(path);
            }
        }
        await InstallAsync(progress, cancellationToken);
    }

    private bool IsValidSize(ModelFile file)
    {
        string path = Path.Combine(ModelDirectory, file.Name);
        return File.Exists(path) && new FileInfo(path).Length == file.Size;
    }

    private async Task DownloadFileAsync(
        ModelFile file,
        string partialPath,
        long completedBeforeCurrent,
        IProgress<ModelInstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        long existing = File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0;
        if (existing > file.Size)
        {
            File.Delete(partialPath);
            existing = 0;
        }
        else if (existing == file.Size)
        {
            progress?.Report(new(file.Name, completedBeforeCurrent + existing, TotalBytes));
            return;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, _urlForFile(file));
        if (existing > 0)
        {
            request.Headers.Range = new RangeHeaderValue(existing, null);
        }

        using HttpResponseMessage response = await _httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);
        response.EnsureSuccessStatusCode();

        bool resumed = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        if (!resumed)
        {
            existing = 0;
        }

        await using Stream remote = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var local = new FileStream(
            partialPath,
            resumed ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            1024 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);

        byte[] buffer = ArrayPool<byte>.Shared.Rent(1024 * 1024);
        try
        {
            long current = existing;
            int read;
            while ((read = await remote.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken)
                .AsTask()
                .WaitAsync(TimeSpan.FromMinutes(2), cancellationToken)) > 0)
            {
                await local.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                current += read;
                progress?.Report(new(file.Name, completedBeforeCurrent + current, TotalBytes));
            }
            await local.FlushAsync(cancellationToken);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }

        if (new FileInfo(partialPath).Length != file.Size)
        {
            throw new InvalidDataException($"模型檔案 {file.Name} 大小不符，下載可能不完整。");
        }
    }

    private async Task DownloadVerifiedFileAsync(
        ModelFile file,
        string partialPath,
        long completedBeforeCurrent,
        IProgress<ModelInstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        const int maximumAttempts = 4;
        for (int attempt = 1; attempt <= maximumAttempts; attempt++)
        {
            try
            {
                await DownloadFileAsync(file, partialPath, completedBeforeCurrent, progress, cancellationToken);
                if (await HasExpectedHashAsync(partialPath, file.Sha256, cancellationToken))
                {
                    return;
                }
                File.Delete(partialPath);
                throw new InvalidDataException($"模型檔案 {file.Name} 的 SHA-256 校驗失敗。");
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) when (
                attempt < maximumAttempts &&
                (ex is HttpRequestException || ex is IOException ||
                 ex is TimeoutException || ex is InvalidDataException))
            {
                await Task.Delay(TimeSpan.FromSeconds(attempt * 2), cancellationToken);
            }
        }
        throw new InvalidDataException($"模型檔案 {file.Name} 下載重試後仍然失敗。");
    }

    private static async Task<bool> HasExpectedHashAsync(
        string path,
        string expected,
        CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            1024 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        byte[] hash = await SHA256.HashDataAsync(stream, cancellationToken);
        return Convert.ToHexString(hash).Equals(expected, StringComparison.OrdinalIgnoreCase);
    }

    public void Dispose() => _httpClient.Dispose();
}
