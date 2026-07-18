using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;

namespace Tippi.Windows.Services;

public sealed class KeywordModelInstaller : IDisposable
{
    private readonly HttpClient _httpClient = new() { Timeout = Timeout.InfiniteTimeSpan };

    public KeywordModelInstaller()
    {
        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("Tippi-Windows/1.0");
    }

    public string ModelDirectory => KeywordModelManifest.DefaultDirectory;
    public string EncoderPath => Path.Combine(ModelDirectory, KeywordModelManifest.EncoderName);
    public string DecoderPath => Path.Combine(ModelDirectory, KeywordModelManifest.DecoderName);
    public string JoinerPath => Path.Combine(ModelDirectory, KeywordModelManifest.JoinerName);
    public string TokensPath => Path.Combine(ModelDirectory, KeywordModelManifest.TokensName);
    public string KeywordsPath => Path.Combine(ModelDirectory, KeywordModelManifest.KeywordsName);

    public bool IsInstalled()
    {
        return KeywordModelManifest.Files.All(IsValidSize) && HasCurrentKeywordDefinition();
    }

    public async Task InstallAsync(
        IProgress<ModelInstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (KeywordModelManifest.Files.All(IsValidSize))
        {
            await WriteKeywordDefinitionAsync(cancellationToken);
            return;
        }

        string downloads = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Tippi",
            "Downloads");
        Directory.CreateDirectory(downloads);
        string archive = Path.Combine(downloads, KeywordModelManifest.ArchiveName);
        string partial = archive + ".part";

        if (!File.Exists(archive) || new FileInfo(archive).Length != KeywordModelManifest.ArchiveBytes ||
            !await HasExpectedHashAsync(archive, KeywordModelManifest.ArchiveSha256, cancellationToken))
        {
            if (File.Exists(archive))
            {
                File.Delete(archive);
            }
            await DownloadArchiveAsync(partial, progress, cancellationToken);
            if (!await HasExpectedHashAsync(partial, KeywordModelManifest.ArchiveSha256, cancellationToken))
            {
                File.Delete(partial);
                throw new InvalidDataException("Voice Command 模型壓縮檔的 SHA-256 校驗失敗。");
            }
            File.Move(partial, archive, true);
        }
        else
        {
            progress?.Report(new(
                KeywordModelManifest.ArchiveName,
                KeywordModelManifest.ArchiveBytes,
                KeywordModelManifest.ArchiveBytes));
        }

        await ExtractAndInstallAsync(archive, cancellationToken);
    }

    public async Task RepairAsync(
        IProgress<ModelInstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (await RuntimeFilesAreValidAsync(cancellationToken))
        {
            await WriteKeywordDefinitionAsync(cancellationToken);
            return;
        }

        foreach (ModelFile file in KeywordModelManifest.Files)
        {
            string path = Path.Combine(ModelDirectory, file.Name);
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        await InstallAsync(progress, cancellationToken);
        if (!await RuntimeFilesAreValidAsync(cancellationToken))
        {
            throw new InvalidDataException("Voice Command 模型修復後仍未通過 SHA-256 校驗。");
        }
    }

    private async Task ExtractAndInstallAsync(string archive, CancellationToken cancellationToken)
    {
        string parent = Path.GetDirectoryName(ModelDirectory)
            ?? throw new InvalidOperationException("無法取得 Voice Command 模型目錄。");
        Directory.CreateDirectory(parent);
        string extractionRoot = Path.Combine(parent, $".kws-extract-{Guid.NewGuid():N}");
        string staging = ModelDirectory + ".installing";
        Directory.CreateDirectory(extractionRoot);
        try
        {
            var startInfo = new ProcessStartInfo("tar.exe")
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            startInfo.ArgumentList.Add("-xjf");
            startInfo.ArgumentList.Add(archive);
            startInfo.ArgumentList.Add("-C");
            startInfo.ArgumentList.Add(extractionRoot);
            using Process process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("無法啟動 Windows tar 解壓縮工具。");
            string standardError = await process.StandardError.ReadToEndAsync(cancellationToken);
            await process.WaitForExitAsync(cancellationToken);
            if (process.ExitCode != 0)
            {
                throw new InvalidDataException($"Voice Command 模型解壓縮失敗：{standardError.Trim()}");
            }

            string source = Path.Combine(extractionRoot, KeywordModelManifest.DirectoryName);
            if (!Directory.Exists(source))
            {
                throw new InvalidDataException("Voice Command 模型壓縮檔內缺少預期目錄。");
            }

            if (Directory.Exists(staging))
            {
                Directory.Delete(staging, true);
            }
            Directory.CreateDirectory(staging);
            foreach (ModelFile file in KeywordModelManifest.Files)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string sourcePath = Path.Combine(source, file.Name);
                if (!File.Exists(sourcePath) || new FileInfo(sourcePath).Length != file.Size ||
                    !await HasExpectedHashAsync(sourcePath, file.Sha256, cancellationToken))
                {
                    throw new InvalidDataException($"Voice Command 模型檔案 {file.Name} 校驗失敗。");
                }
                File.Copy(sourcePath, Path.Combine(staging, file.Name), true);
            }
            await File.WriteAllTextAsync(
                Path.Combine(staging, KeywordModelManifest.KeywordsName),
                KeywordModelManifest.KeywordDefinition,
                new UTF8Encoding(false),
                cancellationToken);

            if (Directory.Exists(ModelDirectory))
            {
                Directory.Delete(ModelDirectory, true);
            }
            Directory.Move(staging, ModelDirectory);
        }
        finally
        {
            if (Directory.Exists(extractionRoot))
            {
                Directory.Delete(extractionRoot, true);
            }
            if (Directory.Exists(staging))
            {
                Directory.Delete(staging, true);
            }
        }
    }

    private async Task DownloadArchiveAsync(
        string partialPath,
        IProgress<ModelInstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        long existing = File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0;
        if (existing > KeywordModelManifest.ArchiveBytes)
        {
            File.Delete(partialPath);
            existing = 0;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, KeywordModelManifest.ArchiveUrl);
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
            while ((read = await remote.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken)) > 0)
            {
                await local.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                current += read;
                progress?.Report(new(
                    KeywordModelManifest.ArchiveName,
                    current,
                    KeywordModelManifest.ArchiveBytes));
            }
            await local.FlushAsync(cancellationToken);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }

        if (new FileInfo(partialPath).Length != KeywordModelManifest.ArchiveBytes)
        {
            throw new InvalidDataException("Voice Command 模型壓縮檔大小不符，下載可能不完整。");
        }
    }

    private bool IsValidSize(ModelFile file)
    {
        string path = Path.Combine(ModelDirectory, file.Name);
        return File.Exists(path) && new FileInfo(path).Length == file.Size;
    }

    private bool HasCurrentKeywordDefinition()
    {
        return File.Exists(KeywordsPath) &&
            string.Equals(File.ReadAllText(KeywordsPath), KeywordModelManifest.KeywordDefinition, StringComparison.Ordinal);
    }

    private async Task<bool> RuntimeFilesAreValidAsync(CancellationToken cancellationToken)
    {
        foreach (ModelFile file in KeywordModelManifest.Files)
        {
            string path = Path.Combine(ModelDirectory, file.Name);
            if (!IsValidSize(file) || !await HasExpectedHashAsync(path, file.Sha256, cancellationToken))
            {
                return false;
            }
        }
        return true;
    }

    private Task WriteKeywordDefinitionAsync(CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(ModelDirectory);
        return File.WriteAllTextAsync(
            KeywordsPath,
            KeywordModelManifest.KeywordDefinition,
            new UTF8Encoding(false),
            cancellationToken);
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
