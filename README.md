# VadSharp - Voice Activity Detection in C#.NET

![VadSharp](https://img.shields.io/badge/.NET-9.0-blue.svg) ![ML.NET](https://img.shields.io/badge/ML.NET-Supported-brightgreen.svg) ![ONNXRuntime](https://img.shields.io/badge/ONNXRuntime-Supported-blue.svg) ![DirectML](https://img.shields.io/badge/DirectML-Supported-orange.svg) ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)

## 🚀 The Best VAD Implementation in C#

VadSharp is the **first and most efficient** implementation of **[Silero VAD](https://github.com/snakers4/silero-vad/) in C#**, supporting the latest **V5 model** with all its advanced features. It is faster than the original Python version and runs on **any GPU (NVIDIA, AMD, Intel) and CPU** with **ONNXRuntime and DirectML**.

This project represents **my first significant contribution to the world of artificial intelligence in terms of development**, and I sincerely hope that users will appreciate this effort. Your support and feedback are highly valued! 🙌

---

## 🛠 Features & Benefits

✅ **Stellar Accuracy** - Excellent results in speech detection tasks.

⚡ **Fast** - Processes 30ms+ audio chunks in under **1ms** on a single CPU thread, even faster with batching or GPU acceleration.

📦 **Lightweight** - Model size is only **~2MB**.

🌎 **General** - Trained on **6,000+ languages**, handling various background noises and recording conditions.

🎚 **Flexible Sampling Rate** - Supports **8000 Hz and 16000 Hz**.

🌍 **Highly Portable** - Runs anywhere **ONNX and ML.NET** are available.

🔓 **No Strings Attached** - **MIT License**, no telemetry, no registration, no vendor lock-in.

---

## 📌 Installation

```sh
# Install ONNXRuntime and ML.NET
Install-Package Microsoft.ML.OnnxRuntime
Install-Package Microsoft.ML
Install-Package NAudio
```

---

## 🧑‍💻 Example Usage

```csharp
using System.Text;
using VadSharp;

public class Program
{
    private const int SAMPLE_RATE = 16000;
    private const float THRESHOLD = 0.5f;
    private const int MIN_SPEECH_DURATION_MS = 250;
    private const float MAX_SPEECH_DURATION_SECONDS = float.PositiveInfinity;
    private const int MIN_SILENCE_DURATION_MS = 100;
    private const int SPEECH_PAD_MS = 30;

    public static void Main()
    {
        Console.Title = "VadSharpExample | Made by https://github.com/GabryB03/";

        string modelPath = Path.Combine(AppContext.BaseDirectory, "resources", "silero_vad.onnx");
        string audioPath = Path.Combine(AppContext.BaseDirectory, "resources", "test.wav");

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Model file not found: {modelPath}");
            return;
        }

        if (!File.Exists(audioPath))
        {
            Console.WriteLine($"Audio file not found: {audioPath}");
            return;
        }

        VadDetector vadDetector = new VadDetector(modelPath, THRESHOLD, SAMPLE_RATE, MIN_SPEECH_DURATION_MS, MAX_SPEECH_DURATION_SECONDS, MIN_SILENCE_DURATION_MS, SPEECH_PAD_MS);
        List<VadSpeechSegment> speechTimeList = vadDetector.GetSpeechSegmentList(audioPath);
        StringBuilder sb = new StringBuilder();

        foreach (VadSpeechSegment speechSegment in speechTimeList)
        {
            sb.AppendLine($"[-] Start second: {speechSegment.StartSecond.ToString().Replace(",", ".")}s, end second: {speechSegment.EndSecond.ToString().Replace(",", ".")}s");
        }

        Console.WriteLine(sb.ToString());
        Console.ReadLine();
    }
}
```

---

## 🌟 Contributing

Contributions are welcome there! 🚀 Follow these steps to create a **Pull Request (PR):**

1. **Fork the repository**
2. **Clone your fork**:
   ```sh
   git clone https://github.com/your-username/VadSharp.git
   ```
3. **Create a new branch**:
   ```sh
   git checkout -b feature-branch
   ```
4. **Make your changes & commit**:
   ```sh
   git add .
   git commit -m "Your awesome feature!"
   ```
5. **Push the branch & create a PR**:
   ```sh
   git push origin feature-branch
   ```
6. **Open a PR on GitHub**

---

## 🐛 Issues & Bug Reports

If you find a bug or have a feature request, please **open an issue**:

1. Go to the [Issues Tab](https://github.com/GabryB03/VadSharp/issues).
2. Click on **"New Issue"**.
3. Provide a **clear and concise** description of the problem.
4. If possible, include **screenshots and logs**.

I will review and respond ASAP! 🚀

---

## ✨ Credits

All of my credits go to the original inventor of the [Silero VAD](https://github.com/snakers4/silero-vad/) project,
which has worked hard to the architecture of the algorithm and trained the models!

## 📜 License

VadSharp is licensed under the **MIT License**. Feel free to use, modify, and distribute it as you like!

📌 **Made with ❤️ by [GabryB03](https://github.com/GabryB03/)**

## TODOs

- [ ] Process a float[] buffer instead of only a wave file
- [ ] Use FFMPEG to allow every format to be uploaded
- [ ] Choose if to use DirectML (for GPU processing) or not
- [ ] Batch processing
- [ ] Real-time processing
