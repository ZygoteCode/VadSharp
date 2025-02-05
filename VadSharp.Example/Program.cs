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
    private const bool USE_DIRECT_ML = true;

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

        VadDetector vadDetector = new VadDetector(modelPath, THRESHOLD, SAMPLE_RATE, MIN_SPEECH_DURATION_MS, MAX_SPEECH_DURATION_SECONDS, MIN_SILENCE_DURATION_MS, SPEECH_PAD_MS, USE_DIRECT_ML);
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