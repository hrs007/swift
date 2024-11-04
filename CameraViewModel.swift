import SwiftUI
import AVFoundation
import CoreML
import Vision

// MARK: - Models
struct DetectedObject: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

// MARK: - CSV Writer
class CSVWriter {
    private let fileHandle: FileHandle?
    private let headers = ["Frame", "Blue Ball", "Red Ball", "Orange Ball"]
    private var frameCount = 0
    
    init() {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        let filename = "webcam_output_\(timestamp).csv"
        
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = documentsPath.appendingPathComponent(filename)
        
        print("CSV file location: \(fileURL.path)")
        
        try? (headers.joined(separator: ",") + "\n").write(to: fileURL, atomically: true, encoding: .utf8)
        fileHandle = try? FileHandle(forWritingTo: fileURL)
    }
    
    func writeRow(centers: [String: CGPoint]) {
        let row = [
            String(frameCount),
            formatPoint(centers["Blue Ball"]),
            formatPoint(centers["Red Ball"]),
            formatPoint(centers["Orange Ball"])
        ].joined(separator: ",") + "\n"
        
        fileHandle?.write(row.data(using: .utf8) ?? Data())
        frameCount += 1
    }
    
    private func formatPoint(_ point: CGPoint?) -> String {
        guard let point = point else { return "None" }
        return "(\(Int(point.x)), \(Int(point.y)))"
    }
    
    deinit {
        fileHandle?.closeFile()
    }
}

// MARK: - Camera Manager
class CameraManager: NSObject, ObservableObject {
    @Published var previewImage: NSImage?
    @Published var detectedObjects: [DetectedObject] = []
    @Published var isRecording = false
    
    private var captureSession: AVCaptureSession?
    private var videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue")
    private var yoloModel: VNCoreMLModel?
    private var csvWriter: CSVWriter?
    private let maxBBoxSize = CGSize(width: 90, height: 90)
    
    override init() {
        super.init()
        setupCaptureSession()
        setupYOLOModel()
    }
    
    private func setupYOLOModel() {
        do {
            // Print available model URLs for debugging
            if let modelURLs = Bundle.main.urls(forResourcesWithExtension: "mlpackage", subdirectory: nil) {
                print("Available model files: \(modelURLs)")
            }
            
            guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlpackage") else {
                print("Could not find model file in bundle")
                return
            }
            
            print("Loading model from: \(modelURL)")
            
            guard let model = try? MLModel(contentsOf: modelURL) else {
                print("Could not create MLModel from URL")
                return
            }
            
            yoloModel = try VNCoreMLModel(for: model)
            print("Successfully loaded YOLO model")
            
        } catch {
            print("Error loading YOLO model: \(error.localizedDescription)")
        }
    }
    
    private func setupCaptureSession() {
        let session = AVCaptureSession()
        session.sessionPreset = .high
        
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("Failed to setup camera input")
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        
        captureSession = session
    }
    
    func startRecording() {
        csvWriter = CSVWriter()
        isRecording = true
        videoQueue.async { [weak self] in
            self?.captureSession?.startRunning()
        }
    }
    
    func stopRecording() {
        captureSession?.stopRunning()
        csvWriter = nil
        isRecording = false
    }
}

// MARK: - Camera Delegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let yoloModel = self.yoloModel else { return }
        
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        
        let handler = VNImageRequestHandler(cgImage: cgImage)
        let request = VNCoreMLRequest(model: yoloModel) { [weak self] request, error in
            if let error = error {
                print("Detection error: \(error.localizedDescription)")
                return
            }
            
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            self?.processDetections(results, imageSize: CGSize(width: cgImage.width, height: cgImage.height))
        }
        
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform detection: \(error.localizedDescription)")
        }
        
        DispatchQueue.main.async {
            self.previewImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        }
    }
    
    private func processDetections(_ results: [VNRecognizedObjectObservation], imageSize: CGSize) {
        var detectedObjects: [DetectedObject] = []
        var centers: [String: CGPoint] = [:]
        
        for result in results {
            let confidence = result.confidence
            guard let classification = result.labels.first else { continue }
            
            let boundingBox = result.boundingBox
            let size = CGSize(
                width: boundingBox.width * imageSize.width,
                height: boundingBox.height * imageSize.height
            )
            
            if size.width <= maxBBoxSize.width && size.height <= maxBBoxSize.height {
                let center = CGPoint(
                    x: boundingBox.midX * imageSize.width,
                    y: boundingBox.midY * imageSize.height
                )
                
                centers[classification.identifier] = center
                
                detectedObjects.append(DetectedObject(
                    label: classification.identifier,
                    confidence: confidence,
                    boundingBox: boundingBox
                ))
            }
        }
        
        csvWriter?.writeRow(centers: centers)
        
        DispatchQueue.main.async {
            self.detectedObjects = detectedObjects
        }
    }
}

// MARK: - Detection Overlay
struct DetectionOverlay: View {
    let detectedObjects: [DetectedObject]
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                ForEach(detectedObjects) { object in
                    let rect = CGRect(
                        x: object.boundingBox.minX * geometry.size.width,
                        y: (1 - object.boundingBox.maxY) * geometry.size.height,
                        width: object.boundingBox.width * geometry.size.width,
                        height: object.boundingBox.height * geometry.size.height
                    )
                    
                    Rectangle()
                        .stroke(Color.green, lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                    
                    Text("\(object.label): \(String(format: "%.2f", object.confidence))")
                        .background(Color.green)
                        .foregroundColor(.black)
                        .font(.caption)
                        .position(x: rect.minX, y: rect.minY - 10)
                }
            }
        }
    }
}

// MARK: - Content View
struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {
        VStack {
            if let image = cameraManager.previewImage {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(minWidth: 640, minHeight: 480)
                    .overlay(DetectionOverlay(detectedObjects: cameraManager.detectedObjects))
            } else {
                Text("No camera feed")
                    .frame(minWidth: 640, minHeight: 480)
            }
            
            HStack {
                Button(cameraManager.isRecording ? "Stop Recording" : "Start Recording") {
                    if cameraManager.isRecording {
                        cameraManager.stopRecording()
                    } else {
                        cameraManager.startRecording()
                    }
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}

// MARK: - App
@main
struct YOLODetectionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
